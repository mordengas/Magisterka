import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EXPERIMENT_PROFILES


def load_results(suffix):
    results_path = PROJECT_ROOT / "Results" / f"wyniki_szczegolowe{suffix}.csv"
    if not results_path.exists():
        results_path = PROJECT_ROOT / f"wyniki_szczegolowe{suffix}.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku wynikow: {results_path}")
    return pd.read_csv(results_path)


def wilcoxon_vs_baseline(df, baseline="raw"):
    """
    Ścisły sparowany test Wilcoxona: każda metoda vs baseline (raw).
    Wymusza dokładne parowanie wierszy po (Dataset, Model, PoziomUszkodzen, DamageRepeat, CVRepeat).
    """
    rows = []
    damaged = df[df["PoziomUszkodzen"] != "ORYGINALNY"].copy()
    match_keys = ["Dataset", "Model", "PoziomUszkodzen", "DamageRepeat", "CVRepeat"]

    for dataset in sorted(damaged["Dataset"].unique()):
        for model in sorted(damaged["Model"].unique()):
            df_pair_base = damaged[
                (damaged["Dataset"] == dataset)
                & (damaged["Model"] == model)
                & (damaged["Metoda"] == baseline)
            ][match_keys + ["AUC"]].rename(columns={"AUC": "AUC_base"})

            if df_pair_base.empty:
                continue

            for method in sorted(damaged["Metoda"].unique()):
                if method == baseline:
                    continue

                df_pair_method = damaged[
                    (damaged["Dataset"] == dataset)
                    & (damaged["Model"] == model)
                    & (damaged["Metoda"] == method)
                ][match_keys + ["AUC"]].rename(columns={"AUC": "AUC_method"})

                merged = pd.merge(df_pair_base, df_pair_method, on=match_keys).dropna()
                n = len(merged)
                if n < 5:
                    continue

                # Obliczenie różnic
                diffs = merged["AUC_method"] - merged["AUC_base"]
                if (diffs == 0).all():
                    stat, p = 0.0, 1.0
                else:
                    try:
                        stat, p = wilcoxon(merged["AUC_base"], merged["AUC_method"])
                    except ValueError:
                        continue

                # Znak istotności (*, **, ***)
                stars = ""
                if p < 0.001:
                    stars = "***"
                elif p < 0.01:
                    stars = "**"
                elif p < 0.05:
                    stars = "*"

                rows.append(
                    {
                        "Dataset": dataset,
                        "Model": model,
                        "Baseline": baseline,
                        "Metoda": method,
                        "n": n,
                        "Statystyka": round(stat, 4),
                        "p_value": round(p, 6),
                        "Istotne_005": "TAK" if p < 0.05 else "NIE",
                        "Znak_Istotnosci": stars,
                        "Srednia_Delta": round(float(diffs.mean()), 4),
                    }
                )

    return pd.DataFrame(rows)


def friedman_all_methods(df):
    """
    Test Friedmana: czy zachodzi istotna różnica między wszystkimi metodami naraz.
    Wymusza spójną macierz obserwacji (pełne wiersze we wszystkich metodach).
    """
    rows = []
    damaged = df[df["PoziomUszkodzen"] != "ORYGINALNY"].copy()
    match_keys = ["PoziomUszkodzen", "DamageRepeat", "CVRepeat"]

    for dataset in sorted(damaged["Dataset"].unique()):
        for model in sorted(damaged["Model"].unique()):
            sub = damaged[(damaged["Dataset"] == dataset) & (damaged["Model"] == model)]
            methods = sorted(sub["Metoda"].unique())

            piv = sub.pivot_table(index=match_keys, columns="Metoda", values="AUC").dropna()
            n = len(piv)
            k = len(piv.columns)

            if n < 5 or k < 3:
                continue

            groups = [piv[col].values for col in piv.columns]
            try:
                stat, p = friedmanchisquare(*groups)
            except ValueError:
                continue

            rows.append(
                {
                    "Dataset": dataset,
                    "Model": model,
                    "Liczba_metod": k,
                    "Metody": ", ".join(piv.columns),
                    "n": n,
                    "Statystyka": round(stat, 4),
                    "p_value": round(p, 6),
                    "Istotne_005": "TAK" if p < 0.05 else "NIE",
                }
            )

    return pd.DataFrame(rows)


def plot_critical_difference_diagram(df, suffix, output_dir):
    """
    Rysuje diagram różnic krytycznych (Critical Difference Diagram wg metodyki Demšar 2006).
    """
    damaged = df[df["PoziomUszkodzen"] != "ORYGINALNY"].copy()
    match_keys = ["Dataset", "Model", "PoziomUszkodzen", "DamageRepeat", "CVRepeat"]

    piv = damaged.pivot_table(index=match_keys, columns="Metoda", values="AUC").dropna()
    k = len(piv.columns)
    N = len(piv)

    if k < 2 or N < 5:
        return

    # Rangi (1 = najlepsza metoda, najwyższe AUC)
    ranks = piv.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean().sort_values()

    # Wartość krytyczna Nemenyi (alpha = 0.05)
    q_alpha_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}
    q_alpha = q_alpha_table.get(k, 2.949)
    cd = q_alpha * np.sqrt((k * (k + 1)) / (6 * N))

    fig, ax = plt.subplots(figsize=(11, 4.5))

    # Oś rang
    ax.plot([1, k], [0, 0], color="black", linewidth=2.0)
    for r in range(1, k + 1):
        ax.plot([r, r], [-0.08, 0.08], color="black", linewidth=1.5)
        ax.text(r, -0.22, str(r), ha="center", va="top", fontsize=11, fontweight="bold")

    ax.text(
        (1 + k) / 2, -0.45,
        "Średnia ranga (1 = najwyższa jakość AUC)",
        ha="center", fontsize=12, fontweight="bold"
    )

    # Odcinek różnicy krytycznej (CD bar u góry)
    cd_start = 1.0
    cd_end = 1.0 + cd
    ax.plot([cd_start, cd_end], [0.9, 0.9], color="#d95f02", linewidth=3.0)
    ax.plot([cd_start, cd_start], [0.82, 0.98], color="#d95f02", linewidth=2.0)
    ax.plot([cd_end, cd_end], [0.82, 0.98], color="#d95f02", linewidth=2.0)
    ax.text(
        (cd_start + cd_end) / 2, 1.05,
        f"Różnica krytyczna CD = {cd:.3f} (Nemenyi, α=0.05)",
        ha="center", va="bottom", fontsize=10.5, color="#d95f02", fontweight="bold"
    )

    # Rysowanie znaczników metod i podpisów (naprzemiennie góra / dół lub lewo / prawo)
    n_methods = len(avg_ranks)
    split = (n_methods + 1) // 2

    # Lewa strona (najlepsze metody)
    y_top = 0.65
    for idx, (method, r_val) in enumerate(avg_ranks.iloc[:split].items()):
        y_pos = y_top - idx * 0.16
        ax.plot([r_val, r_val, 0.5], [0, y_pos, y_pos], color="#1b9e77", linewidth=1.4)
        ax.plot(r_val, 0, marker="o", color="#1b9e77", markersize=7)
        ax.text(0.45, y_pos, f"{method} ({r_val:.2f})", ha="right", va="center", fontsize=10.5, fontweight="bold")

    # Prawa strona (gorsze metody)
    for idx, (method, r_val) in enumerate(avg_ranks.iloc[split:].items()):
        y_pos = y_top - idx * 0.16
        ax.plot([r_val, r_val, k + 0.5], [0, y_pos, y_pos], color="#7570b3", linewidth=1.4)
        ax.plot(r_val, 0, marker="o", color="#7570b3", markersize=7)
        ax.text(k + 0.55, y_pos, f"({r_val:.2f}) {method}", ha="left", va="center", fontsize=10.5, fontweight="bold")

    # Grupy statystycznie nierozróżnialne (cliques)
    cliques = []
    sorted_methods = avg_ranks.index.tolist()
    sorted_vals = avg_ranks.values

    for i in range(len(sorted_vals)):
        for j in range(i + 1, len(sorted_vals)):
            if sorted_vals[j] - sorted_vals[i] <= cd:
                cliques.append((sorted_vals[i], sorted_vals[j]))

    # Scalanie i rysowanie linii klik
    clique_y = -0.05
    drawn_cliques = []
    for c_start, c_end in cliques:
        is_sub = any(s <= c_start and e >= c_end for s, e in drawn_cliques if (s != c_start or e != c_end))
        if not is_sub:
            drawn_cliques.append((c_start, c_end))

    for idx, (c_start, c_end) in enumerate(drawn_cliques):
        y_line = 0.15 + idx * 0.08
        ax.plot([c_start, c_end], [y_line, y_line], color="#333333", linewidth=3.5, alpha=0.85)

    ax.set_xlim(-0.8, k + 1.8)
    ax.set_ylim(-0.6, 1.25)
    ax.axis("off")
    plt.title(
        f"Diagram Różnic Krytycznych (CD Diagram) metod czyszczenia danych (N={N})",
        fontsize=13, fontweight="bold", pad=10
    )
    plt.tight_layout()

    out_file = output_dir / f"cd_diagram{suffix}.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Zapisano diagram CD: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Testy statystyczne wynikow eksperymentu")
    parser.add_argument(
        "--profile",
        default="full",
        choices=list(EXPERIMENT_PROFILES.keys()),
        help="Profil eksperymentu (domyslnie: full)",
    )
    args = parser.parse_args()

    profile = EXPERIMENT_PROFILES[args.profile]
    suffix = profile["output_suffix"]

    print(f"=== TESTY STATYSTYCZNE (profil: {args.profile}) ===")
    df = load_results(suffix)
    print(f"Zaladowano {len(df)} wierszy wynikow.")

    print("\n--- Test Wilcoxona (kazda metoda vs. raw) ---")
    wilcoxon_df = wilcoxon_vs_baseline(df, baseline="raw")
    if not wilcoxon_df.empty:
        print(wilcoxon_df.to_string(index=False))
    else:
        print("Za malo danych do testu Wilcoxona.")

    print("\n--- Test Friedmana (wszystkie metody naraz) ---")
    friedman_df = friedman_all_methods(df)
    if not friedman_df.empty:
        print(friedman_df.to_string(index=False))
    else:
        print("Za malo danych do testu Friedmana.")

    results_dir = PROJECT_ROOT / "Results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / f"testy_statystyczne{suffix}.csv"

    combined = pd.concat(
        [
            wilcoxon_df.assign(Test="Wilcoxon") if not wilcoxon_df.empty else pd.DataFrame(),
            friedman_df.assign(Test="Friedman") if not friedman_df.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    if not combined.empty:
        combined.to_csv(output_path, index=False)
        print(f"\nZapisano: {output_path}")

    # Rysowanie diagramu różnic krytycznych (CD Diagram)
    try:
        plot_critical_difference_diagram(df, suffix, results_dir)
        adv_dir = results_dir / "Wykresy_Zaawansowane" / "3_Wykresy_Radarowe"
        if adv_dir.exists():
            plot_critical_difference_diagram(df, suffix, adv_dir)
    except Exception as e:
        print(f"Blad przy generowaniu diagramu CD: {e}")


if __name__ == "__main__":
    main()
