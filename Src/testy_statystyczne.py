import argparse
import os
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_results(suffix):
    results_path = PROJECT_ROOT / "Results" / f"wyniki_szczegolowe{suffix}.csv"
    if not results_path.exists():
        results_path = PROJECT_ROOT / f"wyniki_szczegolowe{suffix}.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku wynikow: {results_path}")
    return pd.read_csv(results_path)


def wilcoxon_vs_baseline(df, baseline="raw"):
    """Sparowany test Wilcoxona: kazda metoda vs. baseline."""
    rows = []
    damaged = df[df["PoziomUszkodzen"] != "ORYGINALNY"].copy()

    for dataset in sorted(damaged["Dataset"].unique()):
        for model in sorted(damaged["Model"].unique()):
            base_vals = (
                damaged[
                    (damaged["Dataset"] == dataset)
                    & (damaged["Model"] == model)
                    & (damaged["Metoda"] == baseline)
                ]["AUC"]
                .sort_index()
                .values
            )

            if len(base_vals) < 5:
                continue

            for method in sorted(damaged["Metoda"].unique()):
                if method == baseline:
                    continue

                method_vals = (
                    damaged[
                        (damaged["Dataset"] == dataset)
                        & (damaged["Model"] == model)
                        & (damaged["Metoda"] == method)
                    ]["AUC"]
                    .sort_index()
                    .values
                )

                n = min(len(base_vals), len(method_vals))
                if n < 5:
                    continue

                try:
                    stat, p = wilcoxon(base_vals[:n], method_vals[:n])
                except ValueError:
                    continue

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
                    }
                )

    return pd.DataFrame(rows)


def friedman_all_methods(df):
    """Test Friedmana: czy jest roznica miedzy metodami naraz."""
    rows = []
    damaged = df[df["PoziomUszkodzen"] != "ORYGINALNY"].copy()

    for dataset in sorted(damaged["Dataset"].unique()):
        for model in sorted(damaged["Model"].unique()):
            methods = sorted(damaged["Metoda"].unique())
            groups = []
            valid_methods = []

            for m in methods:
                vals = (
                    damaged[
                        (damaged["Dataset"] == dataset)
                        & (damaged["Model"] == model)
                        & (damaged["Metoda"] == m)
                    ]["AUC"]
                    .sort_index()
                    .values
                )
                if len(vals) >= 5:
                    groups.append(vals)
                    valid_methods.append(m)

            if len(groups) < 3:
                continue

            n = min(len(g) for g in groups)
            try:
                stat, p = friedmanchisquare(*[g[:n] for g in groups])
            except ValueError:
                continue

            rows.append(
                {
                    "Dataset": dataset,
                    "Model": model,
                    "Liczba_metod": len(valid_methods),
                    "Metody": ", ".join(valid_methods),
                    "n": n,
                    "Statystyka": round(stat, 4),
                    "p_value": round(p, 6),
                    "Istotne_005": "TAK" if p < 0.05 else "NIE",
                }
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Testy statystyczne wynikow eksperymentu")
    parser.add_argument("--profile", default="10_50", help="Profil eksperymentu")
    args = parser.parse_args()

    suffix_map = {"full": "", "10_50": "_10_50", "fast": "_fast"}
    suffix = suffix_map.get(args.profile, f"_{args.profile}")

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

    os.makedirs(PROJECT_ROOT / "Results", exist_ok=True)
    output_path = PROJECT_ROOT / "Results" / f"testy_statystyczne{suffix}.csv"

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


if __name__ == "__main__":
    main()
