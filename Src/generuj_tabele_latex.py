"""Generowanie tabel LaTeX i Markdown z wynikami eksperymentu.

Używa pandas to_latex() / to_markdown() zamiast ręcznej konkatenacji stringów.

Użycie:
    python Src/generuj_tabele_latex.py --profile full
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EXPERIMENT_PROFILES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generowanie tabel LaTeX i Markdown z wynikami eksperymentu"
    )
    parser.add_argument(
        "--profile",
        default="full",
        choices=list(EXPERIMENT_PROFILES.keys()),
        help="Profil eksperymentu (domyslnie: full)",
    )
    return parser.parse_args()


def format_auc(mean_val, std_val, is_best=False, latex=True):
    """Formatuje wartość AUC ± std z opcjonalnym pogrubieniem."""
    if latex:
        text = f"{mean_val:.3f} $\\pm$ {std_val:.3f}"
        return f"\\textbf{{{text}}}" if is_best else text
    else:
        text = f"{mean_val:.3f} ± {std_val:.3f}"
        return f"**{text}**" if is_best else text


def build_results_table(sub, available_models, method_order, all_levels, latex=True):
    """Buduje tabelę wyników jako DataFrame z sformatowanymi komórkami."""
    rows = []

    for lvl in all_levels:
        lvl_sub = sub[sub["PoziomUszkodzen"] == lvl]

        # Najlepsze AUC per model (do pogrubienia) — tylko dla uszkodzonych
        max_per_model = {}
        if lvl != "ORYGINALNY":
            for m in available_models:
                m_sub = lvl_sub[lvl_sub["Model"] == m]
                if not m_sub.empty:
                    max_per_model[m] = m_sub["AUC_srednia"].max()

        for meth in method_order:
            if meth == "original" and lvl != "ORYGINALNY":
                continue
            meth_sub = lvl_sub[lvl_sub["Metoda"] == meth]
            if meth_sub.empty and meth != "original":
                continue

            row_data = {"Uszkodzenie": lvl, "Metoda": meth}
            for model_name in available_models:
                cell = meth_sub[meth_sub["Model"] == model_name]
                if not cell.empty:
                    auc = cell["AUC_srednia"].iloc[0]
                    std = cell["AUC_std"].iloc[0]
                    is_best = (auc == max_per_model.get(model_name)) and lvl != "ORYGINALNY"
                    row_data[model_name] = format_auc(auc, std, is_best, latex=latex)
                else:
                    row_data[model_name] = "-"
            rows.append(row_data)

    return pd.DataFrame(rows)


def generate_dataset_tables(df_results, available_models, profile, latex_dir, md_dir):
    """Generuje tabele AUC per zbiór danych."""
    available_datasets = sorted(df_results["Dataset"].unique())

    methods_order = ["original", "raw", "fill", "fill_knn", "remove_fill", "fill_norm", "all", "all_knn"]

    for ds_name in available_datasets:
        sub = df_results[df_results["Dataset"] == ds_name].copy()
        if sub.empty:
            continue

        available_methods = [m for m in methods_order if m in sub["Metoda"].unique() or m == "original"]

        levels = sorted(
            [lvl for lvl in sub["PoziomUszkodzen"].unique() if lvl != "ORYGINALNY"],
            key=lambda s: int(s.replace("%", ""))
        )
        all_levels = ["ORYGINALNY"] + levels

        # --- Tabela LaTeX ---
        table_tex = build_results_table(sub, available_models, available_methods, all_levels, latex=True)
        table_tex = table_tex.set_index(["Uszkodzenie", "Metoda"])

        caption = f"Średnie wartości AUC $\\pm$ odchylenie standardowe dla zbioru \\textbf{{{ds_name.capitalize()}}}"
        latex_str = table_tex.to_latex(
            caption=caption,
            label=f"tab:wyniki_{ds_name}",
            escape=False,
            column_format="ll" + "c" * len(available_models),
            bold_rows=False,
            position="htbp",
        )
        tex_file = latex_dir / f"tabela_wyniki_{ds_name}.tex"
        tex_file.write_text(latex_str, encoding="utf-8")

        # --- Tabela Markdown ---
        table_md = build_results_table(sub, available_models, available_methods, all_levels, latex=False)
        md_str = f"# Wyniki AUC dla zbioru {ds_name.capitalize()}\n\n"
        md_str += table_md.to_markdown(index=False)
        md_file = md_dir / f"tabela_wyniki_{ds_name}.md"
        md_file.write_text(md_str, encoding="utf-8")

    print(f"Zapisano tabele per zbior w {latex_dir} oraz {md_dir}")


def format_p_value(p):
    """Formatuje p-value do wyświetlenia."""
    return f"{p:.4f}" if p >= 0.0001 else "< 0.0001"


def generate_statistical_tables(df_stats, latex_dir, md_dir):
    """Generuje tabele LaTeX z testami Wilcoxona i Friedmana."""
    if df_stats is None or df_stats.empty:
        return

    # --- Tabela Wilcoxona ---
    wilc = df_stats[df_stats["Test"] == "Wilcoxon"].copy()
    if not wilc.empty:
        wilc["p_str"] = wilc["p_value"].apply(format_p_value)
        stars = wilc["Znak_Istotnosci"].fillna("")

        # Wersja LaTeX
        wilc_tex = wilc[["Dataset", "Model", "Metoda", "n", "Statystyka", "p_str"]].copy()
        wilc_tex["n"] = wilc_tex["n"].astype(int)
        wilc_tex["Statystyka"] = wilc_tex["Statystyka"].apply(lambda x: f"{x:.1f}")
        wilc_tex["Istotnosc"] = wilc.apply(
            lambda r: f"\\textbf{{TAK {stars[r.name]}}}" if r["Istotne_005"] == "TAK" else "NIE",
            axis=1
        )
        wilc_tex.columns = ["Zbiór", "Model", "Metoda", "n", "Statystyka W", "p-value", "Istotność"]

        latex_str = wilc_tex.to_latex(
            index=False,
            escape=False,
            caption="Wyniki sparowanego testu rangowych znaków Wilcoxona (Metoda vs. RAW)",
            label="tab:test_wilcoxon",
            position="htbp",
        )
        (latex_dir / "tabela_wilcoxon.tex").write_text(latex_str, encoding="utf-8")

        # Wersja Markdown
        wilc_md = wilc[["Dataset", "Model", "Metoda", "n", "Statystyka", "p_str"]].copy()
        wilc_md["n"] = wilc_md["n"].astype(int)
        wilc_md["Statystyka"] = wilc_md["Statystyka"].apply(lambda x: f"{x:.1f}")
        wilc_md["Istotnosc"] = wilc.apply(
            lambda r: f"**TAK {stars[r.name]}**" if r["Istotne_005"] == "TAK" else "NIE",
            axis=1
        )
        wilc_md.columns = ["Zbiór", "Model", "Metoda", "n", "Statystyka W", "p-value", "Istotność"]

        md_str = "# Wyniki testu Wilcoxona (Metoda vs RAW)\n\n"
        md_str += wilc_md.to_markdown(index=False)
        (md_dir / "tabela_wilcoxon.md").write_text(md_str, encoding="utf-8")

    # --- Tabela Friedmana ---
    fried = df_stats[df_stats["Test"] == "Friedman"].copy()
    if not fried.empty:
        fried["p_str"] = fried["p_value"].apply(format_p_value)

        # Wersja LaTeX
        fried_tex = fried[["Dataset", "Model", "Liczba_metod", "Statystyka", "p_str"]].copy()
        fried_tex["Liczba_metod"] = fried_tex["Liczba_metod"].astype(int)
        fried_tex["Statystyka"] = fried_tex["Statystyka"].apply(lambda x: f"{x:.2f}")
        fried_tex["Istotnosc"] = fried.apply(
            lambda r: "\\textbf{TAK}" if r["Istotne_005"] == "TAK" else "NIE",
            axis=1
        )
        fried_tex.columns = [
            "Zbiór", "Model", "Liczba metod",
            "Statystyka $\\chi^2$", "p-value", "Istotność ($\\alpha=0.05$)",
        ]

        latex_str = fried_tex.to_latex(
            index=False,
            escape=False,
            caption="Wyniki testu rangowego Friedmana dla porównania wszystkich metod łącznie",
            label="tab:test_friedman",
            position="htbp",
        )
        (latex_dir / "tabela_friedman.tex").write_text(latex_str, encoding="utf-8")

        # Wersja Markdown
        fried_md = fried[["Dataset", "Model", "Liczba_metod", "Statystyka", "p_str"]].copy()
        fried_md["Liczba_metod"] = fried_md["Liczba_metod"].astype(int)
        fried_md["Statystyka"] = fried_md["Statystyka"].apply(lambda x: f"{x:.2f}")
        fried_md["Istotnosc"] = fried.apply(
            lambda r: "**TAK**" if r["Istotne_005"] == "TAK" else "NIE",
            axis=1
        )
        fried_md.columns = [
            "Zbiór", "Model", "Liczba metod",
            "Statystyka χ²", "p-value", "Istotność",
        ]

        md_str = "# Wyniki testu Friedmana\n\n"
        md_str += fried_md.to_markdown(index=False)
        (md_dir / "tabela_friedman.md").write_text(md_str, encoding="utf-8")

    print(f"Zapisano tabele statystyczne w {latex_dir} oraz {md_dir}")


def main():
    args = parse_args()

    profile = EXPERIMENT_PROFILES[args.profile]
    suffix = profile["output_suffix"]
    results_dir = PROJECT_ROOT / "Results"

    latex_dir = results_dir / f"Tabele_LaTeX{suffix}"
    latex_dir.mkdir(exist_ok=True)
    md_dir = results_dir / f"Tabele_Markdown{suffix}"
    md_dir.mkdir(exist_ok=True)

    csv_results_path = results_dir / f"wyniki_koncowe{suffix}.csv"
    csv_stats_path = results_dir / f"testy_statystyczne{suffix}.csv"

    if not csv_results_path.exists():
        print(f"Nie znaleziono pliku: {csv_results_path}")
        sys.exit(1)

    df_results = pd.read_csv(csv_results_path)
    df_stats = pd.read_csv(csv_stats_path) if csv_stats_path.exists() else None

    available_models = [m for m in profile["models"] if m in df_results["Model"].unique()]

    print(f"=== GENEROWANIE TABEL LATEX I MARKDOWN (Profil: {args.profile}) ===")
    generate_dataset_tables(df_results, available_models, profile, latex_dir, md_dir)
    generate_statistical_tables(df_stats, latex_dir, md_dir)
    print("[SUKCES] Tabele wygenerowane pomyślnie.")


if __name__ == "__main__":
    main()
