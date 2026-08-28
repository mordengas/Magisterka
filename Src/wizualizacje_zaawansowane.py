import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASETS_ALL, EXPERIMENT_PROFILES

parser = argparse.ArgumentParser(description="Generowanie zaawansowanych wykresów do pracy magisterskiej")
parser.add_argument("--profile", default="full", help="Profil eksperymentu (domyslnie: full)")
args = parser.parse_args()

PROFILE = EXPERIMENT_PROFILES.get(args.profile)
if not PROFILE:
    print(f"Brak profilu: {args.profile}")
    sys.exit(1)

suffix = PROFILE["output_suffix"]
results_dir = PROJECT_ROOT / "Results"
adv_dir = results_dir / "Wykresy_Zaawansowane"
adv_dir.mkdir(exist_ok=True)

csv_path = results_dir / f"wyniki_koncowe{suffix}.csv"
if not csv_path.exists():
    print(f"Nie znaleziono pliku wyników: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)
df["PoziomUszkodzen"] = df["PoziomUszkodzen"].astype(str)

available_models = [m for m in PROFILE["models"] if m in df["Model"].unique()]
available_datasets = sorted(df["Dataset"].unique())

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

# Paleta kolorów dla metod
METHOD_PALETTE = {
    "raw": "#d95f02",          # Pomarańczowy / rdza (punkt bazowy uszkodzeń)
    "fill": "#7570b3",         # Fioletowy
    "fill_knn": "#e7298a",     # Różowo-magenta
    "fill_norm": "#66a61e",    # Oliwkowy
    "remove_fill": "#e6ab02",  # Żółto-złoty
    "all": "#1b9e77",          # Ciemnozielony (najsilniejsza metoda standardowa)
    "all_knn": "#1f78b4",      # Głęboki niebieski (najsilniejsza metoda KNN)
}


# ==============================================================================
# 1. KRZYWE ODPORNOŚCI I DEGRADACJI (Robustness / Degradation Curves)
# ==============================================================================
def generate_robustness_curves():
    out_dir = adv_dir / "1_Krzywe_Odpornosci"
    out_dir.mkdir(exist_ok=True)
    print("-> Generowanie 1_Krzywe_Odpornosci...")

    baseline = df[df["PoziomUszkodzen"] == "ORYGINALNY"].copy()
    damaged = df[df["PoziomUszkodzen"] != "ORYGINALNY"].copy()

    levels_sorted = sorted(
        damaged["PoziomUszkodzen"].unique(),
        key=lambda v: int(v.replace("%", ""))
    )
    numeric_levels = [0] + [int(v.replace("%", "")) for v in levels_sorted]

    # A. Wykresy per dataset (panel 2x2 dla 4 modeli)
    for ds_name in available_datasets:
        df_ds = damaged[damaged["Dataset"] == ds_name].copy()
        df_base = baseline[baseline["Dataset"] == ds_name]
        if df_ds.empty:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        methods = [m for m in df_ds["Metoda"].unique()]
        method_order = [m for m in ["raw", "fill", "fill_knn", "remove_fill", "fill_norm", "all", "all_knn"] if m in methods]

        for idx, model_name in enumerate(available_models):
            ax = axes[idx]
            base_val = df_base[df_base["Model"] == model_name]["AUC_srednia"].iloc[0] if not df_base.empty else None

            for method in method_order:
                sub = df_ds[(df_ds["Model"] == model_name) & (df_ds["Metoda"] == method)]
                if sub.empty:
                    continue

                sub_sorted = sub.sort_values("PoziomUszkodzen", key=lambda s: s.str.replace("%", "").astype(int))
                y_vals = [base_val] + sub_sorted["AUC_srednia"].tolist()
                y_errs = [0.0] + sub_sorted["AUC_std"].tolist()

                color = METHOD_PALETTE.get(method, "#333333")
                linestyle = "--" if method == "raw" else "-"
                marker = "o" if "all" in method else ("s" if "knn" in method else "^")

                ax.errorbar(
                    numeric_levels,
                    y_vals,
                    yerr=y_errs,
                    label=method,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.2 if "all" in method or method == "raw" else 1.6,
                    marker=marker,
                    markersize=6,
                    capsize=3,
                    alpha=0.9,
                )

            if base_val is not None:
                ax.axhline(base_val, color="gray", linestyle=":", linewidth=1.2, label="oryginał (0%)")

            ax.set_title(f"Model: {model_name}", fontsize=13, fontweight="bold", pad=8)
            ax.set_xlabel("Poziom uszkodzeń (%)", fontsize=11)
            ax.set_ylabel("Średnie AUC", fontsize=11)
            ax.set_xticks(numeric_levels)
            ax.set_ylim(0.40, 1.0)
            ax.grid(True, linestyle="--", alpha=0.6)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, -0.02),
            frameon=True,
            fontsize=11
        )
        fig.suptitle(f"Krzywe odporności na uszkodzenia danych - {ds_name.capitalize()}", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        plt.savefig(out_dir / f"krzywe_odpornosci_{ds_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    # B. Wykres zbiorczy (uśredniony po wszystkich 5 zbiorach danych)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, model_name in enumerate(available_models):
        ax = axes[idx]
        base_avg = baseline[baseline["Model"] == model_name]["AUC_srednia"].mean() if not baseline.empty else None

        for method in [m for m in ["raw", "fill", "fill_knn", "remove_fill", "fill_norm", "all", "all_knn"] if m in damaged["Metoda"].unique()]:
            sub = damaged[(damaged["Model"] == model_name) & (damaged["Metoda"] == method)]
            if sub.empty:
                continue

            grouped = sub.groupby("PoziomUszkodzen", as_index=False)["AUC_srednia"].mean()
            grouped["level_num"] = grouped["PoziomUszkodzen"].str.replace("%", "").astype(int)
            grouped = grouped.sort_values("level_num")

            y_vals = [base_avg] + grouped["AUC_srednia"].tolist()
            color = METHOD_PALETTE.get(method, "#333333")
            linestyle = "--" if method == "raw" else "-"
            marker = "o" if "all" in method else ("s" if "knn" in method else "^")

            ax.plot(
                numeric_levels,
                y_vals,
                label=method,
                color=color,
                linestyle=linestyle,
                linewidth=2.4 if "all" in method or method == "raw" else 1.7,
                marker=marker,
                markersize=6.5,
                alpha=0.92,
            )

        if base_avg is not None:
            ax.axhline(base_avg, color="gray", linestyle=":", linewidth=1.2, label="oryginał (0%)")

        ax.set_title(f"Model: {model_name} (Średnia z 5 zbiorów)", fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Poziom uszkodzeń (%)", fontsize=11)
        ax.set_ylabel("Średnie AUC", fontsize=11)
        ax.set_xticks(numeric_levels)
        ax.set_ylim(0.50, 0.95)
        ax.grid(True, linestyle="--", alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        fontsize=11
    )
    fig.suptitle("Zagregowane krzywe odporności modeli na uszkodzenia danych", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(out_dir / "krzywe_odpornosci_zbiorcze_srednia.png", dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 2. HEATMAPY ZYSKU Z CZYSZCZENIA (Delta AUC = AUC_metoda - AUC_raw)
# ==============================================================================
def generate_delta_auc_heatmaps():
    out_dir = adv_dir / "2_Heatmapy_Zysku_Delta_AUC"
    out_dir.mkdir(exist_ok=True)
    print("-> Generowanie 2_Heatmapy_Zysku_Delta_AUC...")

    damaged = df[df["PoziomUszkodzen"] != "ORYGINALNY"].copy()

    # Wyliczamy różnicę względem 'raw' dla każdego zestawu (Dataset, PoziomUszkodzen, Model)
    raw_df = damaged[damaged["Metoda"] == "raw"][["Dataset", "PoziomUszkodzen", "Model", "AUC_srednia"]].rename(
        columns={"AUC_srednia": "AUC_raw"}
    )
    merged = pd.merge(damaged, raw_df, on=["Dataset", "PoziomUszkodzen", "Model"])
    merged["Delta_AUC"] = merged["AUC_srednia"] - merged["AUC_raw"]

    methods_clean = [m for m in ["fill", "fill_knn", "remove_fill", "fill_norm", "all", "all_knn"] if m in merged["Metoda"].unique()]

    # A. Heatmapa: Zbiór danych x Metoda naprawy
    piv_dataset = merged[merged["Metoda"].isin(methods_clean)].pivot_table(
        index="Dataset",
        columns="Metoda",
        values="Delta_AUC",
        aggfunc="mean"
    )[methods_clean]

    plt.figure(figsize=(10, 5))
    vmax = max(abs(piv_dataset.min().min()), abs(piv_dataset.max().max()), 0.05)
    sns.heatmap(
        piv_dataset,
        annot=True,
        fmt="+.3f",
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={"label": "Średni zysk Δ AUC (Metoda - RAW)"},
        linewidths=1,
        linecolor="white"
    )
    plt.title("Średni zysk Δ AUC z metod czyszczenia danych w podziale na zbiory", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Metoda naprawy danych", fontsize=11, fontweight="bold")
    plt.ylabel("Zbiór danych", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "heatmapa_delta_auc_dataset_vs_metoda.png", dpi=300, bbox_inches="tight")
    plt.close()

    # B. Heatmapa: Model x Metoda naprawy
    piv_model = merged[merged["Metoda"].isin(methods_clean)].pivot_table(
        index="Model",
        columns="Metoda",
        values="Delta_AUC",
        aggfunc="mean"
    ).reindex(available_models)[methods_clean]

    plt.figure(figsize=(10, 4.5))
    vmax_m = max(abs(piv_model.min().min()), abs(piv_model.max().max()), 0.05)
    sns.heatmap(
        piv_model,
        annot=True,
        fmt="+.3f",
        cmap="vlag",
        center=0,
        vmin=-vmax_m,
        vmax=vmax_m,
        cbar_kws={"label": "Średni zysk Δ AUC (Metoda - RAW)"},
        linewidths=1,
        linecolor="white"
    )
    plt.title("Średni zysk Δ AUC z metod czyszczenia danych w podziale na klasyfikatory", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Metoda naprawy danych", fontsize=11, fontweight="bold")
    plt.ylabel("Model", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "heatmapa_delta_auc_model_vs_metoda.png", dpi=300, bbox_inches="tight")
    plt.close()

    # C. Heatmapa: Poziom uszkodzeń x Metoda naprawy
    piv_lvl = merged[merged["Metoda"].isin(methods_clean)].pivot_table(
        index="PoziomUszkodzen",
        columns="Metoda",
        values="Delta_AUC",
        aggfunc="mean"
    ).reindex(sorted(merged["PoziomUszkodzen"].unique(), key=lambda s: int(s.replace("%", ""))))[methods_clean]

    plt.figure(figsize=(10, 4.5))
    vmax_l = max(abs(piv_lvl.min().min()), abs(piv_lvl.max().max()), 0.05)
    sns.heatmap(
        piv_lvl,
        annot=True,
        fmt="+.3f",
        cmap="vlag",
        center=0,
        vmin=-vmax_l,
        vmax=vmax_l,
        cbar_kws={"label": "Średni zysk Δ AUC (Metoda - RAW)"},
        linewidths=1,
        linecolor="white"
    )
    plt.title("Średni zysk Δ AUC w zależności od stopnia degradacji danych", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Metoda naprawy danych", fontsize=11, fontweight="bold")
    plt.ylabel("Poziom uszkodzeń", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "heatmapa_delta_auc_poziom_uszkodzen.png", dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 3. WYKRESY RADAROWE / PAJĘCZYNOWE (Spider / Radar Charts)
# ==============================================================================
def generate_radar_charts():
    out_dir = adv_dir / "3_Wykresy_Radarowe"
    out_dir.mkdir(exist_ok=True)
    print("-> Generowanie 3_Wykresy_Radarowe...")

    damaged = df[df["PoziomUszkodzen"] != "ORYGINALNY"].copy()

    # A. Radar per Dataset
    piv_ds = damaged.pivot_table(index="Metoda", columns="Dataset", values="AUC_srednia", aggfunc="mean")
    categories_ds = piv_ds.columns.tolist()
    N_ds = len(categories_ds)
    angles_ds = [n / float(N_ds) * 2 * np.pi for n in range(N_ds)]
    angles_ds += angles_ds[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plt.xticks(angles_ds[:-1], [c.capitalize() for c in categories_ds], color="#333333", size=11, fontweight="bold")
    ax.set_rlabel_position(0)
    plt.yticks([0.6, 0.7, 0.8, 0.9], ["0.60", "0.70", "0.80", "0.90"], color="grey", size=9)
    plt.ylim(0.50, 0.95)

    methods_to_plot = [m for m in ["raw", "fill", "remove_fill", "all", "all_knn"] if m in piv_ds.index]
    for method in methods_to_plot:
        values = piv_ds.loc[method].tolist()
        values += values[:1]
        color = METHOD_PALETTE.get(method, "#333333")
        linewidth = 2.4 if "all" in method or method == "raw" else 1.6
        linestyle = "--" if method == "raw" else "-"
        ax.plot(angles_ds, values, linewidth=linewidth, linestyle=linestyle, label=method, color=color)
        if method in ("all", "all_knn"):
            ax.fill(angles_ds, values, color=color, alpha=0.10)

    plt.title("Profil uniwersalności metod czyszczenia danych (Zbiory)", size=14, fontweight="bold", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "radar_metody_per_dataset.png", dpi=300, bbox_inches="tight")
    plt.close()

    # B. Radar per Model
    piv_mod = damaged.pivot_table(index="Metoda", columns="Model", values="AUC_srednia", aggfunc="mean").reindex(columns=available_models)
    categories_mod = piv_mod.columns.tolist()
    N_mod = len(categories_mod)
    angles_mod = [n / float(N_mod) * 2 * np.pi for n in range(N_mod)]
    angles_mod += angles_mod[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plt.xticks(angles_mod[:-1], categories_mod, color="#333333", size=11, fontweight="bold")
    ax.set_rlabel_position(0)
    plt.yticks([0.6, 0.7, 0.8], ["0.60", "0.70", "0.80"], color="grey", size=9)
    plt.ylim(0.50, 0.85)

    for method in methods_to_plot:
        values = piv_mod.loc[method].tolist()
        values += values[:1]
        color = METHOD_PALETTE.get(method, "#333333")
        linewidth = 2.4 if "all" in method or method == "raw" else 1.6
        linestyle = "--" if method == "raw" else "-"
        ax.plot(angles_mod, values, linewidth=linewidth, linestyle=linestyle, label=method, color=color)
        if method in ("all", "all_knn"):
            ax.fill(angles_mod, values, color=color, alpha=0.10)

    plt.title("Profil skuteczności metod czyszczenia danych (Klasyfikatory)", size=14, fontweight="bold", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "radar_metody_per_model.png", dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 4. WYKRESY WAŻNOŚCI CECH (Feature Importance Plots)
# ==============================================================================
def generate_feature_importances():
    out_dir = adv_dir / "4_Waznosc_Cech"
    out_dir.mkdir(exist_ok=True)
    print("-> Generowanie 4_Waznosc_Cech...")

    dataset_rankings = {}

    for ds in DATASETS_ALL:
        ds_name = ds["name"]
        file_path = PROJECT_ROOT / ds["original_file"]
        if not file_path.exists():
            continue

        df_raw = pd.read_csv(file_path, sep=ds["separator"])
        target_col = ds["target_col"]
        if ds["target_map"]:
            df_raw[target_col] = df_raw[target_col].map(ds["target_map"])

        df_clean = df_raw.dropna(subset=[target_col]).copy()
        y = df_clean[target_col].astype(int)
        X = df_clean.drop(columns=[target_col, *ds["drop_columns"]], errors="ignore")
        X = pd.get_dummies(X)

        imputer = SimpleImputer(strategy="median")
        X_imp = imputer.fit_transform(X)

        rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        rf.fit(X_imp, y)

        ranking = pd.DataFrame({
            "Cecha": X.columns,
            "Waznosc": rf.feature_importances_
        }).sort_values(by="Waznosc", ascending=True)

        dataset_rankings[ds_name] = ranking

        # Pojedynczy wykres Top 12 cech
        top12 = ranking.tail(12)
        plt.figure(figsize=(9, 5.5))
        bars = plt.barh(top12["Cecha"], top12["Waznosc"], color="#2c7fb8", alpha=0.88, edgecolor="#1d5a82")
        plt.xlabel("Względna ważność cechy (Gini Importance - Random Forest)", fontsize=10, fontweight="bold")
        plt.title(f"Ranking najważniejszych cech - Zbiór: {ds_name.capitalize()}", fontsize=13, fontweight="bold", pad=12)

        # Dodanie etykiet liczbowych
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.003, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", fontsize=9, color="#222222")

        plt.xlim(0, max(top12["Waznosc"]) * 1.18)
        plt.tight_layout()
        plt.savefig(out_dir / f"waznosc_cech_{ds_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    # B. Panel zbiorczy 5 zbiorów
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, ds_name in enumerate(available_datasets):
        if ds_name not in dataset_rankings:
            continue
        ax = axes[idx]
        top8 = dataset_rankings[ds_name].tail(8)
        bars = ax.barh(top8["Cecha"], top8["Waznosc"], color="#41b6c4", alpha=0.88, edgecolor="#2b8cbe")
        ax.set_title(f"Zbiór: {ds_name.capitalize()}", fontsize=12, fontweight="bold", pad=6)
        ax.set_xlabel("Ważność cechy (Gini)", fontsize=9)
        ax.set_xlim(0, max(top8["Waznosc"]) * 1.22)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.002, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", fontsize=8.5)

    # Ukryj 6. pusty panel
    if len(available_datasets) < len(axes):
        fig.delaxes(axes[-1])

    fig.suptitle("Zestawienie najważniejszych cech klasyfikacyjnych w 5 badanych zbiorach danych", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(out_dir / "waznosc_cech_zbiorczy_panel.png", dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 5. POJEDYNEK: MEDIANA (fill / all) vs KNN (fill_knn / all_knn)
# ==============================================================================
def generate_knn_vs_median_tradeoff():
    out_dir = adv_dir / "5_Pojedynek_Mediana_vs_KNN"
    out_dir.mkdir(exist_ok=True)
    print("-> Generowanie 5_Pojedynek_Mediana_vs_KNN...")

    damaged = df[df["PoziomUszkodzen"] != "ORYGINALNY"].copy()
    if "fill_knn" not in damaged["Metoda"].unique():
        print("Brak metod KNN w bieżącym profilu – pomijanie.")
        return

    # A. fill vs fill_knn
    df_fill = damaged[damaged["Metoda"] == "fill"].rename(columns={"AUC_srednia": "AUC_fill"})
    df_knn = damaged[damaged["Metoda"] == "fill_knn"].rename(columns={"AUC_srednia": "AUC_fill_knn"})
    comp_fill = pd.merge(df_fill, df_knn, on=["Dataset", "PoziomUszkodzen", "Model"])
    comp_fill["Diff_KNN_minus_Median"] = comp_fill["AUC_fill_knn"] - comp_fill["AUC_fill"]

    # B. all vs all_knn
    df_all = damaged[damaged["Metoda"] == "all"].rename(columns={"AUC_srednia": "AUC_all"})
    df_all_knn = damaged[damaged["Metoda"] == "all_knn"].rename(columns={"AUC_srednia": "AUC_all_knn"})
    comp_all = pd.merge(df_all, df_all_knn, on=["Dataset", "PoziomUszkodzen", "Model"])
    comp_all["Diff_KNN_minus_Median"] = comp_all["AUC_all_knn"] - comp_all["AUC_all"]

    # Wykres 1: Różnica per zbiór i model (słupkowy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    # Subplot 1: Sama imputacja (fill_knn vs fill)
    mean_diff_fill = comp_fill.groupby("Dataset")["Diff_KNN_minus_Median"].mean().reset_index()
    colors1 = ["#2ca02c" if val >= 0 else "#d62728" for val in mean_diff_fill["Diff_KNN_minus_Median"]]
    bars1 = ax1.bar(mean_diff_fill["Dataset"], mean_diff_fill["Diff_KNN_minus_Median"], color=colors1, alpha=0.85, edgecolor="#333333")
    ax1.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax1.set_title("Sama imputacja: fill_knn vs fill (Mediana)", fontsize=12, fontweight="bold", pad=8)
    ax1.set_ylabel("Różnica Δ AUC (KNN - Mediana)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Zbiór danych", fontsize=11, fontweight="bold")
    for bar in bars1:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        ax1.text(bar.get_x() + bar.get_width() / 2, h + (0.001 if h >= 0 else -0.003), f"{h:+.3f}", ha="center", va=va, fontsize=9.5, fontweight="bold")

    # Subplot 2: Pełny pipeline (all_knn vs all)
    mean_diff_all = comp_all.groupby("Dataset")["Diff_KNN_minus_Median"].mean().reset_index()
    colors2 = ["#2ca02c" if val >= 0 else "#d62728" for val in mean_diff_all["Diff_KNN_minus_Median"]]
    bars2 = ax2.bar(mean_diff_all["Dataset"], mean_diff_all["Diff_KNN_minus_Median"], color=colors2, alpha=0.85, edgecolor="#333333")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax2.set_title("Kompleksowy pipeline: all_knn vs all (Mediana)", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Zbiór danych", fontsize=11, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        ax2.text(bar.get_x() + bar.get_width() / 2, h + (0.001 if h >= 0 else -0.003), f"{h:+.3f}", ha="center", va=va, fontsize=9.5, fontweight="bold")

    fig.suptitle("Analiza zysku z zaawansowanej imputacji KNN względem prostej mediany", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(out_dir / "pojedynek_mediana_vs_knn.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Wykres 2: Różnica KNN vs Mediana w funkcji poziomu uszkodzeń
    plt.figure(figsize=(10, 5))
    lvl_diff_fill = comp_fill.groupby("PoziomUszkodzen")["Diff_KNN_minus_Median"].mean().reindex(
        sorted(comp_fill["PoziomUszkodzen"].unique(), key=lambda s: int(s.replace("%", "")))
    )
    lvl_diff_all = comp_all.groupby("PoziomUszkodzen")["Diff_KNN_minus_Median"].mean().reindex(
        sorted(comp_all["PoziomUszkodzen"].unique(), key=lambda s: int(s.replace("%", "")))
    )

    x_vals = [int(s.replace("%", "")) for s in lvl_diff_fill.index]
    plt.plot(x_vals, lvl_diff_fill.values, marker="o", linewidth=2.4, color="#e7298a", label="Sama imputacja (fill_knn vs fill)")
    plt.plot(x_vals, lvl_diff_all.values, marker="s", linewidth=2.4, color="#1f78b4", label="Pełny proces (all_knn vs all)")
    plt.axhline(0, color="black", linestyle="--", linewidth=1.2)
    plt.title("Zysk z imputacji KNN w zależności od stopnia uszkodzenia danych", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Poziom uszkodzeń (%)", fontsize=11, fontweight="bold")
    plt.ylabel("Średnia różnica Δ AUC (KNN - Mediana)", fontsize=11, fontweight="bold")
    plt.xticks(x_vals)
    plt.legend(frameon=True, fontsize=10.5)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "roznica_knn_minus_mediana_per_poziom.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print(f"=== GENEROWANIE ZAAWANSOWANYCH WIZUALIZACJI (Profil: {args.profile}) ===")
    generate_robustness_curves()
    generate_delta_auc_heatmaps()
    generate_radar_charts()
    generate_feature_importances()
    generate_knn_vs_median_tradeoff()
    print(f"\n[SUKCES] Wszystkie zaawansowane wykresy zapisano w: {adv_dir}")


if __name__ == "__main__":
    main()
