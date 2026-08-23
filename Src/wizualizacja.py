import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EXPERIMENT_PROFILES

import numpy as np

parser = argparse.ArgumentParser(description="Wizualizacja wyników eksperymentu")
parser.add_argument("--profile", default="10_50", help="Profil eksperymentu")
parser.add_argument("--ylim-min", type=float, default=None, help="Minimalna wartosc osi Y dla AUC (domyslnie: automatycznie)")
parser.add_argument("--ylim-max", type=float, default=1.0, help="Maksymalna wartosc osi Y dla AUC (domyslnie: 1.0)")
args = parser.parse_args()

PROFILE = EXPERIMENT_PROFILES.get(args.profile)
if not PROFILE:
    print(f"Brak profilu: {args.profile}")
    sys.exit(1)

suffix = PROFILE["output_suffix"]

results_dir = PROJECT_ROOT / "Results"
results_dir.mkdir(exist_ok=True)

csv_path = results_dir / f"wyniki_koncowe{suffix}.csv"
if not csv_path.exists():
    print(f"Nie znaleziono pliku: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)
df["PoziomUszkodzen"] = df["PoziomUszkodzen"].astype(str)

models = PROFILE["models"]
available_models = [model for model in models if model in df["Model"].unique()]

sns.set_theme(style="whitegrid")


def get_ylim_range(df_data, df_base=None, custom_min=None, custom_max=None):
    y_max = custom_max if custom_max is not None else 1.0
    if custom_min is not None:
        return custom_min, y_max

    min_val = df_data["AUC_srednia"].min()
    if "AUC_std" in df_data.columns:
        min_with_std = (df_data["AUC_srednia"] - df_data["AUC_std"]).min()
        if pd.notna(min_with_std):
            min_val = min(min_val, min_with_std)

    if df_base is not None and not df_base.empty and "AUC_srednia" in df_base.columns:
        min_val = min(min_val, df_base["AUC_srednia"].min())

    margin = 0.03
    auto_min = max(0.0, np.floor((min_val - margin) * 20) / 20)
    return auto_min, y_max


def save_overall_boxplot(dataframe):
    y_min, y_max = get_ylim_range(dataframe, custom_min=args.ylim_min, custom_max=args.ylim_max)
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=dataframe, x="Model", y="AUC_srednia", order=available_models)
    plt.ylim(y_min, y_max)
    plt.title("Rozklad srednich AUC dla modeli")
    plt.xlabel("Model")
    plt.ylabel("Srednie AUC")
    plt.tight_layout()
    plt.savefig(results_dir / f"rozklad_AUC{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_dataset_plots(dataframe):
    baseline = dataframe[dataframe["PoziomUszkodzen"] == "ORYGINALNY"].copy()
    damaged = dataframe[dataframe["PoziomUszkodzen"] != "ORYGINALNY"].copy()

    palette = sns.color_palette("viridis", n_colors=len(available_models))
    model_colors = dict(zip(available_models, palette))

    for dataset_name in sorted(dataframe["Dataset"].dropna().unique()):
        df_ds = damaged[damaged["Dataset"] == dataset_name].copy()
        if df_ds.empty:
            continue

        df_base = baseline[baseline["Dataset"] == dataset_name]
        y_min, y_max = get_ylim_range(df_ds, df_base=df_base, custom_min=args.ylim_min, custom_max=args.ylim_max)

        method_order = sorted(df_ds["Metoda"].unique())
        g = sns.catplot(
            data=df_ds,
            kind="bar",
            x="Metoda",
            y="AUC_srednia",
            hue="Model",
            col="PoziomUszkodzen",
            order=method_order,
            col_order=sorted(df_ds["PoziomUszkodzen"].unique(), key=lambda value: int(value.replace("%", ""))),
            hue_order=available_models,
            palette=palette,
            errorbar=None,
            alpha=0.92,
            height=5,
            aspect=1.15,
            legend_out=True,
        )

        for ax, (_, facet_df) in zip(g.axes.flat, df_ds.groupby("PoziomUszkodzen", sort=True)):
            facet_df = facet_df.copy()
            facet_df["Metoda"] = pd.Categorical(facet_df["Metoda"], categories=method_order, ordered=True)
            facet_df["Model"] = pd.Categorical(facet_df["Model"], categories=available_models, ordered=True)
            facet_df = facet_df.sort_values(["Metoda", "Model"])
            for patch, (_, row) in zip(ax.patches, facet_df.iterrows()):
                center_x = patch.get_x() + patch.get_width() / 2
                ax.errorbar(
                    x=center_x,
                    y=row["AUC_srednia"],
                    yerr=row["AUC_std"],
                    color="black",
                    linewidth=1,
                    capsize=3,
                )

        for i, ax in enumerate(g.axes.flat):
            ax.set_ylim(y_min, y_max)
            for model_name in available_models:
                base_row = df_base[df_base["Model"] == model_name]
                if base_row.empty:
                    continue

                base_auc = base_row["AUC_srednia"].iloc[0]
                ax.axhline(
                    base_auc,
                    linestyle="--",
                    linewidth=1.4,
                    color=model_colors[model_name],
                    alpha=0.9,
                )
                if i == 0:
                    ax.text(
                        0.02,
                        base_auc,
                        f" {model_name}",
                        color=model_colors[model_name],
                        fontsize=8,
                        fontweight="bold",
                        va="bottom",
                    )

        g.set_axis_labels("Metoda naprawy danych", "Srednie AUC")
        g.set_titles("Poziom uszkodzen: {col_name}")
        g.despine(left=True)

        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")

        plt.subplots_adjust(top=0.82)
        g.fig.suptitle(
            f"Wplyw metod naprawy danych na AUC - {dataset_name.capitalize()}",
            fontsize=16,
        )

        plt.savefig(results_dir / f"wplyw_naprawy_{dataset_name}{suffix}.png", dpi=300, bbox_inches="tight")
        plt.close()


save_overall_boxplot(df[df["Model"].isin(available_models)])
save_dataset_plots(df[df["Model"].isin(available_models)])
