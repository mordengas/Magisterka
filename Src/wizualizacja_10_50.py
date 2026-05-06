import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv("wyniki_koncowe_10_50.csv")
df["PoziomUszkodzen"] = df["PoziomUszkodzen"].astype(str)

models = ["NB", "MLP", "XGBoost", "RF"]
available_models = [model for model in models if model in df["Model"].unique()]

sns.set_theme(style="whitegrid")


def ordered_damage_levels(values):
    return sorted(values, key=lambda value: -1 if value == "ORYGINALNY" else int(value.replace("%", "")))


def add_delta_to_baseline(dataframe):
    baseline = (
        dataframe[dataframe["PoziomUszkodzen"] == "ORYGINALNY"][["Dataset", "Model", "AUC_srednia"]]
        .rename(columns={"AUC_srednia": "AUC_bazowe"})
    )
    merged = dataframe.merge(baseline, on=["Dataset", "Model"], how="left")
    merged["Zmiana_vs_baza"] = (merged["AUC_srednia"] - merged["AUC_bazowe"]).round(4)
    return merged


def save_overall_boxplot(dataframe):
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=dataframe, x="Model", y="AUC_srednia", order=available_models)
    plt.title("Rozklad srednich AUC dla modeli - 10_50")
    plt.xlabel("Model")
    plt.ylabel("Srednie AUC")
    plt.tight_layout()
    plt.savefig("rozklad_AUC_10_50.png", dpi=300, bbox_inches="tight")
    plt.show()


def save_dataset_barplots(dataframe):
    baseline = dataframe[dataframe["PoziomUszkodzen"] == "ORYGINALNY"].copy()
    damaged = dataframe[dataframe["PoziomUszkodzen"] != "ORYGINALNY"].copy()

    palette = sns.color_palette("viridis", n_colors=len(available_models))
    model_colors = dict(zip(available_models, palette))

    for dataset_name in sorted(dataframe["Dataset"].dropna().unique()):
        df_ds = damaged[damaged["Dataset"] == dataset_name].copy()
        if df_ds.empty:
            continue

        df_base = baseline[baseline["Dataset"] == dataset_name]
        method_order = sorted(df_ds["Metoda"].unique())
        level_order = sorted(df_ds["PoziomUszkodzen"].unique(), key=lambda value: int(value.replace("%", "")))

        g = sns.catplot(
            data=df_ds,
            kind="bar",
            x="Metoda",
            y="AUC_srednia",
            hue="Model",
            col="PoziomUszkodzen",
            order=method_order,
            col_order=level_order,
            hue_order=available_models,
            palette=palette,
            errorbar=None,
            alpha=0.93,
            height=5,
            aspect=1.12,
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
                    alpha=0.85,
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
            f"Wplyw metod naprawy danych na AUC - {dataset_name.capitalize()} - 10_50",
            fontsize=16,
        )

        plt.savefig(f"wplyw_naprawy_{dataset_name}_10_50.png", dpi=300, bbox_inches="tight")
        plt.show()


def save_delta_plots(dataframe):
    delta_df = add_delta_to_baseline(dataframe)
    delta_df = delta_df[delta_df["PoziomUszkodzen"] != "ORYGINALNY"].copy()

    for dataset_name in sorted(delta_df["Dataset"].dropna().unique()):
        df_ds = delta_df[delta_df["Dataset"] == dataset_name].copy()
        if df_ds.empty:
            continue

        g = sns.relplot(
            data=df_ds,
            kind="line",
            x="PoziomUszkodzen",
            y="Zmiana_vs_baza",
            hue="Metoda",
            style="Model",
            markers=True,
            dashes=False,
            col="Model",
            col_wrap=2,
            hue_order=sorted(df_ds["Metoda"].unique()),
            col_order=available_models,
            facet_kws={"sharey": True, "sharex": True},
            height=4.2,
            aspect=1.25,
        )

        for ax in g.axes.flat:
            ax.axhline(0, linestyle="--", color="black", linewidth=1)
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")

        g.set_axis_labels("Poziom uszkodzen", "Zmiana AUC vs oryginal")
        g.set_titles("{col_name}")
        plt.subplots_adjust(top=0.90)
        g.fig.suptitle(
            f"Zmiana AUC wzgledem oryginalu - {dataset_name.capitalize()} - 10_50",
            fontsize=16,
        )
        plt.savefig(f"delta_auc_{dataset_name}_10_50.png", dpi=300, bbox_inches="tight")
        plt.show()


filtered = df[df["Model"].isin(available_models)].copy()
save_overall_boxplot(filtered)
save_dataset_barplots(filtered)
save_delta_plots(filtered)
