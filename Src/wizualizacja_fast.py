import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv("wyniki_koncowe_fast.csv")
df["PoziomUszkodzen"] = df["PoziomUszkodzen"].astype(str)

models = ["NB", "MLP", "XGBoost", "RF"]
available_models = [model for model in models if model in df["Model"].unique()]

sns.set_theme(style="whitegrid")


def save_overall_boxplot(dataframe):
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=dataframe, x="Model", y="AUC_srednia", order=available_models)
    plt.title("Rozklad srednich AUC dla modeli - fast")
    plt.xlabel("Model")
    plt.ylabel("Srednie AUC")
    plt.tight_layout()
    plt.savefig("rozklad_AUC_fast.png", dpi=300, bbox_inches="tight")
    plt.show()


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
            f"Wplyw metod naprawy danych na AUC - {dataset_name.capitalize()} - fast",
            fontsize=16,
        )

        plt.savefig(f"wplyw_naprawy_{dataset_name}_fast.png", dpi=300, bbox_inches="tight")
        plt.show()


save_overall_boxplot(df[df["Model"].isin(available_models)])
save_dataset_plots(df[df["Model"].isin(available_models)])
