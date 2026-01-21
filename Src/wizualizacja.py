import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 1. Wczytanie danych
# =========================
df = pd.read_csv("wyniki_koncowe.csv")

models = ['RF', 'NB', 'MLP', 'XGBoost']
available_models = [m for m in models if m in df.columns]

sns.set_theme(style="whitegrid")

# =========================
# WYKRES 1 — BOXPLOT MODELI
# =========================
plt.figure(figsize=(8, 5))
sns.boxplot(data=df[available_models])
plt.title("Rozkład wyników AUC dla poszczególnych modeli")
plt.ylabel("AUC")
plt.xlabel("Model")
plt.tight_layout()
plt.show()

# =========================
# PRZYGOTOWANIE DANYCH
# =========================

# Dataset (zapalenia / serce / diabetes)
df['Dataset'] = df['Plik'].apply(lambda x: x.split('_')[0])

# Oryginały
df_original = df[df['Plik'].str.contains('ORYGINALNY')]

# Dane po uszkodzeniach
df_problems = df[~df['Plik'].str.contains('ORYGINALNY')].copy()

# Melt
df_melted = df_problems.melt(
    id_vars=['Plik', 'Dataset'],
    value_vars=available_models,
    var_name='Model',
    value_name='AUC'
)

# Procent uszkodzeń i metoda
df_melted['Procent'] = df_melted['Plik'].apply(
    lambda x: x.split('_')[1].replace('p', '') + '%'
)
df_melted['Metoda'] = df_melted['Plik'].apply(
    lambda x: '_'.join(x.split('_')[2:])
)

# Paleta spójna dla modeli
palette = sns.color_palette("viridis", n_colors=len(available_models))
model_colors = dict(zip(available_models, palette))

# =========================
# WYKRES 2 — OSOBNO DLA KAŻDEGO DATASETU
# =========================

for dataset_name in ['zapalenia', 'serce', 'diabetes']:

    df_ds = df_melted[df_melted['Dataset'] == dataset_name]
    df_base = df_original[df_original['Dataset'] == dataset_name]

    g = sns.catplot(
        data=df_ds,
        kind="bar",
        x="Metoda",
        y="AUC",
        hue="Model",
        col="Procent",
        palette=palette,
        alpha=0.9,
        height=5,
        aspect=1.2,
        legend_out=True
    )

    # Linie odniesienia (ORYGINALNY)
    for i, ax in enumerate(g.axes.flat):
        for model in available_models:
            base_auc = df_base[model].values[0]
            ax.axhline(
                base_auc,
                linestyle='--',
                linewidth=1.5,
                color=model_colors[model],
                alpha=0.8
            )
            if i == 0:
                ax.text(
                    0.02,
                    base_auc,
                    f" {model}",
                    color=model_colors[model],
                    fontsize=8,
                    fontweight='bold',
                    va='bottom'
                )

    g.set_axis_labels("Metoda naprawy danych", "AUC")
    g.set_titles("Poziom uszkodzeń: {col_name}")
    g.despine(left=True)

    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

    plt.subplots_adjust(top=0.82)
    g.fig.suptitle(
        f"Wpływ metod naprawy danych na AUC — {dataset_name.capitalize()}",
        fontsize=16
    )

    plt.savefig(
        f"wplyw_naprawy_{dataset_name}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
