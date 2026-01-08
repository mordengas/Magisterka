import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Wczytanie wyników
try:
    df = pd.read_csv('wyniki_koncowe.csv')
except FileNotFoundError:
    print("Błąd: Nie znaleziono pliku wyniki_koncowe.csv!")
    exit()

# 2. Oddzielenie oryginału od reszty danych
original_mask = df['Plik'].str.contains('ORYGINALNY')
df_original = df[original_mask]
df_problems = df[~original_mask].copy()

# 3. Przekształcenie danych problematycznych (Melt)
df_melted = df_problems.melt(id_vars=['Plik'], var_name='Model', value_name='AUC')

# Wyciąganie Procentu i Metody
df_melted['Procent'] = df_melted['Plik'].apply(lambda x: x.split('_')[0].replace('p', '') + '%')
df_melted['Metoda'] = df_melted['Plik'].apply(lambda x: '_'.join(x.split('_')[1:]) if '_' in x else 'raw')

# 4. Stylizacja
sns.set_theme(style="whitegrid")
g = sns.catplot(
    data=df_melted, kind="bar",
    x="Metoda", y="AUC", hue="Model",
    col="Procent", palette="magma", alpha=.8, height=6, aspect=1.2
)

# 5. DODANIE LINII ODNIESIENIA (Baseline z pliku oryginalnego)
models = ['RF', 'SVM', 'XGBoost']
colors = ['blue', 'orange', 'green'] # Kolory linii dla modeli

for i, ax in enumerate(g.axes.flat):
    for m_idx, model_name in enumerate(models):
        if model_name in df_original.columns:
            base_auc = df_original[model_name].values[0]
            # Rysujemy linię przerywaną dla każdego modelu
            ax.axhline(base_auc, ls='--', color=sns.color_palette("magma", 3)[m_idx], 
                       label=f'Oryginał {model_name}')

# 6. Detale
g.despine(left=True)
g.set_axis_labels("Metoda naprawy", "AUC")
g.set_titles("Poziom braków: {col_name}")

for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(45)

plt.subplots_adjust(top=0.85)
g.fig.suptitle('Skuteczność naprawy danych względem oryginalnego pliku (linie przerywane)', fontsize=16)

plt.savefig('porownanie_z_finalnym_benchmarkiem.png', dpi=300, bbox_inches='tight')
print("Wykres zapisany: porownanie_z_finalnym_benchmarkiem.png")
plt.show()