import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Wczytanie wyników
try:
    df = pd.read_csv('wyniki_koncowe.csv')
except FileNotFoundError:
    print("Błąd: Nie znaleziono pliku wyniki_koncowe.csv. Najpierw uruchom TestMyClassifier.py")
    exit()

# 2. Przekształcenie danych do formatu "długiego" (idealnego dla Seaborn)
# Z formatu: [Plik, RF, SVM, XGBoost] -> [Plik, Model, AUC]
df_melted = df.melt(id_vars=['Plik'], var_name='Model', value_name='AUC')

# 3. Wyciągnięcie informacji o procencie braków i metodzie z nazwy pliku
# Przykład: p15_1_norm -> Proc: 15, Metoda: 1_norm
df_melted['Procent'] = df_melted['Plik'].apply(lambda x: x.split('_')[0].replace('p', '') + '%')
df_melted['Metoda'] = df_melted['Plik'].apply(lambda x: '_'.join(x.split('_')[1:]) if '_' in x else 'raw')

# 4. Ustawienie stylu graficznego
sns.set_theme(style="whitegrid")
plt.figure(figsize=(15, 10))

# 5. Tworzenie wykresu panelowego (FacetGrid)
# Tworzymy osobny wykres dla każdego procentu braków (15, 25, 50)
g = sns.catplot(
    data=df_melted, kind="bar",
    x="Metoda", y="AUC", hue="Model",
    col="Procent", palette="viridis", alpha=.8, height=6, aspect=1.2
)

# 6. Dodanie detali (tytuły, osie)
g.despine(left=True)
g.set_axis_labels("Metoda naprawy danych", "Wynik AUC")
g.set_titles("Poziom trudności: {col_name}")
g.legend.set_title("Model")

# Obrócenie etykiet metod dla lepszej czytelności
for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(45)

plt.subplots_adjust(top=0.85)
g.fig.suptitle('Porównanie skuteczności modeli w zależności od procentu braków i metody czyszczenia', fontsize=16)

# 7. Zapis i wyświetlenie
plt.savefig('porownanie_graficzne.png', dpi=300, bbox_inches='tight')
print("Wykres został zapisany do pliku: porownanie_graficzne.png")
plt.show()