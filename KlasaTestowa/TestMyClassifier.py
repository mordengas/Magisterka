import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
import buildresultauc 
import myclassifier

# Lista plików do porównania
files_to_compare = [
    'Data/zapalenia_1_norm.csv',
    'Data/zapalenia_2_fill.csv',
    'Data/zapalenia_3_remove.csv',
    'Data/zapalenia_4_remove_fill.csv',
    'Data/zapalenia_5_remove_norm.csv',
    'Data/zapalenia_6_fill_norm.csv',
    'Data/zapalenia_7_all.csv'
]

results = {}

print(f"{'PLIK':<40} | {'AUC':<10}")
print("-" * 55)

for fileName in files_to_compare:
    try:
        # Odczytanie zbioru danych
        dataset = pd.read_csv(fileName, sep='|') 

        # Znajdujemy nazwę ostatniej kolumny (kolumna decyzyjna)
        last_col_name = dataset.columns[-1]
        
        # Usuwamy wiersze, gdzie w kolumnie decyzyjnej jest NaN (pusto)
        # To eliminuje błąd "invalid value encountered in cast"
        dataset = dataset.dropna(subset=[last_col_name])
        
        noColumn = dataset.shape[1]
        
        # Wyodrębnienie cech (odrzucamy pierwszą kolumnę 'Kod' i ostatnią 'Klasa')
        features = dataset.iloc[:, 1:noColumn - 1]
        
        # Wyodrębnienie kolumny decyzyjnej
        labels = dataset.iloc[:, [noColumn - 1]]
        labels = np.ravel(labels)
        
        # WAŻNE: Zamiana na int, aby uniknąć problemu 1.0 != "1"
        # Jeśli w pliku CSV są braki w kolumnie decyzyjnej, trzeba je najpierw usunąć
        if np.issubdtype(labels.dtype, np.floating):
             labels = labels.astype(int)

        # Inicjalizacja klasyfikatora
        model = myclassifier.MyClassifier()

        # Walidacja krzyżowa
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)
        labels_predicted_prob = cross_val_predict(model, features, labels, n_jobs=-1, cv=skf, method='predict_proba')

        # Obliczenie AUC
        buildResults = buildresultauc.BuildResults()
        auc = buildResults.getResultAUC(labels_predicted_prob, labels)
        
        results[fileName] = auc
        print(f"{fileName:<40} | {auc:.4f}")

    except Exception as e:
        print(f"{fileName:<40} | BŁĄD: {e}")

print("-" * 55)

# Znalezienie najlepszego wyniku
if results:
    best_file = max(results, key=results.get)
    print(f"\nNajlepszy wynik (AUC={results[best_file]:.4f}) uzyskano dla pliku:\n-> {best_file}")