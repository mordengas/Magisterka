'''import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
import buildresultauc 
import myclassifier

# Lista plików do porównania
files_to_compare = [
    'Data/zapalenia_naczyn.csv',
    'Data/zapalenia_naczyn_z_problemami.csv',
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
'''

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.impute import SimpleImputer
import myclassifier
import buildresultauc

# Konfiguracja testów
procenty = [15, 25, 50]
metody = ['', '_1_norm', '_2_fill', '_3_remove', '_4_remove_fill', '_5_remove_norm', '_6_fill_norm', '_7_all']
modele = ['RF', 'SVM', 'XGBoost']

results = []

print(f"{'PLIK':<30} | {'RF':<8} | {'SVM':<8} | {'XGB':<8}")
print("-" * 65)

for p in procenty:
    for m_name in metody:
        fileName = f'Data/zapalenia_prob_{p}{m_name}.csv'
        if not os.path.exists(fileName):
            continue
            
        try:
            df = pd.read_csv(fileName, sep='|')
            
            # 1. Przygotowanie danych (usuwamy puste etykiety)
            last_col = df.columns[-1]
            df = df.dropna(subset=[last_col])
            
            y = df.iloc[:, -1].astype(int)
            X = df.iloc[:, 1:-1] # Pomijamy Kod i Zgon
            X = pd.get_dummies(X)
            
            # 2. Imputacja techniczna (konieczna dla SVM i RF przy brakach)
            imputer = SimpleImputer(strategy='constant', fill_value=-999)
            X_clean = imputer.fit_transform(X)
            
            row = {'Plik': f"p{p}{m_name if m_name else '_raw'}"}
            
            for m_type in modele:
                # Ważne: Tworzymy nowy obiekt klasyfikatora dla każdego modelu
                clf = myclassifier.MyClassifier(model_type=m_type)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
                
                # Generowanie prawdopodobieństw przez walidację krzyżową
                probs = cross_val_predict(clf, X_clean, y, cv=skf, method='predict_proba')
                
                # Obliczanie AUC
                auc_tool = buildresultauc.BuildResults()
                auc = auc_tool.getResultAUC(probs, y)
                row[m_type] = round(auc, 4)
            
            results.append(row)
            print(f"{row['Plik']:<30} | {row['RF']:.4f} | {row['SVM']:.4f} | {row['XGBoost'] if 'XGBoost' in row else row['XGBoost']:.4f}")
            
        except Exception as e:
            print(f"BŁĄD w pliku {fileName}: {str(e)[:100]}")

# Zapis do CSV
if results:
    df_res = pd.DataFrame(results)
    df_res.to_csv('wyniki_koncowe.csv', index=False)
    print("\nGotowe! Wyniki zapisano w: wyniki_koncowe.csv")