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

# --- Konfiguracja eksperymentu ---

# 1. Definicja zbiorów danych
# Format: (nazwa_folderu, nazwa_kolumny_celu, mapa_mapowania_celu)
DATASETS_INFO = [
    ('zapalenia', 'Zgon', None),  # Zgon jest już 0/1
    ('diabetes', 'decision', {'tested_negative': 0, 'tested_positive': 1}),
    ('serce', 'diagnoza', {1: 0, 2: 1})  # 1=Zdrowy(0), 2=Chory(1) - typowe dla tego zbioru
]

procenty = [15, 25, 50]
metody = ['', '_1_norm', '_2_fill', '_3_remove', '_4_remove_fill', '_5_remove_norm', '_6_fill_norm', '_7_all']
modele = ['RF', 'NB', 'MLP', 'XGBoost']

results = []

# Nagłówek tabeli
print(f"{'PLIK':<45} | {'RF':<7} | {'NB':<7} | {'MLP':<7} | {'XGB':<7}")
print("-" * 85)

# --- GŁÓWNA PĘTLA PO ZBIORACH DANYCH ---
for ds_name, target_col, target_map in DATASETS_INFO:
    
    # === 1. TEST PLIKU PIERWOTNEGO (Baseline) ===
    # Szukamy oryginału. Może być w Data/nazwa.csv lub Data/nazwa/nazwa.csv
    # Dla zapaleń oryginał nazywa się inaczej (zapalenia_naczyn.csv)
    
    orig_filename = f"{ds_name}.csv"
    if ds_name == 'zapalenia':
        orig_filename = 'zapalenia_naczyn.csv'
        
    # Sprawdzamy możliwe lokalizacje oryginału
    possible_paths = [
        f"Data/{orig_filename}",
        f"Data/{ds_name}/{orig_filename}", # Jeśli przeniosłeś oryginał do podfolderu
        orig_filename # Jeśli jest w głównym folderze
    ]
    
    original_file = None
    for p in possible_paths:
        if os.path.exists(p):
            original_file = p
            break
            
    if original_file:
        try:
            # Separator: '|' dla zapaleń, ',' dla reszty
            sep = '|' if 'zapalenia' in original_file else ','
            df = pd.read_csv(original_file, sep=sep)
            
            # Mapowanie celu (jeśli wymagane)
            if target_map:
                df[target_col] = df[target_col].map(target_map)
            
            # Usuwanie braków w celu
            df = df.dropna(subset=[target_col])
            
            y = df[target_col].astype(int)
            X = df.drop(columns=[target_col])
            
            # Usuwamy kolumnę 'Kod' jeśli istnieje (tylko w zapaleniach)
            if 'Kod' in X.columns:
                X = X.drop(columns=['Kod'])
                
            X = pd.get_dummies(X)
            
            # Imputer techniczny -999 dla baseline
            X_clean = SimpleImputer(strategy='constant', fill_value=-999).fit_transform(X)
            
            row = {'Plik': f'{ds_name}_ORYGINALNY'}
            for m_type in modele:
                clf = myclassifier.MyClassifier(model_type=m_type)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
                probs = cross_val_predict(clf, X_clean, y, cv=skf, method='predict_proba')
                
                auc_tool = buildresultauc.BuildResults()
                auc = auc_tool.getResultAUC(probs, y)
                row[m_type] = round(auc, 4)
            
            results.append(row)
            print(f"{row['Plik']:<45} | {row['RF']:.4f}  | {row['NB']:.4f}  | {row['MLP']:.4f}  | {row['XGBoost']:.4f}")
        except Exception as e:
            print(f"BŁĄD w pliku oryginalnym ({ds_name}): {e}")
    else:
        print(f"UWAGA: Nie znaleziono oryginału dla {ds_name}")


    # === 2. TEST PLIKÓW Z PROBLEMAMI (W PODFOLDERACH) ===
    # Pliki leżą teraz w: Data/{ds_name}/{nazwa_pliku}
    
    for p in procenty:
        for m_name in metody:
            file_prefix = ds_name
            filename_only = f'{file_prefix}_prob_{p}{m_name}.csv'
            full_path = f'Data/{ds_name}/{filename_only}'
            
            if not os.path.exists(full_path):
                continue
                
            try:
                # Pliki przetworzone mają ZAWSZE separator '|' (tak ustawiliśmy w generuj_problemy)
                df = pd.read_csv(full_path, sep='|')
                
                # Mapowanie celu
                if target_map:
                    df[target_col] = df[target_col].map(target_map)

                df = df.dropna(subset=[target_col])
                
                y = df[target_col].astype(int)
                X = df.drop(columns=[target_col])
                
                if 'Kod' in X.columns:
                    X = X.drop(columns=['Kod'])

                X = pd.get_dummies(X)

                # Imputer techniczny (-999) dla modeli
                imputer = SimpleImputer(strategy='constant', fill_value=-999)
                X_clean = imputer.fit_transform(X)
                
                row = {'Plik': f"{ds_name}_p{p}{m_name if m_name else '_raw'}"}
                
                for m_type in modele:
                    clf = myclassifier.MyClassifier(model_type=m_type)
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
                    probs = cross_val_predict(clf, X_clean, y, cv=skf, method='predict_proba')
                    
                    auc_tool = buildresultauc.BuildResults()
                    auc = auc_tool.getResultAUC(probs, y)
                    row[m_type] = round(auc, 4)
                
                results.append(row)
                print(f"{row['Plik']:<45} | {row['RF']:.4f}  | {row['NB']:.4f}  | {row['MLP']:.4f}  | {row['XGBoost']:.4f}")
                
            except Exception as e:
                # print(f"BŁĄD w {filename_only}: {str(e)[:50]}") # Opcjonalnie odkomentuj
                pass

# --- Zapis do CSV ---
if results:
    df_res = pd.DataFrame(results)
    df_res.to_csv('wyniki_koncowe.csv', index=False)
    print("\nGotowe! Wyniki zapisano w: wyniki_koncowe.csv")