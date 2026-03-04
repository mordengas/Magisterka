import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import os

# Konfiguracja baz (Oryginalne pliki)
DATASETS = [
    {'plik': 'zapalenia_naczyn.csv', 'cel': 'Zgon', 'sep': '|', 'mapa': None},
    {'plik': 'diabetes.csv', 'cel': 'decision', 'sep': ',', 'mapa': {'tested_negative': 0, 'tested_positive': 1}},
    {'plik': 'serce.csv', 'cel': 'diagnoza', 'sep': ',', 'mapa': {1: 0, 2: 1}}
]

print("=== RANKING NAJWAŻNIEJSZYCH ZMIENNYCH ===\n")

for ds in DATASETS:
    sciezka = f"Data/{ds['plik']}" if os.path.exists(f"Data/{ds['plik']}") else ds['plik']
    
    if not os.path.exists(sciezka):
        print(f"Brak pliku: {sciezka}")
        continue
        
    # Wczytanie
    df = pd.read_csv(sciezka, sep=ds['sep'])
    
    # Mapowanie kolumny decyzyjnej
    if ds['mapa']:
        df[ds['cel']] = df[ds['cel']].map(ds['mapa'])
        
    df = df.dropna(subset=[ds['cel']])
    y = df[ds['cel']].astype(int)
    X = df.drop(columns=[ds['cel']])
    
    # Usuwamy techniczne kolumny, jeśli są (np. 'Kod' w zapaleniach)
    if 'Kod' in X.columns:
        X = X.drop(columns=['Kod'])
        
    # Zamiana tekstów na liczby (jeśli jakieś zostały)
    X = pd.get_dummies(X)
    
    # Szybkie uzupełnienie braków medianą, by RF mógł zadziałać
    imputer = SimpleImputer(strategy='median')
    X_clean = imputer.fit_transform(X)
    
    # Uczenie modelu Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_clean, y)
    
    # Pobieranie wyników ważności (Feature Importances)
    waznosc = rf.feature_importances_
    
    # Tworzenie rankingu
    ranking = pd.DataFrame({
        'Zmienna': X.columns,
        'Waznosc': waznosc
    }).sort_values(by='Waznosc', ascending=False)
    
    # Wyświetlanie Top 5
    print(f"--- Baza: {ds['plik']} ---")
    for idx, row in ranking.head(5).iterrows():
        print(f"  {row['Zmienna']:<25}: {row['Waznosc']:.4f}")
    print("\n")