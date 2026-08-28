import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASETS_ALL

def main():
    print("=== RANKING NAJWAŻNIEJSZYCH ZMIENNYCH ===\n")

    for ds in DATASETS_ALL:
        sciezka = PROJECT_ROOT / ds["original_file"]
        
        if not sciezka.exists():
            print(f"Brak pliku: {sciezka}")
            continue
            
        # Wczytanie
        df = pd.read_csv(sciezka, sep=ds["separator"])
        
        # Mapowanie kolumny decyzyjnej
        target_col = ds["target_col"]
        if ds.get("target_map"):
            df[target_col] = df[target_col].map(ds["target_map"])
            
        df = df.dropna(subset=[target_col])
        y = df[target_col].astype(int)
        X = df.drop(columns=[target_col, *ds.get("drop_columns", [])], errors="ignore")
        
        # Zamiana tekstów na liczby (One-Hot)
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
        print(f"--- Baza: {ds['name']} ({ds['original_file']}) ---")
        for idx, row in ranking.head(5).iterrows():
            print(f"  {row['Zmienna']:<30}: {row['Waznosc']:.4f}")
        print("\n")


if __name__ == '__main__':
    main()