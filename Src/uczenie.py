import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import os

def ocena_metody(sciezka_pliku, nazwa_metody):
    # 1. Wczytanie
    dane = pd.read_csv(sciezka_pliku, sep='|')
    
    # 2. Przygotowanie X i y
    if 'Zgon' not in dane.columns:
        return {"Metoda": nazwa_metody, "Wynik": "Brak kolumny Zgon"}
    
    dane = dane.dropna(subset=['Zgon']) # Bez celu nie da się uczyć
    y = dane['Zgon']
    X = dane.drop(columns=['Zgon', 'Kod'], errors='ignore')
    
    # 3. Kodowanie zmiennych tekstowych (One-Hot)
    X = pd.get_dummies(X, drop_first=True)
    
    # 4. Obsługa braków dla Random Forest
    # Uwaga: Nawet jeśli metoda to "Usuwanie", mogą zostać jakieś NaNs.
    # Wypełniamy je wartością techniczną -999, żeby model zadziałał,
    # ale żeby nie "naprawiać" danych statystycznie (zachowujemy charakter metody).
    imputer = SimpleImputer(strategy='constant', fill_value=-999)
    X_gotowe = imputer.fit_transform(X)
    
    # 5. Podział
    X_train, X_test, y_train, y_test = train_test_split(X_gotowe, y, test_size=0.2, random_state=42)
    
    # 6. Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    predykcje = clf.predict(X_test)
    
    # 7. Metryki
    acc = accuracy_score(y_test, predykcje)
    f1 = f1_score(y_test, predykcje, average='weighted')
    
    return {
        "Metoda": nazwa_metody,
        "Liczba wierszy": len(dane),
        "Dokładność (Accuracy)": round(acc, 4),
        "F1 Score": round(f1, 4)
    }

# Uruchamiamy analizę dla każdego pliku
wyniki = []
wyniki.append(ocena_metody('Data/zapalenia_tylko_znormalizowane.csv', 'Tylko Normalizacja'))
wyniki.append(ocena_metody('Data/zapalenia_tylko_wypelnione.csv', 'Tylko Wypełnianie'))
wyniki.append(ocena_metody('Data/zapalenia_tylko_usuniete_wiersze.csv', 'Tylko Usuwanie Outlierów'))

# Wyświetlenie wyników
wyniki_df = pd.DataFrame(wyniki)
print("\n" + "="*60)
print("PODSUMOWANIE WYNIKÓW:")
print("="*60)
print(wyniki_df.to_string(index=False))
print("="*60)