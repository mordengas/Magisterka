import pandas as pd
import numpy as np

# 1. Wczytanie danych (zwróć uwagę na separator '|')
df = pd.read_csv('Data/zapalenia_naczyn.csv', sep='|')

# Ustawienie ziarna losowości dla powtarzalności wyników (opcjonalne)
np.random.seed(42)

# --- PROBLEM 1: PUSTE MIEJSCA (Missing Values) ---
# Wstawiamy NaN (Not a Number) w losowych 10% wierszy dla wybranych kolumn
kolumny_z_brakami = ['Wiek', 'Kreatynina', 'Max_CRP', 'Plec']

for col in kolumny_z_brakami:
    if col in df.columns:
        # Tworzymy maskę losową (10% szans na True)
        maska = np.random.random(len(df)) < 0.1
        df.loc[maska, col] = np.nan

# --- PROBLEM 2: WARTOŚCI ODSTAJĄCE (Outliers) ---
# Wybieramy zmienną 'Max_CRP' i sztucznie zawyżamy wartości dla 5 losowych pacjentów
if 'Max_CRP' in df.columns:
    # Losujemy 5 indeksów
    indeksy_outlierow = np.random.choice(df.index, 5, replace=False)
    # Mnożymy ich wartość CRP przez 50 (tworząc nienaturalne piki)
    df.loc[indeksy_outlierow, 'Max_CRP'] = df.loc[indeksy_outlierow, 'Max_CRP'] * 50

# --- PROBLEM 3: DANE WYMAGAJĄCE NORMALIZACJI (Scaling) ---
# Zmieniamy skalę zmiennej 'Kreatynina'. Mnożymy ją przez 1000.
# Algorytmy ML będą teraz traktować tę cechę jako znacznie ważniejszą niż inne (np. Wiek),
# dopóki dane nie zostaną znormalizowane (np. MinMax lub Standard Scaler).
if 'Kreatynina' in df.columns:
    df['Kreatynina'] = df['Kreatynina'] * 1000

# Zapisanie "zepsutego" zbioru danych do nowego pliku
df.to_csv('Data/zapalenia_naczyn_z_problemami.csv', sep='|', index=False)

# Podgląd zmian
print("Liczba pustych miejsc w kolumnach:")
print(df[kolumny_z_brakami].isnull().sum())
print("\nPrzykładowe dane (zmieniona skala Kreatyniny):")
print(df[['Wiek', 'Kreatynina', 'Max_CRP']].head())