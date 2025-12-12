import pandas as pd
import numpy as np

# 1. Wczytanie danych
try:
    df = pd.read_csv('Data/zapalenia_naczyn.csv', sep='|')
except FileNotFoundError:
    # Fallback, gdyby plik był w bieżącym katalogu
    df = pd.read_csv('zapalenia_naczyn.csv', sep='|')
    
np.random.seed(42) # Dla powtarzalności

# --- PROBLEM 1: LOSOWE BRAKI DANYCH (Globalnie ~5%, bez kolumny decyzyjnej) ---
procent_brakow = 0.05
maska_losowa = np.random.random(df.shape) < procent_brakow

# Zabezpieczenie kolumny decyzyjnej (ostatniej) przed NaN
maska_losowa[:, -1] = False
df[maska_losowa] = np.nan

# --- NOWOŚĆ: PROBLEM 1.5 - AUTOMATYCZNE OUTLIERY W KAŻDEJ KOLUMNIE LICZBOWEJ ---
# Cel: "W każdej kolumnie liczbowej (nie binarnej) w 5% komórek dodaj wartości mocno odstające"

numeric_cols = df.select_dtypes(include=[np.number]).columns
procent_outlierow = 0.05

print("Generowanie outlierów w kolumnach liczbowych...")

for col in numeric_cols:
    # Pomijamy kolumny binarne (które mają tylko 2 unikalne wartości, np. 0 i 1, ignorując NaN)
    # Dzięki temu nie psujemy flag typu 'Plec' czy 'Zgon'
    unikalne_wartosci = df[col].dropna().unique()
    
    if len(unikalne_wartosci) > 2:
        # To jest kolumna liczbowa/ciągła (lub kategoryczna o wielu poziomach)
        
        # Wybieramy 5% indeksów, które nie są puste (NaN)
        dostepne_indeksy = df[df[col].notna()].index
        if len(dostepne_indeksy) > 0:
            liczba_do_zmiany = int(len(dostepne_indeksy) * procent_outlierow)
            idx_outliers = np.random.choice(dostepne_indeksy, liczba_do_zmiany, replace=False)
            
            # Tworzymy outliery.
            # Strategia: Mnożymy przez 50 (jak w Twoim przykładzie) LUB dodajemy dużą wartość,
            # aby upewnić się, że zera też staną się outlierami.
            # Używamy wartości bezwzględnej max z kolumny jako bazy do przesunięcia.
            max_val = df[col].abs().max()
            if max_val == 0: max_val = 100 # Zabezpieczenie dla kolumn z samymi zerami
            
            # Wzór: Wartość * 50 + (Max_kolumny * 2) - to gwarantuje silne odstawanie
            # Dla pewności rzutujemy na typ kolumny (żeby nie było błędów float w kolumnach int)
            nowe_wartosci = df.loc[idx_outliers, col] * 50 + (max_val * 2)
            
            df.loc[idx_outliers, col] = nowe_wartosci
            # print(f" - {col}: zmieniono {liczba_do_zmiany} rekordów.")

# --- PROBLEM 2: SPECIFICZNE BŁĘDY LOGICZNE (NADCHODZI PO AUTOMATYCZNYCH) ---
# Te zmiany są "ręczne" i mają specyficzny charakter (np. ujemny wiek), więc wykonujemy je PO pętli,
# aby mieć pewność, że te konkretne przypadki (np. -100 lat) się pojawią.

# A. Max_CRP (Dodatkowe ręczne, ekstremalne piki)
if 'Max_CRP' in df.columns:
    idx = np.random.choice(df[df['Max_CRP'].notna()].index, 5, replace=False)
    df.loc[idx, 'Max_CRP'] = df.loc[idx, 'Max_CRP'] * 50

# B. Wiek (Specyficzne błędy logiczne: ujemne, niemożliwe)
if 'Wiek' in df.columns:
    idx_wiek = np.random.choice(df[df['Wiek'].notna()].index, 5, replace=False)
    # -100, 100000 (ok. 273 lata), 50 (błąd jednostki), itp.
    bledne_wartosci = [-100, 100000, 50, 150000, -5000]
    df.loc[idx_wiek, 'Wiek'] = bledne_wartosci

# C. Sterydy (Ogromna stała)
if 'Sterydy_Dawka_mg' in df.columns:
    idx_st = np.random.choice(df[df['Sterydy_Dawka_mg'].notna()].index, 3, replace=False)
    df.loc[idx_st, 'Sterydy_Dawka_mg'] = 50000 

# --- PROBLEM 3: TRUDNIEJSZA NORMALIZACJA (ZMIA SKALI CAŁEJ KOLUMNY) ---

# A. Kreatynina (x1000)
if 'Kreatynina' in df.columns:
    df['Kreatynina'] = df['Kreatynina'] * 1000

# B. Czas_Pierwsze_Zaostrzenie (x1440 - minuty)
if 'Czas_Pierwsze_Zaostrzenie' in df.columns:
    df['Czas_Pierwsze_Zaostrzenie'] = df['Czas_Pierwsze_Zaostrzenie'] * 1440

# Zapisanie pliku
nazwa_pliku = 'Data/zapalenia_naczyn_z_problemami.csv'
# Utworzenie katalogu Data jeśli nie istnieje
import os
if not os.path.exists('Data'):
    os.makedirs('Data')
    
df.to_csv(nazwa_pliku, sep='|', index=False)

# Raport
print("-" * 30)
print(f"Gotowe. Zapisano plik: {nazwa_pliku}")
print(f"Liczba wstawionych pustych komórek (NaN): {df.isnull().sum().sum()}")
if 'Wiek' in df.columns:
    print(f"Przykładowe wartości wieku (w tym błędy): {df.loc[idx_wiek, 'Wiek'].values}")
print("-" * 30)