import pandas as pd
import numpy as np

# 1. Wczytanie danych
df = pd.read_csv('Data/zapalenia_naczyn.csv', sep='|')
np.random.seed(42) # Dla powtarzalności

# --- PROBLEM 1: LOSOWE BRAKI DANYCH (Globalnie ~5%, bez kolumny decyzyjnej) ---

procent_brakow = 0.05

# 1. Generujemy losową maskę dla całego DataFrame
maska_losowa = np.random.random(df.shape) < procent_brakow

# 2. KLUCZOWA POPRAWKA: Ustawiamy ostatnią kolumnę maski na False
#    Indeks -1 oznacza ostatnią kolumnę. Dzięki temu nigdy nie wstawimy tam braku.
maska_losowa[:, -1] = False

# 3. Wstawiamy NaN tam, gdzie maska jest True
df[maska_losowa] = np.nan

# --- PROBLEM 2: OUTLIERY I BŁĘDY LOGICZNE ---

# A. Max_CRP (Klasyczny outlier wielkości)
if 'Max_CRP' in df.columns:
    idx = np.random.choice(df[df['Max_CRP'].notna()].index, 5, replace=False)
    df.loc[idx, 'Max_CRP'] = df.loc[idx, 'Max_CRP'] * 50

# B. Wiek (Dostosowane do formatu dni)
if 'Wiek' in df.columns:
    # Wybieramy 5 losowych rekordów
    idx_wiek = np.random.choice(df[df['Wiek'].notna()].index, 5, replace=False)
    
    # Wstawiamy różne rodzaje błędów:
    # -100: Wiek ujemny (błąd logiczny)
    # 100000: Wiek ~273 lata (ekstremalny outlier w górę)
    # 50: Wiek wpisany w latach zamiast w dniach (ukryty błąd jednostki, 50 dni to niemowlę)
    bledne_wartosci = [-100, 100000, 50, 150000, -5000]
    df.loc[idx_wiek, 'Wiek'] = bledne_wartosci

# C. Sterydy (Błąd jednostki/przecinka)
if 'Sterydy_Dawka_mg' in df.columns:
    idx_st = np.random.choice(df[df['Sterydy_Dawka_mg'].notna()].index, 3, replace=False)
    df.loc[idx_st, 'Sterydy_Dawka_mg'] = 50000 

# --- PROBLEM 3: TRUDNIEJSZA NORMALIZACJA ---

# A. Kreatynina (Skalowanie)
if 'Kreatynina' in df.columns:
    df['Kreatynina'] = df['Kreatynina'] * 1000

# B. Czas_Pierwsze_Zaostrzenie (Zmiana jednostki na minuty)
if 'Czas_Pierwsze_Zaostrzenie' in df.columns:
    df['Czas_Pierwsze_Zaostrzenie'] = df['Czas_Pierwsze_Zaostrzenie'] * 1440

# Zapisanie pliku
nazwa_pliku = 'Data/zapalenia_naczyn_z_problemami.csv'
df.to_csv(nazwa_pliku, sep='|', index=False)

# Raport
print(f"Gotowe. Zapisano plik: {nazwa_pliku}")
print(f"Liczba wstawionych pustych komórek: {df.isnull().sum().sum()}")
if 'Wiek' in df.columns:
    print(f"Przykładowe zmodyfikowane wartości wieku: {df.loc[idx_wiek, 'Wiek'].values}")