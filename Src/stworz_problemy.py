import pandas as pd
import numpy as np

# 1. Wczytanie danych
df = pd.read_csv('Data/zapalenia_naczyn.csv', sep='|')
np.random.seed(42) # Dla powtarzalności

# --- PROBLEM 1: WIĘCEJ PUSTYCH MIEJSC (Missing Values) ---
# Rozszerzona lista kolumn, w których pojawią się "dziury"
# Teraz obejmuje też leczenie, objawy i wyniki badań
kolumny_z_brakami = [
    'Wiek', 'Kreatynina', 'Max_CRP', 'Plec', 
    'Liczba_Zajetych_Narzadow', 'Czas_Pierwsze_Zaostrzenie', 
    'Sterydy_Dawka_mg', 'ANCA_Obecne', 'Leczenie_Glikokortkosteroidy_Doustnie'
]

for col in kolumny_z_brakami:
    if col in df.columns:
        # Losujemy różny procent braków dla każdej kolumny (od 5% do 25%)
        # Dzięki temu niektóre kolumny będą "bardziej dziurawe" niż inne
        procent_brakow = np.random.uniform(0.05, 0.25)
        maska = np.random.random(len(df)) < procent_brakow
        df.loc[maska, col] = np.nan

# --- PROBLEM 2: WIĘCEJ OUTLIERÓW I BŁĘDÓW LOGICZNYCH ---
# A. Max_CRP
if 'Max_CRP' in df.columns:
    idx = np.random.choice(df.index, 5, replace=False)
    df.loc[idx, 'Max_CRP'] = df.loc[idx, 'Max_CRP'] * 50

# B. Wiek
# Wstawiamy wartości ujemne lub ekstremalnie wysokie (np. 200 lat), symulując błędy wprowadzania
if 'Wiek' in df.columns:
    idx_wiek = np.random.choice(df.index, 5, replace=False)
    df.loc[idx_wiek, 'Wiek'] = np.random.choice([-10, 150, 200, 0], 5)

# C. Sterydy
# Wstawiamy ogromną liczbę, sugerującą np. pomyłkę (wpisanie gramów zamiast miligramów itp.)
if 'Sterydy_Dawka_mg' in df.columns:
    idx_st = np.random.choice(df.index, 3, replace=False)
    df.loc[idx_st, 'Sterydy_Dawka_mg'] = 50000 

# --- PROBLEM 3: TRUDNIEJSZA NORMALIZACJA ---
# A. Kreatynina
if 'Kreatynina' in df.columns:
    df['Kreatynina'] = df['Kreatynina'] * 1000

# B. Czas_Pierwsze_Zaostrzenie
# Zamieniamy dni na minuty (* 1440). To tworzy zmienną o ogromnej wariancji i wielkich liczbach.
if 'Czas_Pierwsze_Zaostrzenie' in df.columns:
    df['Czas_Pierwsze_Zaostrzenie'] = df['Czas_Pierwsze_Zaostrzenie'] * 1440

# Zapisanie pliku
nazwa_pliku = 'Data/zapalenia_naczyn_z_problemami.csv'
df.to_csv(nazwa_pliku, sep='|', index=False)

# Raport dla użytkownika
print(f"Gotowe. Plik zapisany jako: {nazwa_pliku}")
print("Przykładowe nowe problemy:")
print(f"- Braki w 'Czas_Pierwsze_Zaostrzenie': {df['Czas_Pierwsze_Zaostrzenie'].isnull().sum()}")
print(f"- Minimalny wiek (powinien być dodatni): {df['Wiek'].min()}")