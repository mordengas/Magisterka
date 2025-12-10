import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- KROK 0: Upewnienie się, że mamy plik z problemami ---
input_filename = 'Data/zapalenia_naczyn_z_problemami.csv'

try:
    df = pd.read_csv(input_filename, sep='|')
except FileNotFoundError:
    print("Nie znaleziono pliku wejściowego. Generuję go na nowo...")
    # (Tu następuje szybka regeneracja danych z poprzedniego kroku, aby kod był samodzielny)
    df = pd.read_csv('Data/zapalenia_naczyn.csv', sep='|')
    np.random.seed(42)
    cols_miss = ['Wiek', 'Kreatynina', 'Max_CRP', 'Plec', 'Liczba_Zajetych_Narzadow', 
                 'Czas_Pierwsze_Zaostrzenie', 'Sterydy_Dawka_mg', 'ANCA_Obecne']
    for col in cols_miss:
        if col in df.columns:
            df.loc[np.random.random(len(df)) < 0.15, col] = np.nan
    
    # Outliery i Skala
    if 'Max_CRP' in df.columns:
        df.loc[np.random.choice(df.index, 5), 'Max_CRP'] *= 50
    if 'Wiek' in df.columns:
        df.loc[np.random.choice(df.index, 5), 'Wiek'] = -10
    if 'Kreatynina' in df.columns:
        df['Kreatynina'] *= 1000
    if 'Czas_Pierwsze_Zaostrzenie' in df.columns:
        df['Czas_Pierwsze_Zaostrzenie'] *= 1440
        
    df.to_csv(input_filename, sep='|', index=False)

print(f"Przetwarzanie pliku: {input_filename}")

# ==========================================
# PLIK 1: TYLKO NORMALIZACJA (Only Normalization)
# ==========================================
# Naprawiamy tylko skalę (Kreatynina, Czas), ale zostawiamy puste miejsca i outliery.
df_norm = df.copy()

cols_to_normalize = ['Kreatynina', 'Czas_Pierwsze_Zaostrzenie', 'Wiek', 'Max_CRP']
scaler = MinMaxScaler()

# Uwaga: Scikit-learn scaler nie zadziała z NaN bez błędów w starszych wersjach, 
# więc używamy normalizacji ręcznej w Pandas, która ignoruje NaN (zostawia je jako NaN).
for col in cols_to_normalize:
    if col in df_norm.columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        # Wzór MinMax: (x - min) / (max - min)
        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)

df_norm.to_csv('Data/zapalenia_tylko_znormalizowane.csv', sep='|', index=False)
print("1. Utworzono 'zapalenia_tylko_znormalizowane.csv' (Skala 0-1, ale braki pozostały).")


# ==========================================
# PLIK 2: TYLKO WYPEŁNIANIE BRAKÓW (Only Imputation)
# ==========================================
# Naprawiamy tylko puste miejsca (NaN), ale zostawiamy złą skalę i outliery.
df_filled = df.copy()

numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
non_numeric_cols = df_filled.select_dtypes(exclude=[np.number]).columns

# Numeryczne -> średnia
for col in numeric_cols:
    df_filled[col] = df_filled[col].fillna(df_filled[col].mean())

# Tekstowe/Kategoryczne -> moda (najczęstsza wartość)
for col in non_numeric_cols:
    if not df_filled[col].mode().empty:
        df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])

df_filled.to_csv('Data/zapalenia_tylko_wypelnione.csv', sep='|', index=False)
print("2. Utworzono 'zapalenia_tylko_wypelnione.csv' (Brak pustych miejsc, ale skala i outliery zostały).")


# ==========================================
# PLIK 3: TYLKO USUWANIE WIERSZY (Only Row Removal)
# ==========================================
# Usuwamy wiersze, które zawierają zidentyfikowane problemy (Outliery logiczne).
# Interpretuję to jako usunięcie błędnych rekordów (Outlierów), a nie wszystkich z NaN 
# (bo usunęlibyśmy prawie wszystko).
df_rows = df.copy()
initial_len = len(df_rows)

# Logika: (Wiek jest OK) LUB (Wiek jest Pusty)
if 'Wiek' in df_rows.columns:
    warunek_sensowny_wiek = (df_rows['Wiek'] > 0)
    warunek_jest_pusty = df_rows['Wiek'].isnull()
    
    # Zostawiamy wiersze spełniające jeden z tych warunków
    df_rows = df_rows[warunek_sensowny_wiek | warunek_jest_pusty]

# 2. Naprawa CRP: Usuwamy tylko ekstremalne piki
if 'Max_CRP' in df_rows.columns:
    # Obliczamy próg odcięcia (np. 98 centyl), ignorując NaN w obliczeniach
    limit = df_rows['Max_CRP'].quantile(0.98)
    
    warunek_normalne_crp = df_rows['Max_CRP'] < limit
    warunek_puste_crp = df_rows['Max_CRP'].isnull()
    
    df_rows = df_rows[warunek_normalne_crp | warunek_puste_crp]

# 3. Naprawa Sterydów: Usuwamy tylko te błędne 50000
if 'Sterydy_Dawka_mg' in df_rows.columns:
    warunek_dawka_ok = df_rows['Sterydy_Dawka_mg'] < 10000
    warunek_brak_dawki = df_rows['Sterydy_Dawka_mg'].isnull()
    
    df_rows = df_rows[warunek_dawka_ok | warunek_brak_dawki]

# Zapis i raport
df_rows.to_csv('Data/zapalenia_tylko_usuniete_wiersze.csv', sep='|', index=False)

print(f"Wierszy po czyszczeniu: {len(df_rows)}")
print(f"Usunięto: {len(df) - len(df_rows)} wierszy (tylko te z błędnymi danymi).")
print("Plik zapisany jako: zapalenia_tylko_usuniete_wiersze.csv")