'''import pandas as pd
import numpy as np

# Konfiguracja
input_filename = 'Data/zapalenia_naczyn_z_problemami.csv'

try:
    df_raw = pd.read_csv(input_filename, sep='|')
    print(f"Wczytano plik: {input_filename} (Wierszy: {len(df_raw)})")
except FileNotFoundError:
    print("BŁĄD: Nie znaleziono pliku wejściowego!")
    exit()

# ==========================================
# DEFINICJE FUNKCJI CZYSZCZĄCYCH
# ==========================================

def apply_normalization(df_input):
    """
    Metoda 1: Normalizacja MinMax (0-1).
    Automatycznie wykrywa kolumny numeryczne, które nie są binarne (0/1) i nie są decyzją.
    """
    df = df_input.copy()
    
    # 1. Pobieramy nazwy wszystkich kolumn numerycznych
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 2. Ustalamy nazwę kolumny decyzyjnej (zakładamy, że jest ostatnia), 
    #    aby przypadkiem jej nie znormalizować (choć zazwyczaj jest 0/1)
    decision_col = df.columns[-1]
    
    cols_to_normalize = []
    
    for col in numeric_cols:
        # Pomijamy kolumnę decyzyjną
        if col == decision_col:
            continue
        
        # Sprawdzamy, czy kolumna jest binarna (np. Płeć, Zgon, Objawy 0/1)
        # Ignorujemy NaN przy sprawdzaniu liczby unikalnych wartości
        unique_vals = df[col].dropna().unique()
        
        # Kryterium: Jeżeli kolumna ma więcej niż 2 unikalne wartości, 
        # uznajemy ją za cechę ciągłą/wielowartościową wymagającą normalizacji.
        if len(unique_vals) > 2:
            cols_to_normalize.append(col)
    
    # Opcjonalnie: wypisz co będzie normalizowane, aby mieć kontrolę
    print(f"Kolumny wybrane do normalizacji: {cols_to_normalize}")
    
    # 3. Wykonujemy normalizację MinMax tylko na wybranych kolumnach
    for col in cols_to_normalize:
        min_val = df[col].min()
        max_val = df[col].max()
        
        # Unikamy dzielenia przez zero (zabezpieczenie dla kolumn o stałej wartości)
        if max_val != min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
            
    return df

def apply_imputation(df_input):
    """Metoda 2: Wypełnianie braków (Mean dla liczb, Mode dla reszty)."""
    df = df_input.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    # Numeryczne -> średnia
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    # Tekstowe -> moda
    for col in non_numeric_cols:
        if df[col].isnull().any() and not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    return df

def apply_removal(df_input):
    """Metoda 3: Usuwanie wierszy z błędami logicznymi (Outliery)."""
    df = df_input.copy()
    
    # 1. Wiek (musi być > 0 lub pusty)
    if 'Wiek' in df.columns:
        warunek_sensowny_wiek = (df['Wiek'] > 0)
        warunek_jest_pusty = df['Wiek'].isnull()
        df = df[warunek_sensowny_wiek | warunek_jest_pusty]

    # 2. Max_CRP (usuwamy ekstremalne piki powyżej 98 centyla)
    if 'Max_CRP' in df.columns:
        limit = df['Max_CRP'].quantile(0.98)
        warunek_normalne_crp = df['Max_CRP'] < limit
        warunek_puste_crp = df['Max_CRP'].isnull()
        df = df[warunek_normalne_crp | warunek_puste_crp]

    # 3. Sterydy (usuwamy oczywiste błędy > 10000)
    if 'Sterydy_Dawka_mg' in df.columns:
        warunek_dawka_ok = df['Sterydy_Dawka_mg'] < 10000
        warunek_brak_dawki = df['Sterydy_Dawka_mg'].isnull()
        df = df[warunek_dawka_ok | warunek_brak_dawki]
        
    return df

# ==========================================
# GENEROWANIE 7 PLIKÓW
# ==========================================
datasets = {}

# --- A. POJEDYNCZE METODY ---
# 1. Tylko Normalizacja
datasets['zapalenia_1_norm.csv'] = apply_normalization(df_raw)

# 2. Tylko Wypełnianie
datasets['zapalenia_2_fill.csv'] = apply_imputation(df_raw)

# 3. Tylko Usuwanie
datasets['zapalenia_3_remove.csv'] = apply_removal(df_raw)

# --- B. PARY METOD (Kombinacje) ---
# Ważna kolejność: Najpierw usuwamy śmieci, potem wypełniamy/normalizujemy

# 4. Usuwanie + Wypełnianie (Najpierw wyrzucamy błędy, potem uzupełniamy braki średnią z czystych danych)
df_remove = apply_removal(df_raw)
datasets['zapalenia_4_remove_fill.csv'] = apply_imputation(df_remove)

# 5. Usuwanie + Normalizacja (Najpierw wyrzucamy błędy, żeby nie psuły skali MinMax)
df_remove = apply_removal(df_raw) # Ponownie, dla jasności
datasets['zapalenia_5_remove_norm.csv'] = apply_normalization(df_remove)

# 6. Wypełnianie + Normalizacja (Bez usuwania - wypełniamy "brudną" średnią, potem skalujemy)
df_fill = apply_imputation(df_raw)
datasets['zapalenia_6_fill_norm.csv'] = apply_normalization(df_fill)

# --- C. WSZYSTKIE METODY ---
# 7. Usuwanie -> Wypełnianie -> Normalizacja (Pełny proces)
df_full = apply_removal(df_raw)      # Krok 1: Wyrzuć śmieci
df_full = apply_imputation(df_full)  # Krok 2: Uzupełnij luki
datasets['zapalenia_7_all.csv'] = apply_normalization(df_full) # Krok 3: Znormalizuj

# ==========================================
# ZAPIS I RAPORT
# ==========================================
print("-" * 60)
print(f"{'NAZWA PLIKU':<35} | {'WIERSZY':<8} | {'NAN (Suma)':<10}")
print("-" * 60)

for filename, data in datasets.items():
    full_path = f'Data/{filename}'
    data.to_csv(full_path, sep='|', index=False)
    
    nans = data.isnull().sum().sum()
    rows = len(data)
    print(f"{filename:<35} | {rows:<8} | {nans:<10}")

print("-" * 60)
print("Zakończono generowanie 7 plików.")
'''

import pandas as pd
import numpy as np
import os

from sklearn.impute import KNNImputer

#def apply_normalization(df):
#    df = df.copy()
#    for col in df.select_dtypes(include=[np.number]).columns:
#        if col != df.columns[-1] and df[col].nunique() > 2:
#            val_min, val_max = df[col].min(), df[col].max()
#            if val_max != val_min:
#                df[col] = (df[col] - val_min) / (val_max - val_min)
#    return df

def apply_normalization(df):
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != df.columns[-1] and df[col].nunique() > 2:
            srednia = df[col].mean()
            odchylenie = df[col].std()
            
            # Zapobiegamy dzieleniu przez zero
            if odchylenie != 0 and pd.notna(odchylenie):
                df[col] = (df[col] - srednia) / odchylenie
    return df

#def apply_imputation(df):
#    df = df.copy()
#    for col in df.select_dtypes(include=[np.number]).columns:
#        df[col] = df[col].fillna(df[col].median())
#    return df
def apply_imputation(df):
    df = df.copy()
    
    # Wybieramy tylko kolumny liczbowe (bez kolumny decyzyjnej)
    cols_to_impute = [col for col in df.select_dtypes(include=[np.number]).columns if col != df.columns[-1]]
    
    if len(cols_to_impute) > 0:
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])
        
    # Dla kolumn tekstowych/kategorialnych bez zmian zostawiamy modę (jeśli takie masz)
    for col in df.select_dtypes(exclude=[np.number]).columns:
        if df[col].isnull().any() and not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    return df

def apply_removal(df):
    df = df.copy()
    
    # KROK 1: Usuwanie skrajnych outlierów z kolumn ciągłych (liczbowych)
    for col in df.select_dtypes(include=[np.number]).columns:
        # Pomijamy kolumnę decyzyjną (ostatnią)
        if col == df.columns[-1]: 
            continue
        
        # Jeśli kolumna ma mało unikalnych wartości (np. < 10), traktujemy ją jako kategorialną 
        # i zostawiamy na Krok 2
        if df[col].nunique() < 10:
            continue
        
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        
        # Jeśli IQR wynosi 0 (brak odpowiedniej wariancji), pomijamy
        if iqr == 0: 
            continue

        # ZMIANA: Mnożnik 3.0 pozwala usunąć tylko skrajne, sztuczne błędy (np. x100),
        # chroniąc jednocześnie naturalne wartości odchylone (prawdziwy sygnał medyczny).
        mnoznik = 3.0 
        dolna_granica = q1 - mnoznik * iqr
        gorna_granica = q3 + mnoznik * iqr
        
        # Zamiana gigantycznych outlierów na NaN
        mask = (df[col] < dolna_granica) | (df[col] > gorna_granica)
        df.loc[mask, col] = np.nan
        
    # KROK 2: Usuwanie szumu z kolumn kategorialnych (np. sztucznych wartości "9")
    for col in df.columns:
        if col == df.columns[-1]: 
            continue
        
        # Analizujemy tylko kolumny kategorialne (te z małą liczbą unikalnych wartości)
        if df[col].nunique() < 10:
            # Sposób automatyczny: sprawdzamy częstotliwość występowania każdej wartości.
            # Zostawiamy tylko te kategorie, które stanowią rozsądną część danych (np. > 5%).
            # Pozwoli to wyłapać rzadkie błędy ("9") i zamienić je na NaN.
            czestotliwosc = df[col].value_counts(normalize=True)
            poprawne_kategorie = czestotliwosc[czestotliwosc > 0.05].index
            
            # Zaznaczamy wartości, które NIE należą do poprawnych kategorii (i nie są już NaN)
            mask = ~df[col].isin(poprawne_kategorie) & df[col].notna()
            df.loc[mask, col] = np.nan

    return df

def apply_removal_rows(df):
    df = df.copy()
    rows_to_drop = set() # Zbiór indeksów do usunięcia
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == df.columns[-1]: continue
        
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0: continue

        # Znajdź indeksy, gdzie są outliery
        outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)].index
        rows_to_drop.update(outliers)
    
    # Usuń zebrane indeksy
    df = df.drop(index=list(rows_to_drop))
    return df


# Lista folderów/datasetów
datasets = ['zapalenia', 'diabetes', 'serce']
procenty = [20, 40, 60]

for ds in datasets:
    # Ścieżka do podfolderu np. Data/diabetes
    folder_path = f'Data/{ds}'
    
    if not os.path.exists(folder_path):
        print(f"Pominięto folder (nie istnieje): {folder_path}")
        continue

    for p in procenty:
        # Szukamy pliku np. Data/diabetes/diabetes_prob_15.csv
        filename = f"{ds}_prob_{p}.csv"
        path = f"{folder_path}/{filename}"
        
        if not os.path.exists(path):
            continue
            
        print(f"Naprawianie: {ds} (poziom {p}%) w folderze {folder_path}...")
        
        df_raw = pd.read_csv(path, sep='|')
        # Prefix do zapisu też musi zawierać folder
        # np. Data/diabetes/diabetes_prob_15
        save_prefix = f"{folder_path}/{ds}_prob_{p}"

        # Generowanie wariantów - zapisujemy do tego samego podfolderu
        apply_normalization(df_raw).to_csv(f'{save_prefix}_1_norm.csv', sep='|', index=False)
        apply_imputation(df_raw).to_csv(f'{save_prefix}_2_fill.csv', sep='|', index=False)

        df_rem = apply_removal(df_raw)
        df_rem.to_csv(f'{save_prefix}_3_remove.csv', sep='|', index=False)
        
        apply_imputation(df_rem).to_csv(f'{save_prefix}_4_remove_fill.csv', sep='|', index=False)
        apply_normalization(df_rem).to_csv(f'{save_prefix}_5_remove_norm.csv', sep='|', index=False)
        apply_normalization(apply_imputation(df_raw)).to_csv(f'{save_prefix}_6_fill_norm.csv', sep='|', index=False)
        
        df_final = apply_normalization(apply_imputation(df_rem))
        df_final.to_csv(f'{save_prefix}_7_all.csv', sep='|', index=False)

print("\nZakończono naprawianie wszystkich plików w podfolderach.")