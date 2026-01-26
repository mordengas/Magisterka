'''
import pandas as pd
import numpy as np
import os

# Tworzymy folder na dane jeśli nie istnieje
if not os.path.exists('Data'):
    os.makedirs('Data')

def generuj_problemy(sciezka_in, nazwa_out, proc_brakow):
    try:
        # 1. Wczytanie danych
        if os.path.exists(sciezka_in):
            df = pd.read_csv(sciezka_in, sep='|')
        else:
            df = pd.read_csv('zapalenia_naczyn.csv', sep='|')
            
        np.random.seed(42)
        
        # --- PROBLEM 1: BRAKI DANYCH (NaN) ---
        # Procent przekazany w parametrze (15%, 25%, 50%)
        mask = np.random.random(df.shape) < proc_brakow
        mask[:, -1] = False # Zabezpieczamy kolumnę decyzyjną
        mask[:, 0] = False  # Zabezpieczamy Kod
        df[mask] = np.nan
        
        # --- PROBLEM 2: OUTLIERY (5% błędnych rekordów) ---
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == df.columns[-1] or col == 'Kod' or col == 'Wiek': continue
            # Wybieramy 5% niepustych komórek i robimy z nich ogromne wartości
            non_nan_idx = df[df[col].notna()].index
            if len(non_nan_idx) > 0:
                idx_out = np.random.choice(non_nan_idx, max(1, int(len(non_nan_idx) * 0.05)), replace=False)
                df.loc[idx_out, col] *= 50
        
        # --- NOWOŚĆ: PROBLEM 3: CAŁKOWITA ZMIANA SKALI KOLUMN (Problemy normalizacji) ---
        # Wybieramy konkretne kolumny i drastycznie zmieniamy ich skalę (symulacja różnych jednostek)
        if 'Kreatynina' in df.columns:
            df['Kreatynina'] = df['Kreatynina'] * 1000 # Skala x1000
            
        if 'Max_CRP' in df.columns:
            df['Max_CRP'] = df['Max_CRP'] / 100 # Skala /100
            
        if 'Wiek' in df.columns:
            # Tutaj wprowadzamy błędy typu "data urodzenia zamiast wieku" lub inne skrajności
            idx_logic = df[df['Wiek'].notna()].sample(frac=0.05).index
            df.loc[idx_logic, 'Wiek'] = 150000 # Błąd typu "wiek w dniach"
            
        # 4. Zapisanie pliku
        df.to_csv(f'Data/{nazwa_out}', sep='|', index=False)
        print(f"Sukces: Utworzono Data/{nazwa_out} (Braki: {int(proc_brakow*100)}%, Wprowadzono błędy skali)")

    except Exception as e:
        print(f"Błąd przy tworzeniu {nazwa_out}: {e}")

# Generujemy 3 bazy bazowe z różnym natężeniem braków
procenty = [0.15, 0.25, 0.50]
for p in procenty:
    generuj_problemy('Data/zapalenia_naczyn.csv', f'zapalenia_prob_{int(p*100)}.csv', p)
'''

import pandas as pd
import numpy as np
import os

# Konfiguracja (bez zmian w logice kolumn)
DATASETS_CONFIG = {
    'zapalenia_naczyn.csv': {
        'folder': 'zapalenia',
        'target': 'Zgon',
        'protected': ['Kod', 'Zgon'],
        'continuous': ['Wiek', 'Wiek_rozpoznania', 'Kreatynina', 'Max_CRP', 'Sterydy_Dawka_g', 'Anti-PR3_Wartosc']
    },
    'diabetes.csv': {
        'folder': 'diabetes',
        'target': 'decision',
        'protected': ['decision'],
        'continuous': ['plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']
    },
    'serce.csv': {
        'folder': 'serce',
        'target': 'diagnoza',
        'protected': ['diagnoza'],
        'continuous': ['wiek', 'cisnienie_krwi_spoczynek', 'cholesterol_we_krwi', 'ilosc_uderzen_serca', 'max_obnizka_st']
    }
}

OUTPUT_DIR = 'Data'

def generuj_zepsute_dane(filename, config, procent_uszkodzen):
    folder_name = config['folder']
    print(f"\n--- Przetwarzanie: {filename} -> Folder: {folder_name} (Uszkodzenia: {int(procent_uszkodzen*100)}%) ---")
    
    # 1. Tworzenie podfolderu
    save_dir = os.path.join(OUTPUT_DIR, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. Wczytanie
    path = f"Data/{filename}" if os.path.exists(f"Data/{filename}") else filename
    if not os.path.exists(path):
        print(f"POMINIĘTO: Nie znaleziono pliku źródłowego {filename}")
        return

    sep = '|' if 'zapalenia' in filename else ','
    try:
        df = pd.read_csv(path, sep=sep)
    except:
        sep = ',' if sep == '|' else '|'
        df = pd.read_csv(path, sep=sep)

    df_dirty = df.copy()
    rows = len(df)
    
    target_cols = [c for c in df.columns if c not in config['protected']]
    np.random.seed(42 + int(procent_uszkodzen*100))

    # 1. Braki danych (NaN) - zależne od procent_uszkodzen
    for col in target_cols:
        n_missing = int(rows * procent_uszkodzen)
        if n_missing > 0:
            missing_indices = np.random.choice(rows, n_missing, replace=False)
            df_dirty.loc[missing_indices, col] = np.nan

    # 2. Outliery - TERAZ ZALEŻNE OD procent_uszkodzen (zamiast sztywnego 5%)
    # Uwaga: Aplikujemy to do pozostałych (nie-NaN) wartości w kolumnach ciągłych
    existing_cont = [c for c in config['continuous'] if c in df.columns]
    for col in existing_cont:
        valid_idx = df_dirty[df_dirty[col].notna()].index
        if len(valid_idx) > 0:
            # Zmiana: używamy procent_uszkodzen zamiast 0.05
            n_outliers = max(1, int(len(valid_idx) * procent_uszkodzen))
            outlier_idx = np.random.choice(valid_idx, n_outliers, replace=False)
            
            factor = np.random.choice([100, 1000, -100])
            df_dirty.loc[outlier_idx, col] = df_dirty.loc[outlier_idx, col] * factor

    # 3. Szum kategorialny - TERAZ ZALEŻNE OD procent_uszkodzen (zamiast sztywnego 5%)
    # Aplikujemy do kolumn, które nie są ciągłe (binarne/kategorialne)
    categorical_cols = [c for c in target_cols if c not in existing_cont]
    for col in categorical_cols:
        valid_idx = df_dirty[df_dirty[col].notna()].index
        if len(valid_idx) > 0:
            # Zmiana: używamy procent_uszkodzen zamiast 0.05
            n_noise = max(1, int(len(valid_idx) * procent_uszkodzen))
            noise_idx = np.random.choice(valid_idx, n_noise, replace=False)
            df_dirty.loc[noise_idx, col] = 9

    # Zapis
    if 'zapalenia' in filename:
        base_name = 'zapalenia'
    else:
        base_name = filename.replace('.csv', '')
        
    out_name = f"{base_name}_prob_{int(procent_uszkodzen*100)}.csv"
    save_path = os.path.join(save_dir, out_name)
    
    df_dirty.to_csv(save_path, sep='|', index=False)
    print(f"Zapisano w: {save_path}")

# Uruchomienie
procenty = [0.15, 0.25, 0.50]
for filename, config in DATASETS_CONFIG.items():
    for p in procenty:
        generuj_zepsute_dane(filename, config, p)