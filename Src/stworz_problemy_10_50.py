import os

import numpy as np
import pandas as pd


DATASETS_CONFIG = {
    "diabetes.csv": {
        "folder": "diabetes",
        "target": "decision",
        "protected": ["decision"],
        "continuous": ["plas", "pres", "skin", "insu", "mass", "pedi", "age"],
        "separator": ",",
    },
    "serce.csv": {
        "folder": "serce",
        "target": "diagnoza",
        "protected": ["diagnoza"],
        "continuous": [
            "wiek",
            "cisnienie_krwi_spoczynek",
            "cholesterol_we_krwi",
            "ilosc_uderzen_serca",
            "max_obnizka_st",
        ],
        "separator": ",",
    },
    "rezygnacje.csv": {
        "folder": "rezygnacje",
        "target": "REZYGN",
        "protected": ["NR_TEL", "REZYGN"],
        "continuous": [
            "CZAS_POSIADANIA",
            "L_WIAD_POCZTA_G",
            "DZIEN_MIN",
            "DZIEN_L_POL",
            "DZIEN_OPLATA",
            "WIECZOR_MIN",
            "WIECZ_L_POL",
            "WIECZ_OPLATA",
            "NOC_MIN",
            "NOC_L_POL",
            "NOC_OPLATA",
            "MIEDZY_MIN",
            "MIEDZY_L_POL",
            "MIEDZY_OPLATA",
            "L_POL_BIURO",
        ],
        "separator": ",",
    },
}

DAMAGE_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]
DAMAGE_REPEATS = 3


def load_source_dataframe(filename, separator):
    path = os.path.join("Data", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku zrodlowego: {path}")
    return pd.read_csv(path, sep=separator)


def inject_missingness(df_dirty, candidate_columns, damage_level, rng):
    row_count = len(df_dirty)
    for col in candidate_columns:
        missing_count = int(row_count * damage_level)
        if missing_count <= 0:
            continue
        row_indices = rng.choice(row_count, size=missing_count, replace=False)
        df_dirty.loc[row_indices, col] = np.nan


def inject_continuous_outliers(df_dirty, continuous_columns, damage_level, rng):
    for col in continuous_columns:
        if col not in df_dirty.columns:
            continue

        valid_indices = df_dirty[df_dirty[col].notna()].index.to_numpy()
        if len(valid_indices) == 0:
            continue

        outlier_count = max(1, int(len(valid_indices) * max(0.05, damage_level * 0.8)))
        selected_indices = rng.choice(valid_indices, size=outlier_count, replace=False)
        factors = rng.choice([20, 40, 80, -20], size=outlier_count)
        df_dirty.loc[selected_indices, col] = df_dirty.loc[selected_indices, col].to_numpy() * factors


def inject_categorical_noise(df_dirty, categorical_columns, damage_level, rng):
    for col in categorical_columns:
        if col not in df_dirty.columns:
            continue

        valid_indices = df_dirty[df_dirty[col].notna()].index.to_numpy()
        if len(valid_indices) == 0:
            continue

        noise_count = max(1, int(len(valid_indices) * max(0.05, damage_level * 0.8)))
        selected_indices = rng.choice(valid_indices, size=noise_count, replace=False)
        df_dirty.loc[selected_indices, col] = 9


def generate_dirty_dataset(filename, config, damage_level, repeat_no):
    df = load_source_dataframe(filename, config["separator"])
    df_dirty = df.copy()

    save_dir = os.path.join("Data", config["folder"])
    os.makedirs(save_dir, exist_ok=True)

    candidate_columns = [col for col in df.columns if col not in config["protected"]]
    continuous_columns = [col for col in config["continuous"] if col in df.columns]
    categorical_columns = [col for col in candidate_columns if col not in continuous_columns]

    seed = 5000 + repeat_no * 100 + int(damage_level * 100)
    rng = np.random.default_rng(seed)

    inject_missingness(df_dirty, candidate_columns, damage_level, rng)
    inject_continuous_outliers(df_dirty, continuous_columns, damage_level, rng)
    inject_categorical_noise(df_dirty, categorical_columns, damage_level, rng)

    base_name = filename.replace(".csv", "")
    output_name = f"{base_name}_10_50_prob_{int(damage_level * 100)}_r{repeat_no}.csv"
    output_path = os.path.join(save_dir, output_name)

    df_dirty.to_csv(output_path, sep="|", index=False)
    print(
        f"Zapisano: {output_path} | poziom={int(damage_level * 100)}% | "
        f"powtorzenie={repeat_no} | seed={seed}"
    )


def main():
    for filename, config in DATASETS_CONFIG.items():
        print(f"\n=== Generowanie uszkodzen 10-50 dla: {filename} ===")
        for damage_level in DAMAGE_LEVELS:
            for repeat_no in range(1, DAMAGE_REPEATS + 1):
                generate_dirty_dataset(filename, config, damage_level, repeat_no)


if __name__ == "__main__":
    main()
