"""Generowanie uszkodzonych zbiorów danych do eksperymentów.

Użycie:
    python Src/stworz_problemy.py --profile full
    python Src/stworz_problemy.py --profile fast
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASETS_ALL, EXPERIMENT_PROFILES


def load_source_dataframe(filename, separator):
    path = os.path.join("Data", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku zrodlowego: {path}")
    return pd.read_csv(path, sep=separator)


def inject_missingness(df_dirty, candidate_columns, damage_level, rng):
    """Wprowadza losowe braki danych (NaN) do wskazanych kolumn."""
    row_count = len(df_dirty)
    for col in candidate_columns:
        missing_count = int(row_count * damage_level)
        if missing_count <= 0:
            continue
        row_indices = rng.choice(row_count, size=missing_count, replace=False)
        df_dirty.loc[row_indices, col] = np.nan


def inject_continuous_outliers(df_dirty, continuous_columns, damage_level, rng,
                                factors, rate_scale, rate_floor):
    """Wprowadza wartości odstające (outliers) do kolumn ciągłych.

    Parametry rate_scale i rate_floor kontrolują intensywność:
        outlier_rate = max(rate_floor, damage_level * rate_scale)
    """
    for col in continuous_columns:
        if col not in df_dirty.columns:
            continue

        valid_indices = df_dirty[df_dirty[col].notna()].index.to_numpy()
        if len(valid_indices) == 0:
            continue

        outlier_rate = max(rate_floor, damage_level * rate_scale)
        outlier_count = max(1, int(len(valid_indices) * outlier_rate))
        selected_indices = rng.choice(valid_indices, size=outlier_count, replace=False)
        chosen_factors = rng.choice(factors, size=outlier_count)
        df_dirty.loc[selected_indices, col] = (
            df_dirty.loc[selected_indices, col].to_numpy() * chosen_factors
        )


def inject_categorical_noise(df_dirty, categorical_columns, damage_level, rng,
                               rate_scale, rate_floor):
    """Wprowadza szum do kolumn kategorycznych (zamiana na wartość '9')."""
    for col in categorical_columns:
        if col not in df_dirty.columns:
            continue

        valid_indices = df_dirty[df_dirty[col].notna()].index.to_numpy()
        if len(valid_indices) == 0:
            continue

        noise_rate = max(rate_floor, damage_level * rate_scale)
        noise_count = max(1, int(len(valid_indices) * noise_rate))
        selected_indices = rng.choice(valid_indices, size=noise_count, replace=False)
        if pd.api.types.is_numeric_dtype(df_dirty[col]):
            df_dirty.loc[selected_indices, col] = 9
        else:
            df_dirty.loc[selected_indices, col] = "9"


def generate_dirty_dataset(ds_config, damage_level_pct, repeat_no, profile):
    """Generuje i zapisuje jeden uszkodzony zbiór danych.

    Args:
        ds_config: słownik z config.DATASETS_ALL
        damage_level_pct: poziom uszkodzeń w procentach (np. 20, 40)
        repeat_no: numer powtórzenia (1, 2, 3, ...)
        profile: słownik profilu z config.EXPERIMENT_PROFILES
    """
    damage_level = damage_level_pct / 100.0
    df = load_source_dataframe(
        os.path.basename(ds_config["original_file"]),
        ds_config["separator"],
    )
    df_dirty = df.copy()

    save_dir = os.path.join("Data", ds_config["name"])
    os.makedirs(save_dir, exist_ok=True)

    # Kolumny chronione = target + kolumny do usunięcia
    protected = [ds_config["target_col"]] + ds_config.get("drop_columns", [])
    candidate_columns = [col for col in df.columns if col not in protected]

    # Kolumny ciągłe z config (lub wszystkie numeryczne)
    if ds_config.get("continuous_columns"):
        continuous_columns = [col for col in ds_config["continuous_columns"] if col in df.columns]
    else:
        continuous_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        continuous_columns = [c for c in continuous_columns if c not in protected]

    categorical_columns = [col for col in candidate_columns if col not in continuous_columns]

    # Deterministyczny seed zależny od profilu, powtórzenia i poziomu
    seed = profile["damage_seed_base"] + repeat_no * 100 + damage_level_pct
    rng = np.random.default_rng(seed)

    # Parametry uszkodzeń z profilu
    factors = profile["outlier_factors"]
    rate_scale = profile["outlier_rate_scale"]
    rate_floor = profile["outlier_rate_floor"]
    noise_scale = profile["noise_rate_scale"]
    noise_floor = profile["noise_rate_floor"]

    inject_missingness(df_dirty, candidate_columns, damage_level, rng)
    inject_continuous_outliers(
        df_dirty, continuous_columns, damage_level, rng,
        factors=factors, rate_scale=rate_scale, rate_floor=rate_floor,
    )
    inject_categorical_noise(
        df_dirty, categorical_columns, damage_level, rng,
        rate_scale=noise_scale, rate_floor=noise_floor,
    )

    # Nazwa pliku wyjściowego z wzorca profilu
    output_name = profile["dirty_file_pattern"].format(
        name=ds_config["name"], level=damage_level_pct, repeat=repeat_no
    )
    output_path = os.path.join(save_dir, output_name)

    df_dirty.to_csv(output_path, sep="|", index=False)
    print(
        f"Zapisano: {output_path} | poziom={damage_level_pct}% | "
        f"powtorzenie={repeat_no} | seed={seed}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generowanie uszkodzonych zbiorów danych"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        choices=list(EXPERIMENT_PROFILES.keys()),
        help="Profil eksperymentu (domyslnie: full)",
    )
    args = parser.parse_args()

    profile = EXPERIMENT_PROFILES[args.profile]
    datasets = profile["datasets"]
    damage_levels = profile["damage_levels"]
    damage_repeats = profile["damage_repeats"]

    print(f"=== Generowanie uszkodzeń [profil: {args.profile}] ===")
    print(f"Poziomy uszkodzeń: {damage_levels}")
    print(f"Powtórzenia: {damage_repeats}")
    print(f"Seed bazowy: {profile['damage_seed_base']}")

    for ds_config in datasets:
        print(f"\n--- Zbiór: {ds_config['name']} ---")
        for level in damage_levels:
            for repeat_no in damage_repeats:
                generate_dirty_dataset(ds_config, level, repeat_no, profile)

    print(f"\nZakończono generowanie uszkodzeń dla profilu '{args.profile}'.")


if __name__ == "__main__":
    main()
