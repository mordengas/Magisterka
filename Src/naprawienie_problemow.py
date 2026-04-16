import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Src.cleaning_methods import apply_strategy_globally


DATASETS = [
    {"name": "zapalenia", "target": "Zgon", "drop_columns": ["Kod"]},
    {"name": "diabetes", "target": "decision", "drop_columns": []},
    {"name": "serce", "target": "diagnoza", "drop_columns": []},
    {"name": "rezygnacje", "target": "REZYGN", "drop_columns": ["NR_TEL"]},
]

METHODS = [
    ("norm", "1_norm"),
    ("fill", "2_fill"),
    ("remove", "3_remove"),
    ("remove_fill", "4_remove_fill"),
    ("remove_norm", "5_remove_norm"),
    ("fill_norm", "6_fill_norm"),
    ("all", "7_all"),
]

DAMAGE_LEVELS = [20, 40, 60]
DAMAGE_REPEATS = [1, 2, 3, 4, 5]


def iter_dirty_files(dataset_name, damage_level, damage_repeat):
    repeated_path = f"Data/{dataset_name}/{dataset_name}_prob_{damage_level}_r{damage_repeat}.csv"
    legacy_path = f"Data/{dataset_name}/{dataset_name}_prob_{damage_level}.csv"

    if os.path.exists(repeated_path):
        return repeated_path
    if os.path.exists(legacy_path):
        return legacy_path
    return None


def main():
    print("Tworzenie pomocniczych plikow po czyszczeniu.")
    print("Uwaga: te pliki sa tylko do inspekcji i wizualnej kontroli.")
    print("Wlasciwa ewaluacja nadal powinna byc wykonywana wewnatrz CV.")

    for dataset in DATASETS:
        for damage_level in DAMAGE_LEVELS:
            for damage_repeat in DAMAGE_REPEATS:
                source_path = iter_dirty_files(dataset["name"], damage_level, damage_repeat)
                if source_path is None:
                    continue

                df = pd.read_csv(source_path, sep="|")
                prefix = source_path.replace(".csv", "")

                print(f"\nPrzetwarzanie: {source_path}")
                for method_name, suffix in METHODS:
                    cleaned_df = apply_strategy_globally(
                        df=df,
                        target_col=dataset["target"],
                        method_name=method_name,
                        drop_columns=dataset["drop_columns"],
                    )
                    output_path = f"{prefix}_{suffix}.csv"
                    cleaned_df.to_csv(output_path, sep="|", index=False)
                    print(f"  zapisano -> {output_path}")

                if "_r" not in source_path:
                    break

    print("\nZakonczono tworzenie pomocniczych plikow.")


if __name__ == "__main__":
    main()
