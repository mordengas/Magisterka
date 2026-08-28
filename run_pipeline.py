"""Runner pipeline eksperymentalnego.

Użycie:
    python run_pipeline.py --profile full    # pełny eksperyment
    python run_pipeline.py --profile fast    # szybki test
"""
import argparse
import subprocess
import sys
from pathlib import Path

from config import EXPERIMENT_PROFILES

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

# Skrypty uruchamiane w kolejności — każdy otrzyma --profile
PIPELINE_SCRIPTS = [
    PROJECT_ROOT / "Src" / "stworz_problemy.py",
    PROJECT_ROOT / "KlasaTestowa" / "TestMyClassifier.py",
    PROJECT_ROOT / "Src" / "wizualizacja.py",
    PROJECT_ROOT / "Src" / "wizualizacje_zaawansowane.py",
    PROJECT_ROOT / "Src" / "testy_statystyczne.py",
    PROJECT_ROOT / "Src" / "generuj_tabele_latex.py",
]


def run_script(script_path, profile):
    print(f"\n{'='*60}")
    print(f"=== Uruchamianie: {script_path.name} --profile {profile} ===")
    print(f"{'='*60}")
    result = subprocess.run(
        [PYTHON, str(script_path), "--profile", profile],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print(f"\n[BLAD] Skrypt {script_path.name} zakonczyl sie z kodem {result.returncode}")
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Runner pipeline eksperymentalnego")
    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        choices=list(EXPERIMENT_PROFILES.keys()),
        help="Profil eksperymentu (domyslnie: full)",
    )
    args = parser.parse_args()

    print(f"=== PIPELINE EKSPERYMENTALNY [profil: {args.profile}] ===")

    for script in PIPELINE_SCRIPTS:
        run_script(script, args.profile)

    print(f"\n{'='*60}")
    print(f"Pipeline '{args.profile}' zakonczony sukcesem.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
