import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

SCRIPTS = [
    PROJECT_ROOT / "Src" / "stworz_problemy_10_50.py",
    PROJECT_ROOT / "KlasaTestowa" / "TestMyClassifier_10_50.py",
    PROJECT_ROOT / "Src" / "wizualizacja_10_50.py",
]


def run_script(script_path):
    print(f"\n=== Uruchamianie: {script_path.name} ===")
    result = subprocess.run([PYTHON, str(script_path)], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    for script in SCRIPTS:
        run_script(script)

    print("\nPipeline 10_50 zakonczony sukcesem.")


if __name__ == "__main__":
    main()
