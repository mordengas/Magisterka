import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

SCRIPTS = [
    (PROJECT_ROOT / "Src" / "stworz_problemy_10_50.py", []),
    (PROJECT_ROOT / "KlasaTestowa" / "TestMyClassifier.py", ["--profile", "10_50"]),
    (PROJECT_ROOT / "Src" / "wizualizacja.py", ["--profile", "10_50"]),
    (PROJECT_ROOT / "Src" / "testy_statystyczne.py", ["--profile", "10_50"]),
]

def run_script(script_path, extra_args=None):
    extra_args = extra_args or []
    print(f"\n=== Uruchamianie: {script_path.name} {' '.join(extra_args)} ===")
    result = subprocess.run([PYTHON, str(script_path)] + extra_args, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

def main():
    for script, args in SCRIPTS:
        run_script(script, args)
    print("\nPipeline 10_50 zakonczony sukcesem.")

if __name__ == "__main__":
    main()
