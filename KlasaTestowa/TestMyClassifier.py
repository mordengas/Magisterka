import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EXPERIMENT_PROFILES, PARALLEL_JOBS, CUDA_DEVICE, USE_CUDA
from Src.cleaning_methods import build_cleaning_pipeline


# ---------------------------------------------------------------------------
# Konfiguracja profilu – ustawiana w main() na podstawie --profile
# ---------------------------------------------------------------------------
PROFILE = {}


def get_estimator(model_name, random_state):
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=220,
            random_state=random_state,
            n_jobs=1,
        )

    if model_name == "NB":
        return GaussianNB()

    if model_name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=1600,
            random_state=random_state,
        )

    if model_name == "XGBoost":
        xgb_kwargs = {
            "n_estimators": 180,
            "max_depth": 4,
            "learning_rate": 0.07,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "logloss",
            "random_state": random_state,
            "n_jobs": 1,
        }
        if USE_CUDA:
            xgb_kwargs["tree_method"] = "hist"
            xgb_kwargs["device"] = CUDA_DEVICE

        return xgb.XGBClassifier(**xgb_kwargs)

    raise ValueError(f"Nieznany model: {model_name}")


def load_dataset(path, separator, target_col, target_map):
    df = pd.read_csv(path, sep=separator)
    if target_map is not None:
        df[target_col] = df[target_col].map(target_map)

    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(int)
    return df


def build_model_pipeline(df, target_col, method_name, estimator, drop_columns, continuous_columns=None, categorical_columns=None):
    X = df.drop(columns=[target_col, *drop_columns], errors="ignore")
    cleaning_pipeline = build_cleaning_pipeline(
        X,
        method_name,
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
    )

    return Pipeline(
        [
            ("cleaning", cleaning_pipeline),
            ("compatibility_imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", estimator),
        ]
    )


def evaluate_dataframe(df, target_col, method_name, model_name, drop_columns, continuous_columns=None, categorical_columns=None):
    X = df.drop(columns=[target_col, *drop_columns], errors="ignore")
    y = df[target_col]
    rows = []

    cv_states = PROFILE["cv_states"]
    cv_folds = PROFILE["cv_folds"]

    for cv_repeat, cv_seed in enumerate(cv_states, start=1):
        estimator = get_estimator(model_name, cv_seed)
        pipeline = build_model_pipeline(
            df=df,
            target_col=target_col,
            method_name=method_name,
            estimator=clone(estimator),
            drop_columns=drop_columns,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
        )

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cv_seed)
        fold_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
        )

        rows.append(
            {
                "CVRepeat": cv_repeat,
                "CVSeed": cv_seed,
                "AUC": round(float(fold_scores.mean()), 4),
            }
        )

    return rows


def resolve_dirty_path(dataset_name, damage_level, damage_repeat):
    pattern = PROFILE["dirty_file_pattern"]
    filename = pattern.format(name=dataset_name, level=damage_level, repeat=damage_repeat)
    dirty_path = f"Data/{dataset_name}/{filename}"
    if os.path.exists(dirty_path):
        return dirty_path, damage_repeat
    return None, None


def summarize_results(df_details):
    summary = (
        df_details.groupby(["Dataset", "PoziomUszkodzen", "Metoda", "Model"], dropna=False)["AUC"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "AUC_srednia",
                "std": "AUC_std",
                "count": "Liczba_powtorzen",
            }
        )
    )

    summary["AUC_srednia"] = summary["AUC_srednia"].round(4)
    summary["AUC_std"] = summary["AUC_std"].fillna(0.0).round(4)
    return summary


def build_tasks():
    tasks = []
    datasets = PROFILE["datasets"]
    methods = PROFILE["methods"]
    models = PROFILE["models"]
    damage_levels = PROFILE["damage_levels"]
    damage_repeats = PROFILE["damage_repeats"]

    for ds in datasets:
        original_df = load_dataset(
            path=ds["original_file"],
            separator=ds["separator"],
            target_col=ds["target_col"],
            target_map=ds["target_map"],
        )

        for model_name in models:
            tasks.append(
                {
                    "Dataset": ds["name"],
                    "PoziomUszkodzen": "ORYGINALNY",
                    "Metoda": "raw",
                    "Model": model_name,
                    "DamageRepeat": 0,
                    "df": original_df,
                    "target_col": ds["target_col"],
                    "drop_columns": ds["drop_columns"],
                    "continuous_columns": ds.get("continuous_columns"),
                    "categorical_columns": ds.get("categorical_columns"),
                }
            )

        for damage_level in damage_levels:
            for damage_repeat in damage_repeats:
                dirty_path, resolved_repeat = resolve_dirty_path(
                    dataset_name=ds["name"],
                    damage_level=damage_level,
                    damage_repeat=damage_repeat,
                )

                if dirty_path is None:
                    continue

                dirty_df = load_dataset(
                    path=dirty_path,
                    separator="|",
                    target_col=ds["target_col"],
                    target_map=ds["target_map"],
                )

                for method_name in methods:
                    for model_name in models:
                        tasks.append(
                            {
                                "Dataset": ds["name"],
                                "PoziomUszkodzen": f"{damage_level}%",
                                "Metoda": method_name,
                                "Model": model_name,
                                "DamageRepeat": resolved_repeat,
                                "df": dirty_df,
                                "target_col": ds["target_col"],
                                "drop_columns": ds["drop_columns"],
                                "continuous_columns": ds.get("continuous_columns"),
                                "categorical_columns": ds.get("categorical_columns"),
                            }
                        )

    return tasks


def run_task(task):
    print(
        f"[START] {task['Dataset']} | {task['PoziomUszkodzen']} | "
        f"{task['Metoda']} | {task['Model']} | rep={task['DamageRepeat']}"
    )

    eval_rows = evaluate_dataframe(
        df=task["df"],
        target_col=task["target_col"],
        method_name=task["Metoda"],
        model_name=task["Model"],
        drop_columns=task["drop_columns"],
        continuous_columns=task.get("continuous_columns"),
        categorical_columns=task.get("categorical_columns"),
    )

    result_rows = []
    for row in eval_rows:
        result_rows.append(
            {
                "Dataset": task["Dataset"],
                "PoziomUszkodzen": task["PoziomUszkodzen"],
                "Metoda": task["Metoda"],
                "Model": task["Model"],
                "DamageRepeat": task["DamageRepeat"],
                **row,
            }
        )

    print(
        f"[DONE ] {task['Dataset']} | {task['PoziomUszkodzen']} | "
        f"{task['Metoda']} | {task['Model']} | rep={task['DamageRepeat']}"
    )
    return result_rows


def main():
    global PROFILE

    parser = argparse.ArgumentParser(description="Walidacja klasyfikatorow")
    parser.add_argument(
        "--profile",
        type=str,
        default="10_50",
        choices=list(EXPERIMENT_PROFILES.keys()),
        help="Profil eksperymentu (domyslnie: 10_50)",
    )
    args = parser.parse_args()

    PROFILE = EXPERIMENT_PROFILES[args.profile]
    suffix = PROFILE["output_suffix"]

    results_dir = Path("Results")
    results_dir.mkdir(exist_ok=True)

    datasets = PROFILE["datasets"]
    methods = PROFILE["methods"]
    models = PROFILE["models"]
    damage_levels = PROFILE["damage_levels"]
    damage_repeats = PROFILE["damage_repeats"]
    cv_states = PROFILE["cv_states"]
    cv_folds = PROFILE["cv_folds"]

    print(f"=== WALIDACJA BEZ DATA LEAKAGE  [profil: {args.profile}] ===")
    print(f"Datasety: {[ds['name'] for ds in datasets]}")
    print(f"Poziomy uszkodzen: {damage_levels}")
    print(f"Metody: {methods}")
    print(f"Modele: {models}")
    print(f"Powtorzenia uszkodzen: {damage_repeats}")
    print(f"Powtorzenia CV: {len(cv_states)}")
    print(f"Foldy CV: {cv_folds}")
    print(f"Rownolegle zadania: {PARALLEL_JOBS}")

    tasks = build_tasks()
    print(f"Liczba zadan: {len(tasks)}")

    parallel_results = Parallel(n_jobs=PARALLEL_JOBS, backend="loky", verbose=10)(
        delayed(run_task)(task) for task in tasks
    )

    details = [row for task_rows in parallel_results for row in task_rows]
    details_df = pd.DataFrame(details)

    details_path = results_dir / f"wyniki_szczegolowe{suffix}.csv"
    details_df.to_csv(details_path, index=False)

    summary_df = summarize_results(details_df)
    summary_path = results_dir / f"wyniki_koncowe{suffix}.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nZapisano:")
    print(f"- {details_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
