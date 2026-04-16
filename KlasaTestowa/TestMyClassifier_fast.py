import os
import sys
from pathlib import Path

import pandas as pd

from joblib import Parallel, delayed
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

from Src.cleaning_methods import build_cleaning_pipeline


DATASETS_INFO = [
    {
        "name": "zapalenia",
        "original_file": "Data/zapalenia_naczyn.csv",
        "target_col": "Zgon",
        "separator": "|",
        "target_map": None,
        "drop_columns": ["Kod"],
    },
    {
        "name": "diabetes",
        "original_file": "Data/diabetes.csv",
        "target_col": "decision",
        "separator": ",",
        "target_map": {"tested_negative": 0, "tested_positive": 1},
        "drop_columns": [],
    },
    {
        "name": "serce",
        "original_file": "Data/serce.csv",
        "target_col": "diagnoza",
        "separator": ",",
        "target_map": {1: 0, 2: 1},
        "drop_columns": [],
    },
    {
        "name": "rezygnacje",
        "original_file": "Data/rezygnacje.csv",
        "target_col": "REZYGN",
        "separator": ",",
        "target_map": None,
        "drop_columns": ["NR_TEL"],
    },
]

# Szybsza konfiguracja do roboczej analizy.
METHODS = ["raw", "fill", "remove_fill", "fill_norm", "all"]
MODELS = ["RF", "NB", "XGBoost"]
DAMAGE_LEVELS = [20, 40, 60]
DAMAGE_REPEATS = [1, 2]
CV_RANDOM_STATES = [101, 202]
CV_FOLDS = 3
PARALLEL_JOBS = max(1, min(8, (os.cpu_count() or 4) - 2))


def get_estimator(model_name, random_state):
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=180,
            random_state=random_state,
            n_jobs=1,
        )

    if model_name == "NB":
        return GaussianNB()

    if model_name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(80,),
            max_iter=1200,
            random_state=random_state,
        )

    if model_name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=140,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        )

    raise ValueError(f"Nieznany model: {model_name}")


def load_dataset(path, separator, target_col, target_map):
    df = pd.read_csv(path, sep=separator)
    if target_map is not None:
        df[target_col] = df[target_col].map(target_map)

    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(int)
    return df


def build_model_pipeline(df, target_col, method_name, estimator, drop_columns):
    X = df.drop(columns=[target_col, *drop_columns], errors="ignore")
    cleaning_pipeline = build_cleaning_pipeline(X, method_name)

    return Pipeline(
        [
            ("cleaning", cleaning_pipeline),
            ("compatibility_imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", estimator),
        ]
    )


def evaluate_dataframe(df, target_col, method_name, model_name, drop_columns):
    X = df.drop(columns=[target_col, *drop_columns], errors="ignore")
    y = df[target_col]
    rows = []

    for cv_repeat, cv_seed in enumerate(CV_RANDOM_STATES, start=1):
        estimator = get_estimator(model_name, cv_seed)
        pipeline = build_model_pipeline(
            df=df,
            target_col=target_col,
            method_name=method_name,
            estimator=estimator,
            drop_columns=drop_columns,
        )

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=cv_seed)
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
    repeated_name = f"Data/{dataset_name}/{dataset_name}_prob_{damage_level}_r{damage_repeat}.csv"
    legacy_name = f"Data/{dataset_name}/{dataset_name}_prob_{damage_level}.csv"

    if os.path.exists(repeated_name):
        return repeated_name, damage_repeat
    if os.path.exists(legacy_name):
        return legacy_name, 1
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

    for ds in DATASETS_INFO:
        original_df = load_dataset(
            path=ds["original_file"],
            separator=ds["separator"],
            target_col=ds["target_col"],
            target_map=ds["target_map"],
        )

        for model_name in MODELS:
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
                }
            )

        for damage_level in DAMAGE_LEVELS:
            for damage_repeat in DAMAGE_REPEATS:
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

                for method_name in METHODS:
                    for model_name in MODELS:
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
                            }
                        )

                if not os.path.exists(
                    f"Data/{ds['name']}/{ds['name']}_prob_{damage_level}_r{damage_repeat}.csv"
                ):
                    break

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
    print("=== SZYBKA WALIDACJA BEZ DATA LEAKAGE ===")
    print(f"Modele: {MODELS}")
    print(f"Metody: {METHODS}")
    print(f"Powtorzenia uszkodzen: {DAMAGE_REPEATS}")
    print(f"Powtorzenia CV: {len(CV_RANDOM_STATES)}")
    print(f"Foldy CV: {CV_FOLDS}")
    print(f"Rownolegle zadania: {PARALLEL_JOBS}")

    tasks = build_tasks()
    print(f"Liczba zadan: {len(tasks)}")

    parallel_results = Parallel(n_jobs=PARALLEL_JOBS, backend="loky", verbose=10)(
        delayed(run_task)(task) for task in tasks
    )

    details = [row for task_rows in parallel_results for row in task_rows]
    details_df = pd.DataFrame(details)
    details_df.to_csv("wyniki_szczegolowe_fast.csv", index=False)

    summary_df = summarize_results(details_df)
    summary_df.to_csv("wyniki_koncowe_fast.csv", index=False)

    print("\nZapisano:")
    print("- wyniki_szczegolowe_fast.csv")
    print("- wyniki_koncowe_fast.csv")


if __name__ == "__main__":
    main()
