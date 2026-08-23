import os
import numpy as np


def detect_cuda_device():
    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(tree_method="hist", device="cuda", n_estimators=1)
        clf.fit(np.zeros((2, 2)), np.array([0, 1]))
        return "cuda"
    except Exception:
        return "cpu"


CUDA_DEVICE = detect_cuda_device()
USE_CUDA = (CUDA_DEVICE == "cuda")

DATASETS_ALL = [
    {
        "name": "zapalenia",
        "original_file": "Data/zapalenia_naczyn.csv",
        "target_col": "Zgon",
        "separator": "|",
        "target_map": None,
        "drop_columns": ["Kod"],
        "continuous_columns": [
            "Wiek",
            "Wiek_rozpoznania",
            "Opoznienie_Rozpoznia",
            "Paczkolata",
            "Liczba_Zajetych_Narzadow",
            "Liczba_Zaostrzen ",
            "Czas_Pierwsze_Zaostrzenie",
            "Kreatynina",
            "Max_CRP",
            "Sterydy_Dawka_g",
            "Sterydy_Dawka_mg",
            "Czas_Sterydow",
            "Anti-PR3_Wartosc",
            "Anti-MPO_Wartosc",
            "Eozynofilia_Krwi_Obwodowej_Wartosc",
        ],
        "categorical_columns": None,  # dynamicznie wyznaczone jako pozostałe kolumny cech
    },
    {
        "name": "diabetes",
        "original_file": "Data/diabetes.csv",
        "target_col": "decision",
        "separator": ",",
        "target_map": {"tested_negative": 0, "tested_positive": 1},
        "drop_columns": [],
        "continuous_columns": [
            "preg",
            "plas",
            "pres",
            "skin",
            "insu",
            "mass",
            "pedi",
            "age",
        ],
        "categorical_columns": [],
    },
    {
        "name": "serce",
        "original_file": "Data/serce.csv",
        "target_col": "diagnoza",
        "separator": ",",
        "target_map": {1: 0, 2: 1},
        "drop_columns": [],
        "continuous_columns": [
            "wiek",
            "cisnienie_krwi_spoczynek",
            "cholesterol_we_krwi",
            "ilosc_uderzen_serca",
            "max_obnizka_st",
        ],
        "categorical_columns": [
            "plec",
            "typ_bolu_klatka",
            "cukier_we_krwi",
            "wynik_ekg_spoczynek",
            "bol_klatka_wysilek",
            "przebieg_st_szczyt",
            "zwapnienia_miazdzycowe",
            "proba_ta",
        ],
    },
    {
        "name": "rezygnacje",
        "original_file": "Data/rezygnacje.csv",
        "target_col": "REZYGN",
        "separator": ",",
        "target_map": None,
        "drop_columns": ["NR_TEL"],
        "continuous_columns": [
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
        "categorical_columns": [
            "STAN",
            "KOD_OBSZARU",
            "PLAN_MIEDZY",
            "POCZTA_G",
        ],
    },
]

DATASETS_NO_ZAPALENIA = [ds for ds in DATASETS_ALL if ds["name"] != "zapalenia"]

EXPERIMENT_PROFILES = {
    "full": {
        "datasets": DATASETS_ALL,
        "methods": ["raw", "norm", "fill", "remove", "remove_fill", "remove_norm", "fill_norm", "all"],
        "models": ["RF", "NB", "MLP", "XGBoost"],
        "damage_levels": [20, 40, 60],
        "damage_repeats": [1, 2, 3, 4, 5],
        "cv_states": [101, 202, 303, 404, 505],
        "cv_folds": 5,
        "output_suffix": "",
        "dirty_file_pattern": "{name}_prob_{level}_r{repeat}.csv",
    },
    "10_50": {
        "datasets": DATASETS_NO_ZAPALENIA,
        "methods": ["raw", "fill", "remove_fill", "fill_norm", "all", "fill_knn", "all_knn"],
        "models": ["RF", "NB", "MLP", "XGBoost"],
        "damage_levels": [10, 20, 30, 40, 50],
        "damage_repeats": [1, 2, 3],
        "cv_states": [101, 202, 303],
        "cv_folds": 4,
        "output_suffix": "_10_50",
        "dirty_file_pattern": "{name}_10_50_prob_{level}_r{repeat}.csv",
    },
    "fast": {
        "datasets": DATASETS_ALL,
        "methods": ["raw", "fill", "remove_fill", "fill_norm", "all"],
        "models": ["RF", "NB", "XGBoost"],
        "damage_levels": [20, 40, 60],
        "damage_repeats": [1, 2],
        "cv_states": [101, 202],
        "cv_folds": 3,
        "output_suffix": "_fast",
        "dirty_file_pattern": "{name}_prob_{level}_r{repeat}.csv",
    },
}

PARALLEL_JOBS = max(1, min(8, (os.cpu_count() or 4) - 2))
