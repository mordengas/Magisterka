import os
import numpy as np


# ---------------------------------------------------------------------------
# CUDA / GPU — lazy evaluation (wywoływane dopiero przy pierwszym użyciu)
# ---------------------------------------------------------------------------
_cuda_device = None


def detect_cuda_device():
    global _cuda_device
    if _cuda_device is not None:
        return _cuda_device
    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(tree_method="hist", device="cuda", n_estimators=1)
        clf.fit(np.zeros((2, 2)), np.array([0, 1]))
        _cuda_device = "cuda"
    except Exception:
        _cuda_device = "cpu"
    return _cuda_device


def get_cuda_device():
    return detect_cuda_device()


def use_cuda():
    return detect_cuda_device() == "cuda"


# ---------------------------------------------------------------------------
# Definicje zbiorów danych
# ---------------------------------------------------------------------------
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
            "Liczba_Zaostrzen",
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
    {
        "name": "kredyty",
        "original_file": "Data/kredyty.tab",
        "target_col": "Kredyt",
        "separator": r"\s+",
        "target_map": {"good": 0, "bad": 1},
        "drop_columns": [],
        "continuous_columns": [
            "Czas_trwania_konta",
            "Kwota_kredytu",
            "Wiek",
        ],
        "categorical_columns": [
            "Cel_kredytu",
            "Czas_zatrudnienia",
            "Plec_i_stan_cywilny",
            "Czas_od_zamieszkania",
            "Liczba_kredytow_w_banku",
            "Praca",
            "Liczb_osob_na_utrzymaniu",
        ],
    },
]

DATASETS_NO_ZAPALENIA = [ds for ds in DATASETS_ALL if ds["name"] != "zapalenia"]


# ---------------------------------------------------------------------------
# Hiperparametry modeli klasyfikacyjnych (centralna definicja)
# ---------------------------------------------------------------------------
MODEL_PARAMS = {
    "RF": {
        "n_estimators": 220,
        "n_jobs": 1,
    },
    "NB": {},
    "MLP": {
        "hidden_layer_sizes": (100,),
        "max_iter": 500,
        "early_stopping": True,
        "n_iter_no_change": 15,
    },
    "XGBoost": {
        "n_estimators": 180,
        "max_depth": 4,
        "learning_rate": 0.07,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "logloss",
        "n_jobs": 1,
    },
}


# ---------------------------------------------------------------------------
# Profile eksperymentów
# ---------------------------------------------------------------------------
EXPERIMENT_PROFILES = {
    # Profil pełny — 5 poziomów uszkodzeń, metody KNN, 3 powtórzenia
    "full": {
        "datasets": DATASETS_ALL,
        "methods": ["raw", "fill", "remove_fill", "fill_norm", "all", "fill_knn", "all_knn"],
        "models": ["RF", "NB", "MLP", "XGBoost"],
        "damage_levels": [10, 20, 30, 40, 50],
        "damage_repeats": [1, 2, 3],
        "cv_states": [101, 202, 303],
        "cv_folds": 4,
        "output_suffix": "_full",
        "dirty_file_pattern": "{name}_full_prob_{level}_r{repeat}.csv",
        # Parametry generowania uszkodzeń
        "damage_seed_base": 5000,
        "outlier_factors": [20, 40, 80, -20],
        "outlier_rate_scale": 0.8,    # outlier_count = max(0.05, damage * scale)
        "outlier_rate_floor": 0.05,
        "noise_rate_scale": 0.8,
        "noise_rate_floor": 0.05,
    },
    # Profil szybki — 3 poziomy, 2 powtórzenia, mniej modeli
    "fast": {
        "datasets": DATASETS_ALL,
        "methods": ["raw", "fill", "remove_fill", "fill_norm", "all"],
        "models": ["RF", "NB", "XGBoost"],
        "damage_levels": [20, 40, 60],
        "damage_repeats": [1, 2],
        "cv_states": [101, 202],
        "cv_folds": 3,
        "output_suffix": "_fast",
        "dirty_file_pattern": "{name}_fast_prob_{level}_r{repeat}.csv",
        # Parametry generowania uszkodzeń
        "damage_seed_base": 1000,
        "outlier_factors": [25, 50, 100, -25],
        "outlier_rate_scale": 1.0,    # outlier_count = damage_level * 1.0 (bezpośrednio)
        "outlier_rate_floor": 0.0,
        "noise_rate_scale": 1.0,
        "noise_rate_floor": 0.0,
    },
}

PARALLEL_JOBS = max(1, (os.cpu_count() or 4) - 2)
