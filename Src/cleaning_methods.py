import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class OutlierToNaNTransformer(BaseEstimator, TransformerMixin):
    """Zamienia wartości odstające na NaN na podstawie metody IQR.

    Parametry domyślne uzasadnione literaturą:
        iqr_multiplier=3.0 — szeroki zakres (vs typowe 1.5), by usuwać
            tylko skrajne outlier'y, nie normalne wartości brzegowe.
        min_unique=10 — kolumny z <10 unikalnymi wartościami traktowane
            jako kategoryczne (pomijane w detekcji outlierów IQR).
    """
    def __init__(self, numeric_columns=None, iqr_multiplier=3.0, min_unique=10):
        self.numeric_columns = numeric_columns or []
        self.iqr_multiplier = iqr_multiplier
        self.min_unique = min_unique
        self.bounds_ = {}

    def fit(self, X, y=None):
        df = X.copy()
        self.bounds_ = {}

        cols_to_check = (
            list(self.numeric_columns)
            if self.numeric_columns
            else df.select_dtypes(include=[np.number]).columns.tolist()
        )

        for col in cols_to_check:
            if col not in df.columns:
                continue

            series = pd.to_numeric(df[col], errors="coerce")
            if series.nunique(dropna=True) < self.min_unique:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue

            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            self.bounds_[col] = (lower, upper)

        return self

    def transform(self, X):
        df = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            if col not in df.columns:
                continue

            series = pd.to_numeric(df[col], errors="coerce")
            mask = (series < lower) | (series > upper)
            df.loc[mask, col] = np.nan

        return df


class RareCategoryToNaNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns=None, min_frequency=0.05):
        self.categorical_columns = categorical_columns or []
        self.min_frequency = min_frequency
        self.valid_categories_ = {}

    def fit(self, X, y=None):
        df = X.copy()
        self.valid_categories_ = {}

        cols_to_check = (
            list(self.categorical_columns)
            if self.categorical_columns
            else [c for c in df.columns if c not in df.select_dtypes(include=[np.number]).columns]
        )

        for col in cols_to_check:
            if col not in df.columns:
                continue

            # Rzutowanie na string ujednolica kategorie zapisane liczbami (np. 0.0, 1.0, 9) i tekstami
            series_str = df[col].dropna().astype(str)
            if len(series_str) == 0:
                continue

            freq = series_str.value_counts(normalize=True)
            valid = set(freq[freq >= self.min_frequency].index.tolist())
            self.valid_categories_[col] = valid

        return self

    def transform(self, X):
        df = X.copy()
        for col, valid_values in self.valid_categories_.items():
            if col not in df.columns or not valid_values:
                continue

            series_str = df[col].astype(str)
            mask = df[col].notna() & ~series_str.isin(valid_values)
            df.loc[mask, col] = np.nan

        return df


def build_cleaning_pipeline(X, method_name, continuous_columns=None, categorical_columns=None):
    if continuous_columns is not None:
        numeric_columns = [col for col in continuous_columns if col in X.columns]
    else:
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

    if categorical_columns is not None:
        cat_columns = [col for col in categorical_columns if col in X.columns]
    else:
        cat_columns = [col for col in X.columns if col not in numeric_columns]

    cleaning_steps = []
    if method_name in {"remove", "remove_fill", "remove_norm", "all", "all_knn"}:
        if numeric_columns:
            cleaning_steps.append(
                ("remove_outliers", OutlierToNaNTransformer(numeric_columns=numeric_columns))
            )
        if cat_columns:
            cleaning_steps.append(
                ("remove_rare_categories", RareCategoryToNaNTransformer(categorical_columns=cat_columns))
            )

    # Dla metod bez imputacji (np. "raw", "norm", "remove"):
    # - kolumny numeryczne zachowują NaN (lub są skalowane przez StandardScaler)
    # - kolumny kategoryczne brakujące wartości oznaczają jako "MISSING"
    # Dla metod z imputacją ("fill", "all", itp.):
    # - numeryczne uzupełniane medianą lub KNN
    # - kategoryczne uzupełniane dominantą (most_frequent)
    do_imputation = method_name in {"fill", "fill_norm", "remove_fill", "all", "fill_knn", "all_knn"}

    numeric_steps = [
        ("to_float", FunctionTransformer(lambda x: x.astype(float), validate=False, feature_names_out="one-to-one")),
    ]
    if method_name in {"fill_knn", "all_knn"}:
        numeric_steps.append(("imputer", KNNImputer(n_neighbors=5)))
    elif do_imputation:
        numeric_steps.append(("imputer", SimpleImputer(strategy="median")))

    if method_name in {"norm", "fill_norm", "remove_norm", "all", "all_knn"}:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("to_obj", FunctionTransformer(lambda x: x.astype(object), validate=False, feature_names_out="one-to-one")),
    ]
    if do_imputation:
        categorical_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
    else:
        categorical_steps.append(
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING"))
        )

    # Konwersja na string przed OneHotEncoder zapobiega błędowi mieszanych typów (float + str)
    categorical_steps.append(
        ("to_str", FunctionTransformer(lambda x: x.astype(str), validate=False, feature_names_out="one-to-one"))
    )
    categorical_steps.append(("encoder", make_one_hot_encoder()))

    transformers = []
    if numeric_columns:
        transformers.append(("num", Pipeline(numeric_steps), numeric_columns))
    if cat_columns:
        transformers.append(("cat", Pipeline(categorical_steps), cat_columns))

    preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")

    return Pipeline(cleaning_steps + [("preprocess", preprocessing)])


def apply_strategy_globally(df, target_col, method_name, drop_columns=None, continuous_columns=None, categorical_columns=None):
    drop_columns = drop_columns or []
    feature_df = df.drop(columns=[target_col, *drop_columns], errors="ignore")
    prep = build_cleaning_pipeline(
        feature_df,
        method_name,
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
    )
    transformed = prep.fit_transform(feature_df)

    feature_names = prep.named_steps["preprocess"].get_feature_names_out()
    transformed_df = pd.DataFrame(transformed, columns=feature_names, index=df.index)
    transformed_df[target_col] = df[target_col].values
    return transformed_df
