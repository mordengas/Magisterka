import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class OutlierToNaNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_columns, iqr_multiplier=3.0, min_unique=10):
        self.numeric_columns = numeric_columns
        self.iqr_multiplier = iqr_multiplier
        self.min_unique = min_unique
        self.bounds_ = {}

    def fit(self, X, y=None):
        df = X.copy()
        self.bounds_ = {}

        for col in list(self.numeric_columns):
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
    def __init__(self, categorical_columns, min_frequency=0.05):
        self.categorical_columns = categorical_columns
        self.min_frequency = min_frequency
        self.valid_categories_ = {}

    def fit(self, X, y=None):
        df = X.copy()
        self.valid_categories_ = {}

        for col in list(self.categorical_columns):
            if col not in df.columns:
                continue

            freq = df[col].value_counts(normalize=True, dropna=True)
            valid = set(freq[freq >= self.min_frequency].index.tolist())
            self.valid_categories_[col] = valid

        return self

    def transform(self, X):
        df = X.copy()
        for col, valid_values in self.valid_categories_.items():
            if col not in df.columns or not valid_values:
                continue

            mask = df[col].notna() & ~df[col].isin(valid_values)
            df.loc[mask, col] = np.nan

        return df


def build_cleaning_pipeline(X, method_name):
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]

    cleaning_steps = []
    if method_name in {"remove", "remove_fill", "remove_norm", "all"}:
        cleaning_steps.extend([
            ("remove_outliers", OutlierToNaNTransformer(numeric_columns=numeric_columns)),
            ("remove_rare_categories", RareCategoryToNaNTransformer(categorical_columns=categorical_columns)),
        ])

    numeric_imputer_strategy = "constant"
    categorical_imputer_strategy = "constant"
    numeric_fill_value = -999.0
    categorical_fill_value = "MISSING"

    if method_name in {"fill", "fill_norm", "remove_fill", "all"}:
        numeric_imputer_strategy = "median"
        categorical_imputer_strategy = "most_frequent"
        numeric_fill_value = None
        categorical_fill_value = None

    numeric_steps = []
    if numeric_imputer_strategy == "constant":
        numeric_steps.append(
            ("imputer", SimpleImputer(strategy="constant", fill_value=numeric_fill_value))
        )
    else:
        numeric_steps.append(("imputer", SimpleImputer(strategy=numeric_imputer_strategy)))

    if method_name in {"norm", "fill_norm", "remove_norm", "all"}:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = []
    if categorical_imputer_strategy == "constant":
        categorical_steps.append(
            ("imputer", SimpleImputer(strategy="constant", fill_value=categorical_fill_value))
        )
    else:
        categorical_steps.append(("imputer", SimpleImputer(strategy=categorical_imputer_strategy)))
    categorical_steps.append(("encoder", make_one_hot_encoder()))

    transformers = []
    if numeric_columns:
        transformers.append(("num", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        transformers.append(("cat", Pipeline(categorical_steps), categorical_columns))

    preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")

    return Pipeline(cleaning_steps + [("preprocess", preprocessing)])


def apply_strategy_globally(df, target_col, method_name, drop_columns=None):
    drop_columns = drop_columns or []
    feature_df = df.drop(columns=[target_col, *drop_columns], errors="ignore")
    prep = build_cleaning_pipeline(feature_df, method_name)
    transformed = prep.fit_transform(feature_df)

    feature_names = prep.named_steps["preprocess"].get_feature_names_out()
    transformed_df = pd.DataFrame(transformed, columns=feature_names, index=df.index)
    transformed_df[target_col] = df[target_col].values
    return transformed_df
