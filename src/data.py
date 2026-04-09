"""Carregamento e pré-processamento dos CSVs APS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def load_raw(training_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(training_path, na_values=["na"])
    test = pd.read_csv(test_path, na_values=["na"])
    return train, test


def encode_target(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    label_map = {"neg": 0, "pos": 1}
    y_train = train["class"].map(label_map).to_numpy()
    y_test = test["class"].map(label_map).to_numpy()
    X_train = train.drop(columns=["class"])
    X_test = test.drop(columns=["class"])
    return X_train, X_test, y_train.astype(np.int64), y_test.astype(np.int64)


def drop_degenerate_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove colunas constantes ou totalmente ausentes no treino (após leitura)."""
    all_nan = X_train.columns[X_train.isna().all()].tolist()
    nunique = X_train.nunique(dropna=True)
    constant = nunique[nunique <= 1].index.tolist()
    drop_cols = sorted(set(all_nan + constant))
    if drop_cols:
        X_train = X_train.drop(columns=drop_cols)
        X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns], errors="ignore")
    return X_train, X_test


def fit_imputer(X_train: pd.DataFrame) -> SimpleImputer:
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train.to_numpy())
    return imputer


def transform_imputer(imputer: SimpleImputer, X: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return imputer.transform(X[columns].to_numpy())


def ensure_numeric_feature_names(X: pd.DataFrame) -> list[str]:
    return list(X.columns)
