"""
Data loading and preprocessing pipeline for the HMEQ dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from src.data.schema import (
    TARGET_COLUMN, MODEL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)

RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "hmeq.csv"


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    num_imputer = SimpleImputer(strategy="median")
    df[NUMERIC_FEATURES] = num_imputer.fit_transform(df[NUMERIC_FEATURES])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[CATEGORICAL_FEATURES] = cat_imputer.fit_transform(df[CATEGORICAL_FEATURES])

    label_encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders


def split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[MODEL_FEATURES]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
