"""Tests for data loading and preprocessing."""

import pandas as pd
import numpy as np

from src.data.schema import TARGET_COLUMN, MODEL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.ml.preprocessing import load_raw, preprocess, split


def test_load_raw():
    df = load_raw()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5960
    assert TARGET_COLUMN in df.columns
    for feat in MODEL_FEATURES:
        assert feat in df.columns


def test_load_raw_has_expected_columns():
    df = load_raw()
    expected = [TARGET_COLUMN] + MODEL_FEATURES
    for col in expected:
        assert col in df.columns


def test_preprocess_no_nulls():
    df = load_raw()
    processed, encoders = preprocess(df)
    assert processed[MODEL_FEATURES].isnull().sum().sum() == 0


def test_preprocess_encodes_categoricals():
    df = load_raw()
    processed, encoders = preprocess(df)
    for col in CATEGORICAL_FEATURES:
        assert col in encoders
        assert processed[col].dtype in [np.int32, np.int64, np.intp]


def test_preprocess_preserves_target():
    df = load_raw()
    processed, _ = preprocess(df)
    assert set(processed[TARGET_COLUMN].unique()) == {0, 1}
    assert len(processed) == len(df)


def test_split_stratified():
    df = load_raw()
    processed, _ = preprocess(df)
    X_train, X_test, y_train, y_test = split(processed)

    assert len(X_train) + len(X_test) == len(processed)
    assert len(X_test) == int(len(processed) * 0.2) or abs(len(X_test) - len(processed) * 0.2) <= 1

    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.02


def test_split_features_only():
    df = load_raw()
    processed, _ = preprocess(df)
    X_train, X_test, _, _ = split(processed)
    assert TARGET_COLUMN not in X_train.columns
    assert TARGET_COLUMN not in X_test.columns
    assert list(X_train.columns) == MODEL_FEATURES
