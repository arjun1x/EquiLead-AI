"""Tests for feature schema definitions."""

from src.data.schema import (
    FEATURE_SCHEMA, TARGET_COLUMN, MODEL_FEATURES,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, NULLABLE_FEATURES,
)


def test_target_column():
    assert TARGET_COLUMN == "BAD"


def test_feature_count():
    assert len(FEATURE_SCHEMA) == 12


def test_model_features_match_schema():
    schema_names = [f.name for f in FEATURE_SCHEMA]
    assert MODEL_FEATURES == schema_names


def test_numeric_features():
    assert "LOAN" in NUMERIC_FEATURES
    assert "DEBTINC" in NUMERIC_FEATURES
    assert "REASON" not in NUMERIC_FEATURES
    assert len(NUMERIC_FEATURES) == 10


def test_categorical_features():
    assert "REASON" in CATEGORICAL_FEATURES
    assert "JOB" in CATEGORICAL_FEATURES
    assert len(CATEGORICAL_FEATURES) == 2


def test_nullable_features():
    assert "LOAN" not in NULLABLE_FEATURES
    assert "DEBTINC" in NULLABLE_FEATURES
    assert len(NULLABLE_FEATURES) == 11


def test_no_protected_class_features():
    for f in FEATURE_SCHEMA:
        assert not hasattr(f, "protected_class") or not getattr(f, "protected_class", False)
