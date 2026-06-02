"""Tests for mini_ml.evaluation module."""

import numpy as np
import pytest

from mini_ml.evaluation import cross_validate, evaluate_model


class DummyEstimator:
    """A simple estimator that memorizes labels and predicts based on nearest neighbor."""

    def __init__(self):
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        # Simple nearest-neighbor approach
        predictions = np.empty(len(X), dtype=self.y_train_.dtype)
        for i, x in enumerate(X):
            distances = np.sum((self.X_train_ - x) ** 2, axis=1)
            nearest = np.argmin(distances)
            predictions[i] = self.y_train_[nearest]
        return predictions


class TestCrossValidate:
    """Tests for cross_validate."""

    def test_returns_correct_structure(self):
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        results = cross_validate(DummyEstimator(), X, y, cv=3, random_state=42)
        assert "test_accuracy" in results
        assert "train_accuracy" in results
        assert len(results["test_accuracy"]) == 3
        assert len(results["train_accuracy"]) == 3

    def test_scores_in_valid_range(self):
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        results = cross_validate(DummyEstimator(), X, y, cv=3, random_state=42)
        for score in results["test_accuracy"]:
            assert 0.0 <= score <= 1.0

    def test_unsupported_scoring_raises(self):
        X = np.array([[1], [2], [3]])
        y = np.array([0, 1, 0])
        with pytest.raises(ValueError, match="Unsupported scoring"):
            cross_validate(DummyEstimator(), X, y, scoring="f1")

    def test_mismatched_shapes_raises(self):
        X = np.array([[1], [2], [3]])
        y = np.array([0, 1])
        with pytest.raises(ValueError, match="Mismatched"):
            cross_validate(DummyEstimator(), X, y)

    def test_empty_arrays_raises(self):
        X = np.array([])
        y = np.array([])
        with pytest.raises(ValueError, match="empty"):
            cross_validate(DummyEstimator(), X, y)


class TestEvaluateModel:
    """Tests for evaluate_model."""

    def test_returns_expected_keys(self):
        X_train = np.array([[1], [2], [3]])
        X_test = np.array([[1.5], [2.5]])
        y_train = np.array([0, 1, 1])
        y_test = np.array([0, 1])
        result = evaluate_model(DummyEstimator(), X_train, X_test, y_train, y_test)
        assert "accuracy" in result
        assert "y_pred" in result
        assert "y_true" in result
        assert len(result["y_pred"]) == len(y_test)

    def test_accuracy_calculation(self):
        X_train = np.array([[1.0], [3.0], [5.0], [7.0], [9.0]])
        X_test = np.array([[1.1], [5.1], [9.1]])
        y_train = np.array([0, 1, 0, 1, 0])
        y_test = np.array([0, 0, 0])
        result = evaluate_model(DummyEstimator(), X_train, X_test, y_train, y_test)
        assert isinstance(result["accuracy"], float)

    def test_mismatched_train_shapes_raises(self):
        with pytest.raises(ValueError, match="Mismatched train"):
            evaluate_model(
                DummyEstimator(),
                np.array([[1], [2]]),
                np.array([[3]]),
                np.array([0, 1, 2]),
                np.array([1]),
            )

    def test_empty_train_raises(self):
        with pytest.raises(ValueError, match="empty"):
            evaluate_model(
                DummyEstimator(),
                np.array([]),
                np.array([[1]]),
                np.array([]),
                np.array([1]),
            )
