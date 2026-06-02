"""Tests for mini_ml.metrics module."""

import numpy as np
import pytest

from mini_ml.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# ── Classification Metrics ────────────────────────────────────────────────

class TestAccuracyScore:
    """Tests for accuracy_score."""

    def test_perfect(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert accuracy_score(y_true, y_pred) == 1.0

    def test_half_correct(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        assert accuracy_score(y_true, y_pred) == 0.75

    def test_all_wrong(self):
        y_true = np.array([0, 1])
        y_pred = np.array([1, 0])
        assert accuracy_score(y_true, y_pred) == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            accuracy_score(np.array([]), np.array([]))

    def test_mismatched_shapes_raises(self):
        with pytest.raises(ValueError, match="Mismatched"):
            accuracy_score(np.array([0, 1]), np.array([0, 1, 1]))


class TestPrecisionScore:
    """Tests for precision_score."""

    def test_binary_perfect(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        assert precision_score(y_true, y_pred) == 1.0

    def test_binary_no_true_positives(self):
        y_true = np.array([1, 1])
        y_pred = np.array([0, 0])
        assert precision_score(y_true, y_pred) == 0.0

    def test_macro_average(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 2])
        p = precision_score(y_true, y_pred, average="macro")
        assert 0.0 <= p <= 1.0

    def test_micro_average(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        p = precision_score(y_true, y_pred, average="micro")
        assert p == 0.5

    def test_weighted_average(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        p = precision_score(y_true, y_pred, average="weighted")
        assert 0.0 <= p <= 1.0

    def test_unknown_average_raises(self):
        with pytest.raises(ValueError, match="Unknown average"):
            precision_score(np.array([0, 1]), np.array([0, 1]), average="unknown")


class TestRecallScore:
    """Tests for recall_score."""

    def test_binary_perfect(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        assert recall_score(y_true, y_pred) == 1.0

    def test_binary_missed_positives(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        assert recall_score(y_true, y_pred) == 0.0

    def test_macro_average(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 2])
        r = recall_score(y_true, y_pred, average="macro")
        assert 0.0 <= r <= 1.0

    def test_unknown_average_raises(self):
        with pytest.raises(ValueError, match="Unknown average"):
            recall_score(np.array([0, 1]), np.array([0, 1]), average="unknown")


class TestF1Score:
    """Tests for f1_score."""

    def test_binary_perfect(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        assert f1_score(y_true, y_pred) == 1.0

    def test_binary_worst(self):
        y_true = np.array([1, 1])
        y_pred = np.array([0, 0])
        assert f1_score(y_true, y_pred) == 0.0

    def test_macro_average(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 2])
        f = f1_score(y_true, y_pred, average="macro")
        assert 0.0 <= f <= 1.0

    def test_mismatched_shapes_raises(self):
        with pytest.raises(ValueError, match="Mismatched"):
            f1_score(np.array([0, 1]), np.array([0]))


class TestConfusionMatrix:
    """Tests for confusion_matrix."""

    def test_binary(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        cm = confusion_matrix(y_true, y_pred)
        expected = np.array([[1, 1], [0, 2]])
        np.testing.assert_array_equal(cm, expected)

    def test_with_custom_labels(self):
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1, 1])
        cm = confusion_matrix(y_true, y_pred, labels=np.array([1, 0]))
        expected = np.array([[1, 0], [1, 1]])
        np.testing.assert_array_equal(cm, expected)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            confusion_matrix(np.array([]), np.array([]))


# ── Regression Metrics ────────────────────────────────────────────────────

class TestMeanSquaredError:
    """Tests for mean_squared_error."""

    def test_perfect(self):
        y_true = np.array([3.0, 5.0, 2.5])
        y_pred = np.array([3.0, 5.0, 2.5])
        assert mean_squared_error(y_true, y_pred) == 0.0

    def test_known_result(self):
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        expected = np.mean((y_true - y_pred) ** 2)
        assert mean_squared_error(y_true, y_pred) == pytest.approx(expected)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            mean_squared_error(np.array([]), np.array([]))


class TestMeanAbsoluteError:
    """Tests for mean_absolute_error."""

    def test_perfect(self):
        y_true = np.array([3.0, 5.0, 2.5])
        y_pred = np.array([3.0, 5.0, 2.5])
        assert mean_absolute_error(y_true, y_pred) == 0.0

    def test_known_result(self):
        y_true = np.array([3.0, 5.0, 2.5])
        y_pred = np.array([2.5, 5.0, 2.0])
        expected = np.mean(np.abs(y_true - y_pred))
        assert mean_absolute_error(y_true, y_pred) == pytest.approx(expected)


class TestR2Score:
    """Tests for r2_score."""

    def test_perfect(self):
        y_true = np.array([3.0, 5.0, 2.5, 7.0])
        y_pred = np.array([3.0, 5.0, 2.5, 7.0])
        assert r2_score(y_true, y_pred) == 1.0

    def test_worse_than_mean(self):
        y_true = np.array([3.0, 5.0, 2.5])
        y_pred = np.array([100.0, -50.0, 200.0])
        r2 = r2_score(y_true, y_pred)
        assert r2 < 0.0

    def test_single_sample_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            r2_score(np.array([1.0]), np.array([1.0]))
