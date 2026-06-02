"""Tests for mini_ml.visualize module."""

import numpy as np
import pytest

from mini_ml.visualize import plot_confusion_matrix, classification_report_text


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix."""

    def test_text_output_contains_title(self):
        cm = np.array([[5, 1], [0, 4]])
        result = plot_confusion_matrix(cm, as_text=True)
        assert "Confusion Matrix" in result

    def test_text_output_contains_labels(self):
        cm = np.array([[5, 1], [0, 4]])
        result = plot_confusion_matrix(cm, labels=["Neg", "Pos"], as_text=True)
        assert "Neg" in result
        assert "Pos" in result

    def test_text_output_contains_values(self):
        cm = np.array([[5, 1], [0, 4]])
        result = plot_confusion_matrix(cm, as_text=True)
        assert "5" in result
        assert "1" in result
        assert "0" in result
        assert "4" in result

    def test_non_square_raises(self):
        cm = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match="square"):
            plot_confusion_matrix(cm, as_text=True)

    def test_mismatched_labels_raises(self):
        cm = np.array([[1, 1], [0, 2]])
        with pytest.raises(ValueError, match="Number of labels"):
            plot_confusion_matrix(cm, labels=["A"], as_text=True)

    def test_default_labels_numeric(self):
        cm = np.array([[5, 1], [0, 4]])
        result = plot_confusion_matrix(cm, as_text=True)
        assert "0" in result
        assert "1" in result


class TestClassificationReportText:
    """Tests for classification_report_text."""

    def test_output_contains_headers(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        report = classification_report_text(y_true, y_pred)
        assert "precision" in report
        assert "recall" in report
        assert "f1-score" in report
        assert "support" in report

    def test_output_contains_macro_avg(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 2])
        report = classification_report_text(y_true, y_pred)
        assert "macro" in report

    def test_output_contains_weighted_avg(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 2])
        report = classification_report_text(y_true, y_pred)
        assert "weighted" in report

    def test_output_contains_class_labels(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        report = classification_report_text(y_true, y_pred)
        assert "0" in report
        assert "1" in report
