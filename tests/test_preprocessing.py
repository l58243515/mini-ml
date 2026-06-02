"""Tests for mini_ml.preprocessing module."""

import numpy as np
import pytest

from mini_ml.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


# ── StandardScaler ────────────────────────────────────────────────────────

class TestStandardScaler:
    """Tests for StandardScaler."""

    def test_fit_sets_mean_and_std(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        scaler.fit(X)
        assert scaler.mean_ is not None
        assert scaler.std_ is not None
        np.testing.assert_array_almost_equal(scaler.mean_, [3.0, 4.0])
        expected_std = np.std(X, axis=0, ddof=0)
        np.testing.assert_array_almost_equal(scaler.std_, expected_std)

    def test_transform_returns_scaled_data(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        np.testing.assert_array_almost_equal(np.mean(X_scaled, axis=0), [0.0, 0.0])
        np.testing.assert_array_almost_equal(np.std(X_scaled, axis=0, ddof=0), [1.0, 1.0])

    def test_fit_transform(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        np.testing.assert_array_almost_equal(np.mean(X_scaled, axis=0), [0.0, 0.0])
        np.testing.assert_array_almost_equal(np.std(X_scaled, axis=0, ddof=0), [1.0, 1.0])

    def test_inverse_transform(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_back = scaler.inverse_transform(X_scaled)
        np.testing.assert_array_almost_equal(X_back, X)

    def test_1d_input(self):
        X_1d = np.array([1.0, 2.0, 3.0])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_1d)
        assert X_scaled.ndim == 2
        assert X_scaled.shape[1] == 1
        np.testing.assert_array_almost_equal(np.mean(X_scaled, axis=0), [0.0])

    def test_single_sample(self):
        X = np.array([[5.0, 10.0]])
        with pytest.raises(ValueError, match="zero or near-zero standard deviation"):
            StandardScaler().fit(X)

    def test_constant_feature_raises(self):
        X = np.array([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])
        with pytest.raises(ValueError, match="zero or near-zero standard deviation"):
            StandardScaler().fit(X)

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="empty"):
            StandardScaler().fit(np.array([]))

    def test_transform_before_fit_raises(self):
        X = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="not fitted"):
            StandardScaler().transform(X)

    def test_inverse_transform_before_fit_raises(self):
        X = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="not fitted"):
            StandardScaler().inverse_transform(X)


# ── MinMaxScaler ──────────────────────────────────────────────────────────

class TestMinMaxScaler:
    """Tests for MinMaxScaler."""

    def test_default_range(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        np.testing.assert_array_almost_equal(np.min(X_scaled, axis=0), [0.0, 0.0])
        np.testing.assert_array_almost_equal(np.max(X_scaled, axis=0), [1.0, 1.0])

    def test_custom_range(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        np.testing.assert_array_almost_equal(np.min(X_scaled, axis=0), [-1.0, -1.0])
        np.testing.assert_array_almost_equal(np.max(X_scaled, axis=0), [1.0, 1.0])

    def test_inverse_transform(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_back = scaler.inverse_transform(X_scaled)
        np.testing.assert_array_almost_equal(X_back, X)

    def test_1d_input(self):
        X_1d = np.array([1.0, 2.0, 3.0])
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_1d)
        assert X_scaled.ndim == 2
        assert X_scaled.shape[1] == 1

    def test_constant_feature_raises(self):
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        with pytest.raises(ValueError, match="zero range"):
            MinMaxScaler().fit(X)

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="empty"):
            MinMaxScaler().fit(np.array([]))

    def test_transform_before_fit_raises(self):
        X = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="not fitted"):
            MinMaxScaler().transform(X)


# ── LabelEncoder ──────────────────────────────────────────────────────────

class TestLabelEncoder:
    """Tests for LabelEncoder."""

    def test_fit_learns_classes(self):
        y = np.array(["cat", "dog", "cat", "bird"])
        le = LabelEncoder()
        le.fit(y)
        np.testing.assert_array_equal(le.classes_, np.array(["bird", "cat", "dog"]))
        assert le.class_to_index_ == {"bird": 0, "cat": 1, "dog": 2}

    def test_transform_integers(self):
        y = np.array([10, 20, 10, 30])
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        np.testing.assert_array_equal(y_encoded, np.array([0, 1, 0, 2]))

    def test_inverse_transform(self):
        y = np.array(["cat", "dog", "cat", "bird"])
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        y_back = le.inverse_transform(y_encoded)
        np.testing.assert_array_equal(y_back, y)

    def test_list_input(self):
        le = LabelEncoder()
        y_encoded = le.fit_transform(["a", "b", "a", "c"])
        np.testing.assert_array_equal(y_encoded, np.array([0, 1, 0, 2]))

    def test_unseen_label_raises(self):
        le = LabelEncoder()
        le.fit(np.array(["cat", "dog"]))
        with pytest.raises(ValueError, match="were not seen"):
            le.transform(np.array(["cat", "bird"]))

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            LabelEncoder().fit(np.array([]))

    def test_transform_before_fit_raises(self):
        with pytest.raises(ValueError, match="not fitted"):
            LabelEncoder().transform(np.array([1, 2]))
