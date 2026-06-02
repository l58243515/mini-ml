"""Tests for mini_ml.split module."""

import numpy as np
import pytest

from mini_ml.split import train_test_split, kfold_split


# ── train_test_split ──────────────────────────────────────────────────────

class TestTrainTestSplit:
    """Tests for train_test_split."""

    def test_shapes_single_array(self):
        X = np.arange(20).reshape(10, 2)
        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
        assert X_train.shape[0] == 7
        assert X_test.shape[0] == 3
        assert X_train.shape[1] == 2
        assert X_test.shape[1] == 2

    def test_shapes_multiple_arrays(self):
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)
        result = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train, X_test, y_train, y_test = result
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert X_train.shape[0] == 7
        assert X_test.shape[0] == 3

    def test_reproducibility(self):
        X = np.arange(100).reshape(50, 2)
        X_train1, X_test1 = train_test_split(X, test_size=0.2, random_state=42)
        X_train2, X_test2 = train_test_split(X, test_size=0.2, random_state=42)
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)

    def test_different_random_state_different_split(self):
        X = np.arange(100).reshape(50, 2)
        X_train1, _ = train_test_split(X, test_size=0.2, random_state=42)
        X_train2, _ = train_test_split(X, test_size=0.2, random_state=123)
        assert not np.array_equal(X_train1, X_train2)

    def test_no_shuffle(self):
        X = np.arange(20).reshape(10, 2)
        X_train, X_test = train_test_split(X, test_size=0.3, shuffle=False)
        np.testing.assert_array_equal(X_train, X[:7])
        np.testing.assert_array_equal(X_test, X[7:])

    def test_integer_test_size(self):
        X = np.arange(20).reshape(10, 2)
        X_train, X_test = train_test_split(X, test_size=3, random_state=42)
        assert X_test.shape[0] == 3
        assert X_train.shape[0] == 7

    def test_float_test_size_invalid_range(self):
        X = np.arange(20).reshape(10, 2)
        with pytest.raises(ValueError, match="test_size"):
            train_test_split(X, test_size=0.0)
        with pytest.raises(ValueError, match="test_size"):
            train_test_split(X, test_size=1.0)

    def test_integer_test_size_invalid(self):
        X = np.arange(20).reshape(10, 2)
        with pytest.raises(ValueError, match="test_size"):
            train_test_split(X, test_size=0)
        with pytest.raises(ValueError, match="test_size"):
            train_test_split(X, test_size=10)

    def test_empty_arrays_raises(self):
        X = np.array([])
        with pytest.raises(ValueError, match="empty"):
            train_test_split(X)

    def test_mismatched_lengths_raises(self):
        X = np.arange(20).reshape(10, 2)
        y = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="Mismatched"):
            train_test_split(X, y)

    def test_no_arrays_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            train_test_split()


# ── kfold_split ───────────────────────────────────────────────────────────

class TestKfoldSplit:
    """Tests for kfold_split."""

    def test_num_folds(self):
        folds = kfold_split(10, n_splits=5, shuffle=False)
        assert len(folds) == 5

    def test_all_samples_covered(self):
        n_samples = 10
        folds = kfold_split(n_samples, n_splits=5, shuffle=False)
        all_test = np.concatenate([fold[1] for fold in folds])
        np.testing.assert_array_equal(np.sort(all_test), np.arange(n_samples))

    def test_train_test_disjoint(self):
        folds = kfold_split(15, n_splits=3, shuffle=False)
        for train_idx, test_idx in folds:
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_shuffle_reproducibility(self):
        folds1 = kfold_split(10, n_splits=3, shuffle=True, random_state=42)
        folds2 = kfold_split(10, n_splits=3, shuffle=True, random_state=42)
        for (t1, v1), (t2, v2) in zip(folds1, folds2):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(v1, v2)

    def test_uneven_folds(self):
        folds = kfold_split(11, n_splits=3, shuffle=False)
        test_sizes = [len(fold[1]) for fold in folds]
        assert sum(test_sizes) == 11
        assert max(test_sizes) - min(test_sizes) <= 1

    def test_n_splits_too_small_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            kfold_split(10, n_splits=1)

    def test_n_splits_too_large_raises(self):
        with pytest.raises(ValueError, match="cannot be greater"):
            kfold_split(5, n_splits=10)

    def test_invalid_n_samples_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            kfold_split(0)
