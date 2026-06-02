"""Model evaluation utilities for cross-validation and scoring."""

import numpy as np

from .metrics import accuracy_score
from .split import kfold_split


_SUPPORTED_SCORING = {
    "accuracy": accuracy_score,
}

_METRIC_NAMES = {
    "accuracy": "accuracy",
}


def _resolve_scoring(scoring):
    """Resolve a scoring name or list of names to callables and their keys."""
    if isinstance(scoring, str):
        scoring = [scoring]
    resolved = {}
    for s in scoring:
        if s not in _SUPPORTED_SCORING:
            raise ValueError(
                f"Unsupported scoring '{s}'. "
                f"Supported: {list(_SUPPORTED_SCORING.keys())}."
            )
        resolved[s] = _SUPPORTED_SCORING[s]
    return resolved


def cross_validate(estimator, X, y, cv=5, scoring="accuracy", random_state=None):
    """Evaluate metric(s) by cross-validation.

    Performs ``cv``-fold cross-validation, fitting the estimator on the
    training fold and evaluating on the test fold for each split.

    Parameters
    ----------
    estimator : object
        Any object with ``.fit(X, y)`` and ``.predict(X)`` methods.
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target labels.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str or list of str, default='accuracy'
        Scoring metric(s) to evaluate. Currently supports 'accuracy'.
    random_state : int, optional
        Seed for reproducible fold generation.

    Returns
    -------
    dict
        Dictionary with keys like ``'test_accuracy'`` and ``'train_accuracy'``.
        Each value is a list of scores, one per fold.

    Examples
    --------
    >>> import numpy as np
    >>> from mini_ml.evaluation import cross_validate
    >>>
    >>> class DummyEstimator:
    ...     def fit(self, X, y):
    ...         self.X_ = X
    ...         self.y_ = y
    ...         return self
    ...     def predict(self, X):
    ...         return np.zeros(len(X), dtype=np.int64)
    >>>
    >>> X = np.array([[1], [2], [3], [4], [5], [6]])
    >>> y = np.array([0, 1, 0, 1, 0, 1])
    >>> results = cross_validate(DummyEstimator(), X, y, cv=3, random_state=42)
    >>> 'test_accuracy' in results
    True
    >>> len(results['test_accuracy'])
    3
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Mismatched samples: X has {X.shape[0]} samples, "
            f"y has {y.shape[0]} samples."
        )
    if X.shape[0] == 0:
        raise ValueError("Input arrays must not be empty.")

    scoring_funcs = _resolve_scoring(scoring)
    folds = kfold_split(X.shape[0], n_splits=cv, shuffle=True,
                         random_state=random_state)

    results = {}
    for key in scoring_funcs:
        results[f"test_{key}"] = []
        results[f"train_{key}"] = []

    for train_idx, test_idx in folds:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        estimator.fit(X_train, y_train)
        y_pred_train = estimator.predict(X_train)
        y_pred_test = estimator.predict(X_test)

        for key, scorer in scoring_funcs.items():
            results[f"train_{key}"].append(
                float(scorer(y_train, y_pred_train))
            )
            results[f"test_{key}"].append(
                float(scorer(y_test, y_pred_test))
            )

    return results


def evaluate_model(estimator, X_train, X_test, y_train, y_test):
    """Fit estimator on train data, predict on test data, and return metrics.

    Parameters
    ----------
    estimator : object
        Any object with ``.fit(X, y)`` and ``.predict(X)`` methods.
    X_train : np.ndarray of shape (n_train, n_features)
        Training feature matrix.
    X_test : np.ndarray of shape (n_test, n_features)
        Test feature matrix.
    y_train : np.ndarray of shape (n_train,)
        Training labels.
    y_test : np.ndarray of shape (n_test,)
        Test labels.

    Returns
    -------
    dict
        Dictionary with keys: 'accuracy', 'y_pred', 'y_true'.

    Examples
    --------
    >>> import numpy as np
    >>> from mini_ml.evaluation import evaluate_model
    >>>
    >>> class DummyEstimator:
    ...     def fit(self, X, y):
    ...         return self
    ...     def predict(self, X):
    ...         return np.ones(len(X), dtype=np.int64)
    >>>
    >>> X_train, X_test = np.array([[1], [2]]), np.array([[3], [4]])
    >>> y_train, y_test = np.array([1, 1]), np.array([1, 0])
    >>> result = evaluate_model(DummyEstimator(), X_train, X_test, y_train, y_test)
    >>> result['accuracy']
    0.5
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("Mismatched train samples between X_train and y_train.")
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError("Mismatched test samples between X_test and y_test.")
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("Train and test sets must not be empty.")

    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {
        "accuracy": acc,
        "y_pred": y_pred,
        "y_true": y_test,
    }
