"""Evaluation metrics implemented from scratch using NumPy."""

import numpy as np


def _validate_1d(y_true, y_pred, name="arrays"):
    """Validate that inputs are 1D numpy arrays of the same length.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.
    name : str
        Descriptive name for error messages.

    Returns
    -------
    y_true, y_pred : tuple of np.ndarray
        Validated 1D arrays.

    Raises
    ------
    ValueError
        If inputs are empty or have mismatched shapes.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError(f"Input {name} must not be empty.")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Mismatched shapes for {name}: y_true has {y_true.shape[0]} samples, "
            f"y_pred has {y_pred.shape[0]} samples."
        )
    return y_true, y_pred


def accuracy_score(y_true, y_pred):
    """Compute the accuracy classification score.

    Accuracy is the fraction of correctly classified samples:
    ``(y_true == y_pred).mean()``

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    float
        Accuracy score in [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> from mini_ml.metrics import accuracy_score
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> accuracy_score(y_true, y_pred)
    0.75
    """
    y_true, y_pred = _validate_1d(y_true, y_pred, "accuracy_score")
    return float(np.mean(y_true == y_pred))


def _binary_precision_recall_f1(y_true, y_pred, pos_label=1):
    """Compute tp, fp, fn for binary classification."""
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    return tp, fp, fn


def precision_score(y_true, y_pred, average="binary", pos_label=1):
    """Compute the precision score.

    Precision is the ratio ``tp / (tp + fp)``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    average : {'binary', 'macro', 'micro', 'weighted'}, default='binary'
        Averaging strategy for multiclass data.
        - 'binary': Report precision for the class specified by ``pos_label``.
        - 'macro': Calculate metrics for each class and average.
        - 'micro': Calculate metrics globally by counting total tp, fp, fn.
        - 'weighted': Macro average weighted by class support.
    pos_label : int, default=1
        The positive class label for binary averaging.

    Returns
    -------
    float
        Precision score.

    Examples
    --------
    >>> import numpy as np
    >>> from mini_ml.metrics import precision_score
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0, 1, 0, 0, 1])
    >>> precision_score(y_true, y_pred)
    0.666...
    """
    y_true, y_pred = _validate_1d(y_true, y_pred, "precision_score")
    classes = np.unique(np.concatenate([y_true, y_pred]))

    if average == "binary":
        tp, fp, _ = _binary_precision_recall_f1(y_true, y_pred, pos_label)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    per_class = []
    supports = []
    for cls in classes:
        tp, fp, _ = _binary_precision_recall_f1(y_true, y_pred, pos_label=cls)
        per_class.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        supports.append(np.sum(y_true == cls))

    if average == "macro":
        return float(np.mean(per_class))
    elif average == "micro":
        # For multi-class, micro precision == micro recall == accuracy
        total_tp = sum(
            np.sum((y_true == c) & (y_pred == c)) for c in classes
        )
        total_fp = sum(
            np.sum((y_true != c) & (y_pred == c)) for c in classes
        )
        return total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    elif average == "weighted":
        return float(np.average(per_class, weights=supports))
    else:
        raise ValueError(
            f"Unknown average type '{average}'. "
            f"Expected one of 'binary', 'macro', 'micro', 'weighted'."
        )


def recall_score(y_true, y_pred, average="binary", pos_label=1):
    """Compute the recall score.

    Recall is the ratio ``tp / (tp + fn)``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    average : {'binary', 'macro', 'micro', 'weighted'}, default='binary'
        Averaging strategy for multiclass data.
    pos_label : int, default=1
        The positive class label for binary averaging.

    Returns
    -------
    float
        Recall score.

    Examples
    --------
    >>> import numpy as np
    >>> from mini_ml.metrics import recall_score
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0, 1, 0, 0, 1])
    >>> recall_score(y_true, y_pred)
    0.666...
    """
    y_true, y_pred = _validate_1d(y_true, y_pred, "recall_score")
    classes = np.unique(np.concatenate([y_true, y_pred]))

    if average == "binary":
        tp, _, fn = _binary_precision_recall_f1(y_true, y_pred, pos_label)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    per_class = []
    supports = []
    for cls in classes:
        tp, _, fn = _binary_precision_recall_f1(y_true, y_pred, pos_label=cls)
        per_class.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        supports.append(np.sum(y_true == cls))

    if average == "macro":
        return float(np.mean(per_class))
    elif average == "micro":
        total_tp = sum(
            np.sum((y_true == c) & (y_pred == c)) for c in classes
        )
        total_fn = sum(
            np.sum((y_true == c) & (y_pred != c)) for c in classes
        )
        return total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    elif average == "weighted":
        return float(np.average(per_class, weights=supports))
    else:
        raise ValueError(
            f"Unknown average type '{average}'. "
            f"Expected one of 'binary', 'macro', 'micro', 'weighted'."
        )


def f1_score(y_true, y_pred, average="binary", pos_label=1):
    """Compute the F1 score.

    F1 = 2 * precision * recall / (precision + recall)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    average : {'binary', 'macro', 'micro', 'weighted'}, default='binary'
        Averaging strategy for multiclass data.
    pos_label : int, default=1
        The positive class label for binary averaging.

    Returns
    -------
    float
        F1 score.

    Examples
    --------
    >>> import numpy as np
    >>> from mini_ml.metrics import f1_score
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0, 1, 0, 0, 1])
    >>> f1_score(y_true, y_pred)
    0.666...
    """
    y_true, y_pred = _validate_1d(y_true, y_pred, "f1_score")
    p = precision_score(y_true, y_pred, average=average, pos_label=pos_label)
    r = recall_score(y_true, y_pred, average=average, pos_label=pos_label)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def confusion_matrix(y_true, y_pred, labels=None):
    """Compute confusion matrix to evaluate classification accuracy.

    By definition, entry (i, j) is the number of samples with true label i
    and predicted label j.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    labels : array-like of shape (n_classes,), optional
        List of labels to index the matrix. If None, uses all unique values.

    Returns
    -------
    cm : np.ndarray of shape (n_classes, n_classes)
        Confusion matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from mini_ml.metrics import confusion_matrix
    >>> y_true = np.array([0, 1, 0, 1])
    >>> y_pred = np.array([0, 1, 1, 1])
    >>> confusion_matrix(y_true, y_pred)
    array([[1, 1],
           [0, 2]])
    """
    y_true, y_pred = _validate_1d(y_true, y_pred, "confusion_matrix")
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)

    n_labels = len(labels)
    label_to_index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((n_labels, n_labels), dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        if t in label_to_index and p in label_to_index:
            cm[label_to_index[t], label_to_index[p]] += 1

    return cm


def mean_squared_error(y_true, y_pred):
    """Mean squared error regression loss.

    MSE = (1/n) * sum((y_true_i - y_pred_i)^2)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    float
        Mean squared error.

    Examples
    --------
    >>> import numpy as np
    >>> from mini_ml.metrics import mean_squared_error
    >>> y_true = np.array([3.0, 5.0, 2.5, 7.0])
    >>> y_pred = np.array([2.5, 5.0, 2.0, 8.0])
    >>> mean_squared_error(y_true, y_pred)
    0.375
    """
    y_true, y_pred = _validate_1d(y_true, y_pred, "mean_squared_error")
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true, y_pred):
    """Mean absolute error regression loss.

    MAE = (1/n) * sum(|y_true_i - y_pred_i|)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    float
        Mean absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> from mini_ml.metrics import mean_absolute_error
    >>> y_true = np.array([3.0, 5.0, 2.5, 7.0])
    >>> y_pred = np.array([2.5, 5.0, 2.0, 8.0])
    >>> mean_absolute_error(y_true, y_pred)
    0.5
    """
    y_true, y_pred = _validate_1d(y_true, y_pred, "mean_absolute_error")
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred):
    """R^2 (coefficient of determination) regression score.

    R^2 = 1 - SS_res / SS_tot

    where SS_res = sum((y_true_i - y_pred_i)^2) and
    SS_tot = sum((y_true_i - mean(y_true))^2).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    float
        R^2 score. Best possible score is 1.0.

    Examples
    --------
    >>> import numpy as np
    >>> from mini_ml.metrics import r2_score
    >>> y_true = np.array([3.0, 5.0, 2.5, 7.0])
    >>> y_pred = np.array([3.0, 5.0, 2.5, 7.0])
    >>> r2_score(y_true, y_pred)
    1.0
    """
    y_true, y_pred = _validate_1d(y_true, y_pred, "r2_score")
    if len(y_true) < 2:
        raise ValueError("r2_score requires at least 2 samples.")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0 if ss_res == 0 else float("-inf")
    return float(1 - ss_res / ss_tot)
