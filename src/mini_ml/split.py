"""Dataset splitting utilities implemented from scratch using NumPy."""

import numpy as np


def train_test_split(*arrays, test_size=0.25, train_size=None, random_state=None,
                     shuffle=True):
    """Split arrays into random train and test subsets.

    Accepts multiple arrays (e.g., X, y) and returns them split consistently
    using the same random permutation for all arrays.

    Parameters
    ----------
    *arrays : array-like
        Arrays to split. All must have the same first dimension (n_samples).
    test_size : float or int, default=0.25
        If float, proportion of the dataset to include in the test split
        (between 0.0 and 1.0). If int, absolute number of test samples.
    train_size : float or int, optional
        If float, proportion of the dataset to include in the train split.
        If int, absolute number of train samples. If None, complement of test_size.
    random_state : int, optional
        Seed for reproducible shuffling.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.

    Returns
    -------
    list of np.ndarray
        Train-test split of inputs, in the same order as the input arrays.
        If one array is passed, returns [train, test]. If two, returns
        [X_train, X_test, y_train, y_test], etc.

    Raises
    ------
    ValueError
        If inputs are empty, have mismatched lengths, or invalid test_size.

    Examples
    --------
    >>> import numpy as np
    >>> from mini_ml.split import train_test_split
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([0, 1, 0, 1])
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.5, random_state=42
    ... )
    >>> len(X_train)
    2
    >>> len(X_test)
    2
    """
    if len(arrays) == 0:
        raise ValueError("At least one array is required.")

    arrays = [np.asarray(a) for a in arrays]
    n_samples = arrays[0].shape[0]

    if n_samples == 0:
        raise ValueError("Input arrays must not be empty.")

    for a in arrays[1:]:
        if a.shape[0] != n_samples:
            raise ValueError(
                f"Mismatched lengths: first array has {n_samples} samples, "
                f"another has {a.shape[0]} samples."
            )

    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError(
                f"test_size as float must be in (0.0, 1.0), got {test_size}."
            )
        n_test = int(np.ceil(n_samples * test_size)) if test_size > 0 else 0
    elif isinstance(test_size, int):
        if not 0 < test_size < n_samples:
            raise ValueError(
                f"test_size as int must be in (0, {n_samples}), got {test_size}."
            )
        n_test = test_size
    else:
        raise ValueError(
            f"test_size must be float or int, got {type(test_size).__name__}."
        )

    if train_size is not None:
        if isinstance(train_size, float):
            n_train = int(train_size * n_samples)
        else:
            n_train = train_size
    else:
        n_train = n_samples - n_test

    if n_train + n_test > n_samples:
        raise ValueError(
            f"train_size ({n_train}) + test_size ({n_test}) "
            f"> n_samples ({n_samples})."
        )

    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)
    if shuffle:
        rng.shuffle(indices)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:n_train + n_test]

    result = []
    for a in arrays:
        result.append(a[train_indices])
        result.append(a[test_indices])

    return result


def kfold_split(n_samples, n_splits=5, shuffle=True, random_state=None):
    """Generate train/test indices for K-fold cross-validation.

    Splits the dataset into ``n_splits`` consecutive folds (each fold used
    once as the test set while the remaining folds form the training set).

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    n_splits : int, default=5
        Number of folds. Must be at least 2 and at most n_samples.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting into batches.
    random_state : int, optional
        Seed for reproducible shuffling.

    Returns
    -------
    list of tuple
        A list of (train_indices, test_indices) tuples, one per fold.

    Raises
    ------
    ValueError
        If n_splits is invalid or n_samples is not positive.

    Examples
    --------
    >>> from mini_ml.split import kfold_split
    >>> folds = kfold_split(10, n_splits=3, shuffle=False)
    >>> len(folds)
    3
    >>> for train_idx, test_idx in folds:
    ...     assert len(train_idx) > 0 and len(test_idx) > 0
    >>> # All samples covered
    >>> all_indices = []
    >>> for _, test_idx in folds:
    ...     all_indices.extend(test_idx.tolist())
    >>> sorted(all_indices)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples}.")
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}.")
    if n_splits > n_samples:
        raise ValueError(
            f"n_splits ({n_splits}) cannot be greater than n_samples ({n_samples})."
        )

    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)
    if shuffle:
        rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int64)
    fold_sizes[:n_samples % n_splits] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, end = current, current + fold_size
        test_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_indices, test_indices))
        current = end

    return folds
