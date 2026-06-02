"""Data preprocessing utilities implemented from scratch using NumPy."""

import numpy as np


def _validate_input(X, require_2d=True):
    """Validate and convert input to a numpy array.

    Parameters
    ----------
    X : array-like
        Input data.
    require_2d : bool, default=True
        If True, ensure the output is a 2D array.

    Returns
    -------
    np.ndarray
        Validated array.

    Raises
    ------
    ValueError
        If the input is empty or invalid.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.size == 0:
        raise ValueError("Input array is empty. At least one sample is required.")
    if require_2d and X.ndim == 1:
        X = X.reshape(-1, 1)
    if require_2d and X.ndim != 2:
        raise ValueError(
            f"Expected 1D or 2D array, got {X.ndim}D array with shape {X.shape}."
        )
    return X


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample ``x`` is calculated as ``z = (x - u) / s``,
    where ``u`` is the mean of the training samples and ``s`` is the standard
    deviation.

    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling.
    with_std : bool, default=True
        If True, scale the data to unit variance.

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        Per-feature mean of the training data.
    std_ : np.ndarray of shape (n_features,)
        Per-feature standard deviation of the training data.

    Examples
    --------
    >>> import numpy as np
    >>> from mini_ml.preprocessing import StandardScaler
    >>> X = np.array([[1., 2.], [3., 4.], [5., 6.]])
    >>> scaler = StandardScaler()
    >>> X_scaled = scaler.fit_transform(X)
    >>> X_scaled
    array([[-1.224..., -1.224...],
           [ 0.     ,  0.     ],
           [ 1.224...,  1.224...]])
    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """Compute the mean and std for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : StandardScaler
            Fitted scaler.
        """
        X = _validate_input(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0, ddof=0)

        if self.with_std:
            zero_std = self.std_ <= 1e-15
            if np.any(zero_std):
                zero_cols = np.where(zero_std)[0]
                raise ValueError(
                    f"Features {list(zero_cols)} have zero or near-zero standard "
                    f"deviation. StandardScaler cannot scale constant features."
                )

        return self

    def transform(self, X):
        """Perform standardization by centering and scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        X = _validate_input(X)
        if self.mean_ is None or self.std_ is None:
            raise ValueError(
                "This StandardScaler instance is not fitted yet. "
                "Call 'fit' before 'transform'."
            )
        X_copy = np.copy(X)
        if self.with_mean:
            X_copy = X_copy - self.mean_
        if self.with_std:
            X_copy = X_copy / self.std_
        return X_copy

    def fit_transform(self, X):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Scale back the data to the original representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Scaled data.

        Returns
        -------
        X_original : np.ndarray of shape (n_samples, n_features)
            Data in original scale.
        """
        X = _validate_input(X)
        if self.mean_ is None or self.std_ is None:
            raise ValueError(
                "This StandardScaler instance is not fitted yet. "
                "Call 'fit' before 'inverse_transform'."
            )
        X_copy = np.copy(X)
        if self.with_std:
            X_copy = X_copy * self.std_
        if self.with_mean:
            X_copy = X_copy + self.mean_
        return X_copy


class MinMaxScaler:
    """Transform features by scaling each feature to a given range.

    The transformation is given by::

        X_scaled = X * (max - min) / (X_max - X_min) + min

    where ``min, max = feature_range``.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    Attributes
    ----------
    min_ : np.ndarray of shape (n_features,)
        Per-feature minimum seen in the data.
    max_ : np.ndarray of shape (n_features,)
        Per-feature maximum seen in the data.

    Examples
    --------
    >>> import numpy as np
    >>> from mini_ml.preprocessing import MinMaxScaler
    >>> X = np.array([[1., 2.], [3., 4.], [5., 6.]])
    >>> scaler = MinMaxScaler()
    >>> X_scaled = scaler.fit_transform(X)
    >>> X_scaled
    array([[0. , 0. ],
           [0.5, 0.5],
           [1. , 1. ]])
    """

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        """Compute the minimum and maximum for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : MinMaxScaler
            Fitted scaler.
        """
        X = _validate_input(X)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)

        range_mask = (self.max_ - self.min_) <= 1e-15
        if np.any(range_mask):
            zero_cols = np.where(range_mask)[0]
            raise ValueError(
                f"Features {list(zero_cols)} have zero range "
                f"(min == max). MinMaxScaler cannot scale constant features."
            )

        return self

    def transform(self, X):
        """Scale features to the specified range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        X = _validate_input(X)
        if self.min_ is None or self.max_ is None:
            raise ValueError(
                "This MinMaxScaler instance is not fitted yet. "
                "Call 'fit' before 'transform'."
            )
        X_copy = np.copy(X)
        data_min, data_max = self.feature_range
        scale = (data_max - data_min) / (self.max_ - self.min_)
        X_copy = X_copy * scale + (data_min - self.min_ * scale)
        return X_copy

    def fit_transform(self, X):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Scale back the data to the original representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Scaled data.

        Returns
        -------
        X_original : np.ndarray of shape (n_samples, n_features)
            Data in original scale.
        """
        X = _validate_input(X)
        if self.min_ is None or self.max_ is None:
            raise ValueError(
                "This MinMaxScaler instance is not fitted yet. "
                "Call 'fit' before 'inverse_transform'."
            )
        X_copy = np.copy(X)
        data_min, data_max = self.feature_range
        scale = (data_max - data_min) / (self.max_ - self.min_)
        return (X_copy - (data_min - self.min_ * scale)) / scale


class LabelEncoder:
    """Encode target labels with values between 0 and n_classes-1.

    This transformer converts class labels to integers in the range
    [0, n_classes-1] and provides the inverse mapping.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        The unique class labels in sorted order.
    class_to_index_ : dict
        Mapping from original class label to integer encoding.

    Examples
    --------
    >>> import numpy as np
    >>> from mini_ml.preprocessing import LabelEncoder
    >>> le = LabelEncoder()
    >>> y = np.array(['cat', 'dog', 'cat', 'bird'])
    >>> y_encoded = le.fit_transform(y)
    >>> y_encoded
    array([0, 2, 0, 1])
    >>> le.inverse_transform(y_encoded)
    array(['cat', 'dog', 'cat', 'bird'], dtype='<U4')
    """

    def __init__(self):
        self.classes_ = None
        self.class_to_index_ = None

    def fit(self, y):
        """Fit label encoder by discovering unique classes.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : LabelEncoder
            Fitted encoder.
        """
        y = np.asarray(y)
        if y.size == 0:
            raise ValueError("Input array is empty. At least one label is required.")
        if y.ndim != 1:
            y = y.ravel()
        self.classes_ = np.unique(y)
        self.class_to_index_ = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, y):
        """Transform labels to normalized integer encoding.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Labels to encode.

        Returns
        -------
        y_encoded : np.ndarray of shape (n_samples,)
            Integer-encoded labels.
        """
        y = np.asarray(y)
        if y.size == 0:
            raise ValueError("Input array is empty. At least one label is required.")
        if y.ndim != 1:
            y = y.ravel()
        if self.class_to_index_ is None:
            raise ValueError(
                "This LabelEncoder instance is not fitted yet. "
                "Call 'fit' before 'transform'."
            )
        unseen = set(y) - set(self.classes_)
        if unseen:
            raise ValueError(
                f"Labels {list(unseen)} were not seen during fit. "
                f"Fitted classes: {list(self.classes_)}"
            )
        return np.array([self.class_to_index_[label] for label in y], dtype=np.int64)

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        y_encoded : np.ndarray of shape (n_samples,)
            Integer-encoded labels.
        """
        return self.fit(y).transform(y)

    def inverse_transform(self, y_encoded):
        """Transform integer labels back to original encoding.

        Parameters
        ----------
        y_encoded : array-like of shape (n_samples,)
            Integer-encoded labels.

        Returns
        -------
        y_original : np.ndarray of shape (n_samples,)
            Original class labels.
        """
        y_encoded = np.asarray(y_encoded)
        if y_encoded.size == 0:
            raise ValueError("Input array is empty. At least one label is required.")
        if self.classes_ is None:
            raise ValueError(
                "This LabelEncoder instance is not fitted yet. "
                "Call 'fit' before 'inverse_transform'."
            )
        return np.array([self.classes_[int(idx)] for idx in y_encoded.ravel()])
