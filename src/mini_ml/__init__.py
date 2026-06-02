"""
mini-ml: A lightweight machine learning toolkit built on NumPy.

mini-ml provides essential ML utilities — preprocessing, metrics, dataset
splitting, cross-validation, and lightweight visualization — all implemented
from scratch using NumPy. It is designed to be educational, auditable, and
dependency-light.

Quick Start
-----------
>>> import numpy as np
>>> from mini_ml import StandardScaler, train_test_split
>>> X = np.array([[1, 2], [3, 4], [5, 6]])
>>> scaler = StandardScaler()
>>> X_scaled = scaler.fit_transform(X)
"""

__version__ = "0.1.0"

# Preprocessing
from .preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Metrics
from .metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Dataset splitting
from .split import train_test_split, kfold_split

# Model evaluation
from .evaluation import cross_validate, evaluate_model

# Visualization
from .visualize import plot_confusion_matrix, classification_report_text

__all__ = [
    # Version
    "__version__",
    # Preprocessing
    "StandardScaler",
    "MinMaxScaler",
    "LabelEncoder",
    # Metrics
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "confusion_matrix",
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    # Split
    "train_test_split",
    "kfold_split",
    # Evaluation
    "cross_validate",
    "evaluate_model",
    # Visualization
    "plot_confusion_matrix",
    "classification_report_text",
]
