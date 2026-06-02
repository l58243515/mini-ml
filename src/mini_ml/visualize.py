"""Lightweight visualization and reporting utilities."""

import numpy as np

from .metrics import precision_score, recall_score, f1_score


def plot_confusion_matrix(cm, labels=None, title="Confusion Matrix", as_text=False):
    """Generate a text-based or graphical confusion matrix visualization.

    If matplotlib is available and ``as_text=False``, uses ``plt.imshow``
    to create a colormap plot. Otherwise returns a formatted ASCII table.

    Parameters
    ----------
    cm : np.ndarray of shape (n_classes, n_classes)
        Confusion matrix.
    labels : list of str, optional
        Class labels. If None, uses numeric indices.
    title : str, default="Confusion Matrix"
        Title for the plot or text block.
    as_text : bool, default=False
        If True, forces text-based output even when matplotlib is available.

    Returns
    -------
    str
        A string representation of the confusion matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from mini_ml.visualize import plot_confusion_matrix
    >>> cm = np.array([[10, 2], [1, 7]])
    >>> print(plot_confusion_matrix(cm, labels=['Neg', 'Pos'], as_text=True))
    Confusion Matrix
    ╔══════════╤══════╤══════╤═══════╗
    ║          │ Neg  │ Pos  │ Total ║
    ╠══════════╪══════╪══════╪═══════╣
    ║ Neg      │   10 │    2 │    12 ║
    ╟──────────┼──────┼──────┼───────╢
    ║ Pos      │    1 │    7 │     8 ║
    ╠══════════╪══════╪══════╪═══════╣
    ║ Total    │   11 │    9 │    20 ║
    ╚══════════╧══════╧══════╧═══════╝
    <BLANKLINE>
    """
    cm = np.asarray(cm, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(
            f"Confusion matrix must be square 2D array, got shape {cm.shape}."
        )

    n_classes = cm.shape[0]
    if labels is None:
        labels = [str(i) for i in range(n_classes)]
    if len(labels) != n_classes:
        raise ValueError(
            f"Number of labels ({len(labels)}) does not match "
            f"matrix dimensions ({n_classes})."
        )

    # Try matplotlib first
    if not as_text:
        try:
            return _plot_matplotlib(cm, labels, title)
        except ImportError:
            pass

    # Fallback to ASCII
    return _plot_ascii(cm, labels, title)


def _plot_matplotlib(cm, labels, title):
    """Create a matplotlib confusion matrix plot. Returns descriptive text."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate each cell
    for i in range(len(labels)):
        for j in range(len(labels)):
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.close(fig)
    return f"[Matplotlib plot generated for {title}]\n{_plot_ascii(cm, labels, title)}"


def _plot_ascii(cm, labels, title):
    """Build a formatted ASCII confusion matrix table with totals."""
    # Calculate column widths
    label_widths = [len(str(l)) for l in labels]
    max_label_w = max(max(label_widths), len("Total"))

    cell_widths = []
    for j in range(len(labels)):
        max_cell = max(len(str(cm[i, j])) for i in range(len(labels)))
        cell_widths.append(max(max_cell, label_widths[j], 4))

    col_sep = "    "
    total_cell_w = 6
    n_cols = len(labels)

    lines = [title]
    lines.append(_make_border(max_label_w, cell_widths, total_cell_w, "top"))

    # Header row
    lines.append(f"{chr(0x2551)} {'':{max_label_w}} {chr(0x2502)} "
                 f"{'Predicted'.center(sum(cell_widths) + len(col_sep) * (n_cols - 1))} "
                 f"{chr(0x2502)} {'Total':{total_cell_w}} {chr(0x2551)}")

    # Sub-header with column labels
    sub_cols = " " + col_sep.join(f"{l:{cell_widths[j]}}"
                                   for j, l in enumerate(labels))
    lines.append(f"{chr(0x2551)} {'':{max_label_w}} {chr(0x2502)}{sub_cols}"
                 f" {chr(0x2502)} {'':{total_cell_w}} {chr(0x2551)}")

    # Separator
    lines.append(_make_border(max_label_w, cell_widths, total_cell_w, "mid"))

    # Data rows
    row_totals = np.sum(cm, axis=1)
    col_totals = np.sum(cm, axis=0)
    for i in range(n_cols):
        row_data = " " + col_sep.join(
            f"{cm[i, j]:{cell_widths[j]}}" for j in range(n_cols))
        lines.append(
            f"{chr(0x2551)} {labels[i]:{max_label_w}} {chr(0x2502)}{row_data}"
            f" {chr(0x2502)} {row_totals[i]:{total_cell_w}} {chr(0x2551)}"
        )
        if i < n_cols - 1:
            lines.append(_make_border(max_label_w, cell_widths,
                                      total_cell_w, "row"))

    # Total row
    lines.append(_make_border(max_label_w, cell_widths, total_cell_w, "mid"))
    total_row = " " + col_sep.join(
        f"{col_totals[j]:{cell_widths[j]}}" for j in range(n_cols))
    lines.append(
        f"{chr(0x2551)} {'Total':{max_label_w}} {chr(0x2502)}{total_row}"
        f" {chr(0x2502)} {np.sum(cm):{total_cell_w}} {chr(0x2551)}"
    )
    lines.append(_make_border(max_label_w, cell_widths, total_cell_w, "bottom"))
    lines.append("")
    return "\n".join(lines)


def _make_border(label_w, cell_widths, total_w, kind):
    """Create a box-drawing border row."""
    chars = {
        "top":    (chr(0x2554), chr(0x2564), chr(0x2566), chr(0x2557)),
        "mid":    (chr(0x2560), chr(0x255E), chr(0x256C), chr(0x2563)),
        "row":    (chr(0x255F), chr(0x2502), chr(0x253C), chr(0x2562)),
        "bottom": (chr(0x255A), chr(0x2567), chr(0x2569), chr(0x255D)),
    }
    left, inner, mid_sep, right = chars[kind]
    parts = [left, chr(0x2550) * label_w, inner]
    for w in cell_widths:
        parts.append(chr(0x2550) * w)
        parts.append(inner)
    parts[-1] = mid_sep
    parts.append(chr(0x2550) * total_w)
    parts.append(right)
    return "".join(parts)


def classification_report_text(y_true, y_pred, labels=None):
    """Generate a text-based classification report.

    Produces a formatted report showing precision, recall, F1-score,
    and support for each class, plus aggregate (macro avg, weighted avg)
    metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    labels : list, optional
        Optional display names for the classes.

    Returns
    -------
    str
        Formatted classification report string.

    Examples
    --------
    >>> import numpy as np
    >>> from mini_ml.visualize import classification_report_text
    >>> y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    >>> y_pred = np.array([0, 0, 1, 1, 1, 2, 2, 2])
    >>> print(classification_report_text(y_true, y_pred))
                  precision    recall  f1-score   support
    <BLANKLINE>
             0       1.00      0.67      0.80         3
             1       0.50      1.00      0.67         2
             2       1.00      1.00      1.00         3
    <BLANKLINE>
        macro       0.83      0.89      0.82         8
     weighted       0.88      0.88      0.84         8
    <BLANKLINE>
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if labels is not None:
        display_labels = labels
    else:
        display_labels = [str(c) for c in classes]

    header = "              precision    recall  f1-score   support"
    lines = [header, ""]

    supports = []
    precisions = []
    recalls = []
    f1s = []

    for cls in classes:
        support = int(np.sum(y_true == cls))
        supports.append(support)
        p = precision_score(y_true, y_pred, average="binary", pos_label=cls)
        r = recall_score(y_true, y_pred, average="binary", pos_label=cls)
        f = f1_score(y_true, y_pred, average="binary", pos_label=cls)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

        label_str = display_labels[len(precisions) - 1]
        lines.append(
            f"   {label_str:>8}   {p:6.2f}   {r:6.2f}   {f:6.2f}   {support:7d}"
        )

    lines.append("")

    # Macro avg
    macro_p = np.mean(precisions)
    macro_r = np.mean(recalls)
    macro_f = np.mean(f1s)
    total = sum(supports)
    lines.append(
        f"    {'macro':>8}   {macro_p:6.2f}   {macro_r:6.2f}"
        f"   {macro_f:6.2f}   {total:7d}"
    )

    # Weighted avg
    weighted_p = np.average(precisions, weights=supports)
    weighted_r = np.average(recalls, weights=supports)
    weighted_f = np.average(f1s, weights=supports)
    lines.append(
        f" {'weighted':>8}   {weighted_p:6.2f}   {weighted_r:6.2f}"
        f"   {weighted_f:6.2f}   {total:7d}"
    )

    lines.append("")
    return "\n".join(lines)
