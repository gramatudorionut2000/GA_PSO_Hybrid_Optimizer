
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .common import ScoringMetric, ensure_numpy

if TYPE_CHECKING:
    from collections.abc import Callable



def accuracy_score(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute accuracy score.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Accuracy score in [0, 1].
    """
    y_true = ensure_numpy(y_true).ravel()
    y_pred = ensure_numpy(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def balanced_accuracy_score(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute balanced accuracy (average of recall for each class).

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Balanced accuracy score.
    """
    y_true = ensure_numpy(y_true).ravel()
    y_pred = ensure_numpy(y_pred).ravel()

    classes = np.unique(y_true)
    recalls = []
    for cls in classes:
        mask = y_true == cls
        if np.sum(mask) > 0:
            recalls.append(np.mean(y_pred[mask] == cls))

    return float(np.mean(recalls)) if recalls else 0.0


def precision_score(
    y_true: NDArray, y_pred: NDArray, average: str = "binary"
) -> float:
    """Compute precision score.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        average: Averaging method ('binary', 'macro', 'micro', 'weighted').

    Returns:
        Precision score.
    """
    y_true = ensure_numpy(y_true).ravel()
    y_pred = ensure_numpy(y_pred).ravel()

    if average == "binary":
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    classes = np.unique(y_true)
    precisions = []
    weights = []

    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(prec)
        weights.append(np.sum(y_true == cls))

    if average == "macro":
        return float(np.mean(precisions))
    if average == "weighted":
        weights_arr = np.array(weights, dtype=np.float64)
        return float(np.average(precisions, weights=weights_arr))
    # micro: compute globally
    tp_total = sum(np.sum((y_pred == cls) & (y_true == cls)) for cls in classes)
    fp_total = sum(np.sum((y_pred == cls) & (y_true != cls)) for cls in classes)
    return float(tp_total / (tp_total + fp_total)) if (tp_total + fp_total) > 0 else 0.0


def recall_score(
    y_true: NDArray, y_pred: NDArray, average: str = "binary"
) -> float:
    """Compute recall score.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        average: Averaging method ('binary', 'macro', 'micro', 'weighted').

    Returns:
        Recall score.
    """
    y_true = ensure_numpy(y_true).ravel()
    y_pred = ensure_numpy(y_pred).ravel()

    if average == "binary":
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    classes = np.unique(y_true)
    recalls = []
    weights = []

    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(rec)
        weights.append(np.sum(y_true == cls))

    if average == "macro":
        return float(np.mean(recalls))
    if average == "weighted":
        weights_arr = np.array(weights, dtype=np.float64)
        return float(np.average(recalls, weights=weights_arr))
    # micro: same as macro for recall
    return float(np.mean(recalls))


def f1_score(y_true: NDArray, y_pred: NDArray, average: str = "binary") -> float:
    """Compute F1 score.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        average: Averaging method.

    Returns:
        F1 score.
    """
    prec = precision_score(y_true, y_pred, average=average)
    rec = recall_score(y_true, y_pred, average=average)
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def roc_auc_score(y_true: NDArray, y_prob: NDArray) -> float:
    """Compute ROC AUC score using trapezoidal rule.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for positive class.

    Returns:
        ROC AUC score.
    """
    y_true = ensure_numpy(y_true).ravel()
    y_prob = ensure_numpy(y_prob)

    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]
    y_prob = y_prob.ravel()

    desc_idx = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[desc_idx]


    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    tpr = tps / n_pos
    fpr = fps / n_neg

    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    # Trapezoidal integration
    return float(np.trapezoid(tpr, fpr))


def log_loss_score(
    y_true: NDArray, y_prob: NDArray, eps: float = 1e-15
) -> float:
    """Compute log loss (cross-entropy).

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        eps: Small value to avoid log(0).

    Returns:
        Log loss (lower is better).
    """
    y_true = ensure_numpy(y_true).ravel()
    y_prob = ensure_numpy(y_prob)

    # Clip probabilities
    y_prob = np.clip(y_prob, eps, 1 - eps)

    if y_prob.ndim == 1:
        # Binary classification
        return float(
            -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
        )

    # Multiclass
    n_samples = len(y_true)
    return float(
        -np.sum(np.log(y_prob[np.arange(n_samples), y_true.astype(int)])) / n_samples
    )


def cohen_kappa_score(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute Cohen's Kappa score.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Cohen's Kappa score.
    """
    y_true = ensure_numpy(y_true).ravel()
    y_pred = ensure_numpy(y_pred).ravel()

    n = len(y_true)
    po = np.mean(y_true == y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    pe = 0.0

    for cls in classes:
        true_freq = np.sum(y_true == cls) / n
        pred_freq = np.sum(y_pred == cls) / n
        pe += true_freq * pred_freq

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0

    return float((po - pe) / (1 - pe))


def matthews_corrcoef_score(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute Matthews Correlation Coefficient.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        MCC score in [-1, 1].
    """
    y_true = ensure_numpy(y_true).ravel()
    y_pred = ensure_numpy(y_pred).ravel()

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)



def mse_score(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute mean squared error.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        MSE (lower is better).
    """
    y_true = ensure_numpy(y_true).ravel()
    y_pred = ensure_numpy(y_pred).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def rmse_score(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute root mean squared error.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        RMSE (lower is better).
    """
    return float(np.sqrt(mse_score(y_true, y_pred)))


def mae_score(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute mean absolute error.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        MAE (lower is better).
    """
    y_true = ensure_numpy(y_true).ravel()
    y_pred = ensure_numpy(y_pred).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute R-squared score.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        R² score (higher is better, max 1.0).
    """
    y_true = ensure_numpy(y_true).ravel()
    y_pred = ensure_numpy(y_pred).ravel()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def mape_score(y_true: NDArray, y_pred: NDArray, eps: float = 1e-10) -> float:
    """Compute mean absolute percentage error.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        eps: Small value to avoid division by zero.

    Returns:
        MAPE (lower is better).
    """
    y_true = ensure_numpy(y_true).ravel()
    y_pred = ensure_numpy(y_pred).ravel()
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))


def explained_variance_score(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute Explained Variance Score.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Explained variance score.
    """
    y_true = ensure_numpy(y_true).ravel()
    y_pred = ensure_numpy(y_pred).ravel()

    residual = y_true - y_pred
    var_residual = np.var(residual)
    var_y = np.var(y_true)

    return float(1 - var_residual / var_y) if var_y > 0 else 0.0


def max_error_score(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute Maximum Error.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Maximum error (lower is better).
    """
    y_true = ensure_numpy(y_true).ravel()
    y_pred = ensure_numpy(y_pred).ravel()
    return float(np.max(np.abs(y_true - y_pred)))




SCORING_FUNCTIONS: dict[str, Callable[..., float]] = {
    # Classification
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "roc_auc": roc_auc_score,
    "log_loss": log_loss_score,
    "cohen_kappa": cohen_kappa_score,
    "matthews_corrcoef": matthews_corrcoef_score,
    # Regression
    "mse": mse_score,
    "rmse": rmse_score,
    "mae": mae_score,
    "r2": r2_score,
    "mape": mape_score,
    "explained_variance": explained_variance_score,
    "max_error": max_error_score,
}

# Metrics that need probability predictions rather than class predictions
PROBA_METRICS = frozenset({"roc_auc", "log_loss"})

# Metrics where lower is better
MINIMIZE_METRICS = frozenset({"mse", "rmse", "mae", "log_loss", "mape", "max_error"})


def get_scorer(metric: str | ScoringMetric) -> Callable[..., float]:
    """Get scoring function by name.

    Args:
        metric: Metric name or ScoringMetric enum.

    Returns:
        Scoring function.

    Raises:
        ValueError: If metric is not supported.
    """
    if isinstance(metric, ScoringMetric):
        metric_name = metric.value
    else:
        metric_name = metric.lower()

    if metric_name not in SCORING_FUNCTIONS:
        available = list(SCORING_FUNCTIONS.keys())
        msg = f"Unknown metric '{metric_name}'. Available: {available}"
        raise ValueError(msg)

    return SCORING_FUNCTIONS[metric_name]


def needs_probability_predictions(metric: str | ScoringMetric) -> bool:
    """Check if a metric needs probability predictions.

    Args:
        metric: Metric name or ScoringMetric enum.

    Returns:
        True if metric requires probability predictions.
    """
    if isinstance(metric, ScoringMetric):
        metric_name = metric.value
    else:
        metric_name = metric.lower()

    return metric_name in PROBA_METRICS


def is_minimization_metric(metric: str | ScoringMetric) -> bool:
    """Check if a metric should be minimized.

    Args:
        metric: Metric name or ScoringMetric enum.

    Returns:
        True if lower values are better.
    """
    if isinstance(metric, ScoringMetric):
        metric_name = metric.value
    else:
        metric_name = metric.lower()

    return metric_name in MINIMIZE_METRICS




def batch_score(
    y_true: NDArray,
    y_preds: list[NDArray],
    metric: str = "accuracy",
) -> NDArray:
    """Compute scores for multiple predictions.

    Args:
        y_true: True labels/values.
        y_preds: List of prediction arrays.
        metric: Scoring metric name.

    Returns:
        Array of scores for each prediction.
    """
    scorer = get_scorer(metric)
    n_preds = len(y_preds)
    scores = np.zeros(n_preds, dtype=np.float64)

    y_true_np = ensure_numpy(y_true).ravel()

    for i, y_pred in enumerate(y_preds):
        y_pred_np = ensure_numpy(y_pred)
        try:
            scores[i] = scorer(y_true_np, y_pred_np)
        except Exception:  # noqa: BLE001
            scores[i] = np.nan

    return scores