from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from .utils.common import ScoringMetric

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

ArrayType = TypeVar("ArrayType", bound=np.ndarray)





class PipelineStage(Enum):
    """Pipeline stages for evaluation."""

    FEATURE_SELECTION = auto()
    PREPROCESSING = auto()
    MODEL_FITTING = auto()
    PREDICTION = auto()
    SCORING = auto()




def _get_array_module(arr: Any) -> Any:
    """Get the array module (numpy or cupy) for an array."""
    if hasattr(arr, "__cuda_array_interface__"):
        try:
            import cupy as cp

            return cp
        except ImportError:
            pass
    return np


def _ensure_numpy(arr: Any) -> NDArray:
    """Convert array to numpy if needed."""
    if hasattr(arr, "get"):  # CuPy array
        return arr.get()
    return np.asarray(arr)


def _ensure_contiguous(arr: Any) -> Any:
    """Ensure array is contiguous in memory."""
    xp = _get_array_module(arr)
    if not arr.flags.c_contiguous:
        return xp.ascontiguousarray(arr)
    return arr


def _is_gpu_array(arr: Any) -> bool:
    """Check if array is a GPU array."""
    return hasattr(arr, "__cuda_array_interface__")




class GPUScorer:
    """GPU-accelerated scoring functions.

    Provides unified interface for scoring that automatically uses GPU
    when available and beneficial, falling back to CPU otherwise.
    """

    def __init__(self, use_gpu: bool = True, min_samples_for_gpu: int = 1000) -> None:
        """Initialize GPU scorer.

        Args:
            use_gpu: Whether to attempt GPU acceleration.
            min_samples_for_gpu: Minimum samples to use GPU.
        """
        self.use_gpu = use_gpu
        self.min_samples_for_gpu = min_samples_for_gpu
        self._gpu_available = self._check_gpu()

    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        if not self.use_gpu:
            return False
        try:
            import cupy as cp

            _ = cp.array([1, 2, 3])
            return True
        except (ImportError, Exception):
            return False

    def _should_use_gpu(self, n_samples: int) -> bool:
        """Determine if GPU should be used for this operation."""
        return (
            self._gpu_available
            and self.use_gpu
            and n_samples >= self.min_samples_for_gpu
        )


    def accuracy(self, y_true: Any, y_pred: Any) -> float:
        """Compute accuracy score.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Accuracy score in [0, 1].
        """
        xp = _get_array_module(y_true)
        y_true = xp.asarray(y_true).ravel()
        y_pred = xp.asarray(y_pred).ravel()

        if self._should_use_gpu(len(y_true)):
            result = xp.mean(y_true == y_pred)
            return float(result.get() if hasattr(result, "get") else result)

        y_true_np = _ensure_numpy(y_true)
        y_pred_np = _ensure_numpy(y_pred)
        return float(np.mean(y_true_np == y_pred_np))

    def balanced_accuracy(self, y_true: Any, y_pred: Any) -> float:
        """Compute balanced accuracy (average recall per class).

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Balanced accuracy score.
        """
        y_true_np = _ensure_numpy(y_true).ravel()
        y_pred_np = _ensure_numpy(y_pred).ravel()

        classes = np.unique(y_true_np)
        recalls = np.zeros(len(classes), dtype=np.float64)

        for i, cls in enumerate(classes):
            mask = y_true_np == cls
            if np.sum(mask) > 0:
                recalls[i] = np.mean(y_pred_np[mask] == cls)

        return float(np.mean(recalls))

    def precision(
        self,
        y_true: Any,
        y_pred: Any,
        average: str = "binary",
    ) -> float:
        """Compute precision score.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            average: Averaging method ('binary', 'macro', 'micro', 'weighted').

        Returns:
            Precision score.
        """
        y_true_np = _ensure_numpy(y_true).ravel()
        y_pred_np = _ensure_numpy(y_pred).ravel()

        if average == "binary":
            tp = np.sum((y_pred_np == 1) & (y_true_np == 1))
            fp = np.sum((y_pred_np == 1) & (y_true_np == 0))
            return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

        classes = np.unique(y_true_np)
        precisions = np.zeros(len(classes), dtype=np.float64)
        supports = np.zeros(len(classes), dtype=np.float64)

        for i, cls in enumerate(classes):
            tp = np.sum((y_pred_np == cls) & (y_true_np == cls))
            fp = np.sum((y_pred_np == cls) & (y_true_np != cls))
            precisions[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            supports[i] = np.sum(y_true_np == cls)

        if average == "macro":
            return float(np.mean(precisions))
        if average == "weighted":
            return float(np.average(precisions, weights=supports))
        # micro
        tp_total = sum(
            np.sum((y_pred_np == cls) & (y_true_np == cls)) for cls in classes
        )
        fp_total = sum(
            np.sum((y_pred_np == cls) & (y_true_np != cls)) for cls in classes
        )
        return float(tp_total / (tp_total + fp_total)) if (tp_total + fp_total) > 0 else 0.0

    def recall(
        self,
        y_true: Any,
        y_pred: Any,
        average: str = "binary",
    ) -> float:
        """Compute recall score.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            average: Averaging method.

        Returns:
            Recall score.
        """
        y_true_np = _ensure_numpy(y_true).ravel()
        y_pred_np = _ensure_numpy(y_pred).ravel()

        if average == "binary":
            tp = np.sum((y_pred_np == 1) & (y_true_np == 1))
            fn = np.sum((y_pred_np == 0) & (y_true_np == 1))
            return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        classes = np.unique(y_true_np)
        recalls = np.zeros(len(classes), dtype=np.float64)
        supports = np.zeros(len(classes), dtype=np.float64)

        for i, cls in enumerate(classes):
            tp = np.sum((y_pred_np == cls) & (y_true_np == cls))
            fn = np.sum((y_pred_np != cls) & (y_true_np == cls))
            recalls[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            supports[i] = np.sum(y_true_np == cls)

        if average == "macro":
            return float(np.mean(recalls))
        if average == "weighted":
            return float(np.average(recalls, weights=supports))
        return float(np.mean(recalls))

    def f1(
        self,
        y_true: Any,
        y_pred: Any,
        average: str = "binary",
    ) -> float:
        """Compute F1 score.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            average: Averaging method.

        Returns:
            F1 score.
        """
        prec = self.precision(y_true, y_pred, average=average)
        rec = self.recall(y_true, y_pred, average=average)
        return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    def roc_auc(self, y_true: Any, y_prob: Any) -> float:
        """Compute ROC AUC score using trapezoidal rule.

        GPU-accelerated for large arrays.

        Args:
            y_true: True binary labels.
            y_prob: Predicted probabilities for positive class.

        Returns:
            ROC AUC score.
        """
        y_true_np = _ensure_numpy(y_true).ravel()
        y_prob_np = _ensure_numpy(y_prob)

        # Handle probability arrays
        if y_prob_np.ndim == 2:
            y_prob_np = y_prob_np[:, 1]
        y_prob_np = y_prob_np.ravel()

        n_pos = np.sum(y_true_np == 1)
        n_neg = np.sum(y_true_np == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.5  # Undefined, return chance level

        # Sort by probability (descending)
        desc_idx = np.argsort(y_prob_np)[::-1]
        y_true_sorted = y_true_np[desc_idx]

        # Compute TPR and FPR at each threshold
        tps = np.cumsum(y_true_sorted == 1)
        fps = np.cumsum(y_true_sorted == 0)

        tpr = tps / n_pos
        fpr = fps / n_neg

        # Add (0, 0) point
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])

        # Trapezoidal integration
        return float(np.trapezoid(tpr, fpr))

    def log_loss(
        self,
        y_true: Any,
        y_prob: Any,
        eps: float = 1e-15,
    ) -> float:
        """Compute log loss (cross-entropy).

        GPU-accelerated for large arrays.

        Args:
            y_true: True labels.
            y_prob: Predicted probabilities.
            eps: Small value to avoid log(0).

        Returns:
            Log loss (lower is better).
        """
        xp = _get_array_module(y_prob)
        y_true_arr = xp.asarray(y_true).ravel()
        y_prob_arr = xp.asarray(y_prob)

        # Clip probabilities
        y_prob_arr = xp.clip(y_prob_arr, eps, 1 - eps)

        if y_prob_arr.ndim == 1:
            # Binary classification
            loss = -xp.mean(
                y_true_arr * xp.log(y_prob_arr)
                + (1 - y_true_arr) * xp.log(1 - y_prob_arr)
            )
        else:
            # Multiclass
            n_samples = len(y_true_arr)
            indices = xp.arange(n_samples)
            loss = -xp.sum(xp.log(y_prob_arr[indices, y_true_arr.astype(int)])) / n_samples

        return float(loss.get() if hasattr(loss, "get") else loss)

    def cohen_kappa(self, y_true: Any, y_pred: Any) -> float:
        """Compute Cohen's Kappa score.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Cohen's Kappa score.
        """
        y_true_np = _ensure_numpy(y_true).ravel()
        y_pred_np = _ensure_numpy(y_pred).ravel()

        # Compute confusion matrix elements
        n = len(y_true_np)
        po = np.mean(y_true_np == y_pred_np)  # Observed agreement

        classes = np.unique(np.concatenate([y_true_np, y_pred_np]))
        pe = 0.0  # Expected agreement

        for cls in classes:
            true_freq = np.sum(y_true_np == cls) / n
            pred_freq = np.sum(y_pred_np == cls) / n
            pe += true_freq * pred_freq

        if pe == 1.0:
            return 1.0 if po == 1.0 else 0.0

        return float((po - pe) / (1 - pe))

    def matthews_corrcoef(self, y_true: Any, y_pred: Any) -> float:
        """Compute Matthews Correlation Coefficient.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            MCC score in [-1, 1].
        """
        y_true_np = _ensure_numpy(y_true).ravel()
        y_pred_np = _ensure_numpy(y_pred).ravel()

        # For binary classification
        tp = np.sum((y_pred_np == 1) & (y_true_np == 1))
        tn = np.sum((y_pred_np == 0) & (y_true_np == 0))
        fp = np.sum((y_pred_np == 1) & (y_true_np == 0))
        fn = np.sum((y_pred_np == 0) & (y_true_np == 1))

        numerator = tp * tn - fp * fn
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        if denominator == 0:
            return 0.0

        return float(numerator / denominator)


    def mse(self, y_true: Any, y_pred: Any) -> float:
        """Compute Mean Squared Error.

        GPU-accelerated for large arrays.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            MSE (lower is better).
        """
        xp = _get_array_module(y_true)
        y_true_arr = xp.asarray(y_true).ravel()
        y_pred_arr = xp.asarray(y_pred).ravel()

        if self._should_use_gpu(len(y_true_arr)):
            result = xp.mean((y_true_arr - y_pred_arr) ** 2)
            return float(result.get() if hasattr(result, "get") else result)

        y_true_np = _ensure_numpy(y_true_arr)
        y_pred_np = _ensure_numpy(y_pred_arr)
        return float(np.mean((y_true_np - y_pred_np) ** 2))

    def rmse(self, y_true: Any, y_pred: Any) -> float:
        """Compute Root Mean Squared Error.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            RMSE (lower is better).
        """
        return float(np.sqrt(self.mse(y_true, y_pred)))

    def mae(self, y_true: Any, y_pred: Any) -> float:
        """Compute Mean Absolute Error.

        GPU-accelerated for large arrays.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            MAE (lower is better).
        """
        xp = _get_array_module(y_true)
        y_true_arr = xp.asarray(y_true).ravel()
        y_pred_arr = xp.asarray(y_pred).ravel()

        if self._should_use_gpu(len(y_true_arr)):
            result = xp.mean(xp.abs(y_true_arr - y_pred_arr))
            return float(result.get() if hasattr(result, "get") else result)

        y_true_np = _ensure_numpy(y_true_arr)
        y_pred_np = _ensure_numpy(y_pred_arr)
        return float(np.mean(np.abs(y_true_np - y_pred_np)))

    def r2(self, y_true: Any, y_pred: Any) -> float:
        """Compute R-squared (coefficient of determination).

        GPU-accelerated for large arrays.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            RÃ‚Â² score (higher is better, max 1.0).
        """
        xp = _get_array_module(y_true)
        y_true_arr = xp.asarray(y_true).ravel()
        y_pred_arr = xp.asarray(y_pred).ravel()

        if self._should_use_gpu(len(y_true_arr)):
            ss_res = xp.sum((y_true_arr - y_pred_arr) ** 2)
            ss_tot = xp.sum((y_true_arr - xp.mean(y_true_arr)) ** 2)
            if hasattr(ss_tot, "get"):
                ss_tot_val = float(ss_tot.get())
                ss_res_val = float(ss_res.get())
            else:
                ss_tot_val = float(ss_tot)
                ss_res_val = float(ss_res)
            return 1 - ss_res_val / ss_tot_val if ss_tot_val > 0 else 0.0

        y_true_np = _ensure_numpy(y_true_arr)
        y_pred_np = _ensure_numpy(y_pred_arr)
        ss_res = np.sum((y_true_np - y_pred_np) ** 2)
        ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def mape(self, y_true: Any, y_pred: Any, eps: float = 1e-10) -> float:
        """Compute Mean Absolute Percentage Error.

        Args:
            y_true: True values.
            y_pred: Predicted values.
            eps: Small value to avoid division by zero.

        Returns:
            MAPE (lower is better).
        """
        y_true_np = _ensure_numpy(y_true).ravel()
        y_pred_np = _ensure_numpy(y_pred).ravel()
        return float(np.mean(np.abs((y_true_np - y_pred_np) / (np.abs(y_true_np) + eps))))

    def explained_variance(self, y_true: Any, y_pred: Any) -> float:
        """Compute Explained Variance Score.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            Explained variance score.
        """
        y_true_np = _ensure_numpy(y_true).ravel()
        y_pred_np = _ensure_numpy(y_pred).ravel()

        residual = y_true_np - y_pred_np
        var_residual = np.var(residual)
        var_y = np.var(y_true_np)

        return float(1 - var_residual / var_y) if var_y > 0 else 0.0

    def max_error(self, y_true: Any, y_pred: Any) -> float:
        """Compute Maximum Error.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            Maximum error (lower is better).
        """
        y_true_np = _ensure_numpy(y_true).ravel()
        y_pred_np = _ensure_numpy(y_pred).ravel()
        return float(np.max(np.abs(y_true_np - y_pred_np)))



    def score(
        self,
        y_true: Any,
        y_pred: Any,
        metric: str | ScoringMetric,
        **kwargs: Any,
    ) -> float:
        """Compute score using specified metric.

        Args:
            y_true: True values/labels.
            y_pred: Predicted values/labels/probabilities.
            metric: Scoring metric.
            **kwargs: Additional arguments for specific metrics.

        Returns:
            Score value.

        Raises:
            ValueError: If metric is not supported.
        """
        if isinstance(metric, ScoringMetric):
            metric_name = metric.value  # pyright: ignore[reportAttributeAccessIssue]
        else:
            metric_name = metric.lower()

        metric_map: dict[str, Callable[..., float]] = {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "log_loss": self.log_loss,
            "cohen_kappa": self.cohen_kappa,
            "matthews_corrcoef": self.matthews_corrcoef,
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "mape": self.mape,
            "explained_variance": self.explained_variance,
            "max_error": self.max_error,
        }

        if metric_name not in metric_map:
            available = list(metric_map.keys())
            msg = f"Unknown metric '{metric_name}'. Available: {available}"
            raise ValueError(msg)

        return metric_map[metric_name](y_true, y_pred, **kwargs)




@dataclass
class PipelineConfig:
    """Configuration for the evaluation pipeline.

    Attributes:
        use_gpu: Whether to use GPU acceleration.
        scoring: Scoring metric name or enum.
        greater_is_better: Whether higher scores are better.
        n_jobs: Number of parallel jobs for CPU operations.
        batch_size: Size of evaluation batches.
        cache_predictions: Whether to cache predictions.
        min_samples_for_gpu: Minimum samples to use GPU.
        timeout_seconds: Timeout for single evaluation.
        return_predictions: Whether to return predictions.
    """

    use_gpu: bool = True
    scoring: str | ScoringMetric = "accuracy"
    greater_is_better: bool | None = None  # Auto-detect from metric
    n_jobs: int = 1
    batch_size: int = 32
    cache_predictions: bool = False
    min_samples_for_gpu: int = 1000
    timeout_seconds: float | None = None
    return_predictions: bool = False

    def __post_init__(self) -> None:
        """Post-initialization validation and defaults."""
        if self.greater_is_better is None:
            if isinstance(self.scoring, ScoringMetric):
                self.greater_is_better = self.scoring.greater_is_better  # pyright: ignore[reportAttributeAccessIssue]
            else:
                # Default based on metric name
                minimize_metrics = {"mse", "rmse", "mae", "log_loss", "mape", "max_error"}
                self.greater_is_better = self.scoring.lower() not in minimize_metrics

    @property
    def metric_needs_proba(self) -> bool:
        """Check if metric needs probability predictions."""
        if isinstance(self.scoring, ScoringMetric):
            return self.scoring.needs_proba  # pyright: ignore[reportAttributeAccessIssue]
        return self.scoring.lower() in {"roc_auc", "log_loss"}


@dataclass
class StageResult:
    """Result from a pipeline stage.

    Attributes:
        stage: The pipeline stage.
        data: Output data from the stage.
        time_seconds: Execution time.
        success: Whether stage completed successfully.
        error: Error message if failed.
    """

    stage: PipelineStage
    data: Any
    time_seconds: float
    success: bool = True
    error: str | None = None


@dataclass
class EvaluationResult:
    """Result of a single evaluation.

    Attributes:
        score: The computed score.
        fit_time: Time for model fitting.
        score_time: Time for scoring.
        total_time: Total evaluation time.
        predictions: Predictions (if requested).
        probabilities: Probability predictions (if available).
        stage_results: Results from each pipeline stage.
        success: Whether evaluation succeeded.
        error: Error message if failed.
    """

    score: float
    fit_time: float
    score_time: float
    total_time: float
    predictions: NDArray | None = None
    probabilities: NDArray | None = None
    stage_results: list[StageResult] = field(default_factory=list)
    success: bool = True
    error: str | None = None


@dataclass
class BatchEvaluationResult:
    """Result of batch evaluation.

    Attributes:
        scores: Array of scores for each configuration.
        mean_score: Mean score across configurations.
        std_score: Standard deviation of scores.
        best_idx: Index of best configuration.
        best_score: Best score achieved.
        total_time: Total batch evaluation time.
        individual_results: Individual evaluation results (if requested).
        n_successful: Number of successful evaluations.
        n_failed: Number of failed evaluations.
    """

    scores: NDArray[np.float64]
    mean_score: float
    std_score: float
    best_idx: int
    best_score: float
    total_time: float
    individual_results: list[EvaluationResult] | None = None
    n_successful: int = 0
    n_failed: int = 0



class PipelineStageExecutor(ABC):
    """Abstract base class for pipeline stage executors."""

    @property
    @abstractmethod
    def stage(self) -> PipelineStage:
        """Get the pipeline stage."""

    @abstractmethod
    def execute(self, data: Any, context: dict[str, Any]) -> StageResult:
        """Execute the pipeline stage.

        Args:
            data: Input data for the stage.
            context: Pipeline context with configuration.

        Returns:
            StageResult with output data.
        """


class FeatureSelectionStage(PipelineStageExecutor):
    """Feature selection pipeline stage."""

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.FEATURE_SELECTION

    def execute(self, data: Any, context: dict[str, Any]) -> StageResult:
        """Apply feature selection mask.

        Args:
            data: Tuple of (X, y) arrays.
            context: Must contain 'feature_mask' key.

        Returns:
            StageResult with selected features.
        """
        start_time = time.perf_counter()

        try:
            X, y = data
            feature_mask = context.get("feature_mask")

            if feature_mask is None:
                # No feature selection, pass through
                return StageResult(
                    stage=self.stage,
                    data=(X, y),
                    time_seconds=time.perf_counter() - start_time,
                )

            # Get array module
            xp = _get_array_module(X)
            feature_mask_arr = xp.asarray(feature_mask)

            # Apply feature selection
            X_selected = X[:, feature_mask_arr]

            return StageResult(
                stage=self.stage,
                data=(X_selected, y),
                time_seconds=time.perf_counter() - start_time,
            )

        except Exception as e:
            return StageResult(
                stage=self.stage,
                data=None,
                time_seconds=time.perf_counter() - start_time,
                success=False,
                error=str(e),
            )


class PreprocessingStage(PipelineStageExecutor):
    """Preprocessing pipeline stage."""

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PREPROCESSING

    def execute(self, data: Any, context: dict[str, Any]) -> StageResult:
        """Apply preprocessing transformations.

        Args:
            data: Tuple of (X_train, y_train, X_val, y_val) or (X, y).
            context: Must contain 'preprocessor' key if preprocessing needed.

        Returns:
            StageResult with preprocessed data.
        """
        start_time = time.perf_counter()

        try:
            preprocessor = context.get("preprocessor")

            if preprocessor is None:
                return StageResult(
                    stage=self.stage,
                    data=data,
                    time_seconds=time.perf_counter() - start_time,
                )

            if len(data) == 4:
                X_train, y_train, X_val, y_val = data
            else:
                X_train, y_train = data
                X_val, y_val = None, None

            X_train_np = _ensure_numpy(X_train)

            if hasattr(preprocessor, "fit_transform"):
                X_train_transformed = preprocessor.fit_transform(X_train_np)
            else:
                preprocessor.fit(X_train_np)
                X_train_transformed = preprocessor.transform(X_train_np)

            if X_val is not None:
                X_val_np = _ensure_numpy(X_val)
                X_val_transformed = preprocessor.transform(X_val_np)
                result_data = (X_train_transformed, y_train, X_val_transformed, y_val)
            else:
                result_data = (X_train_transformed, y_train)

            return StageResult(
                stage=self.stage,
                data=result_data,
                time_seconds=time.perf_counter() - start_time,
            )

        except Exception as e:
            return StageResult(
                stage=self.stage,
                data=None,
                time_seconds=time.perf_counter() - start_time,
                success=False,
                error=str(e),
            )


class ModelFittingStage(PipelineStageExecutor):
    """Model fitting pipeline stage."""

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.MODEL_FITTING

    def execute(self, data: Any, context: dict[str, Any]) -> StageResult:
        """Fit the model to training data.

        Args:
            data: Tuple of (X_train, y_train, X_val, y_val).
            context: Must contain 'model' key.

        Returns:
            StageResult with fitted model.
        """
        start_time = time.perf_counter()

        try:
            model = context.get("model")
            if model is None:
                return StageResult(
                    stage=self.stage,
                    data=None,
                    time_seconds=time.perf_counter() - start_time,
                    success=False,
                    error="No model provided in context",
                )

            # Unpack data
            if len(data) == 4:
                X_train, y_train, X_val, y_val = data
            else:
                X_train, y_train = data
                X_val, y_val = None, None

            # Convert GPU arrays to numpy for sklearn models
            # (cuML models can handle CuPy arrays directly)
            X_train_fit = _ensure_numpy(X_train)
            y_train_fit = _ensure_numpy(y_train)

            model.fit(X_train_fit, y_train_fit)

            return StageResult(
                stage=self.stage,
                data=(model, X_val, y_val),
                time_seconds=time.perf_counter() - start_time,
            )

        except Exception as e:
            return StageResult(
                stage=self.stage,
                data=None,
                time_seconds=time.perf_counter() - start_time,
                success=False,
                error=str(e),
            )


class PredictionStage(PipelineStageExecutor):
    """Prediction pipeline stage."""

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.PREDICTION

    def execute(self, data: Any, context: dict[str, Any]) -> StageResult:
        """Generate predictions.

        Args:
            data: Tuple of (model, X_val, y_val).
            context: Pipeline context.

        Returns:
            StageResult with predictions.
        """
        start_time = time.perf_counter()

        try:
            model, X_val, y_val = data

            if X_val is None:
                return StageResult(
                    stage=self.stage,
                    data=None,
                    time_seconds=time.perf_counter() - start_time,
                    success=False,
                    error="No validation data provided",
                )

            X_val_pred = _ensure_numpy(X_val)

            y_pred = model.predict(X_val_pred)

            needs_proba = context.get("needs_proba", False)
            y_prob = None

            if needs_proba and hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_val_pred)

            return StageResult(
                stage=self.stage,
                data=(y_val, y_pred, y_prob),
                time_seconds=time.perf_counter() - start_time,
            )

        except Exception as e:
            return StageResult(
                stage=self.stage,
                data=None,
                time_seconds=time.perf_counter() - start_time,
                success=False,
                error=str(e),
            )


class ScoringStage(PipelineStageExecutor):
    """Scoring pipeline stage."""

    def __init__(self, scorer: GPUScorer | None = None) -> None:
        """Initialize scoring stage.

        Args:
            scorer: GPU scorer instance.
        """
        self._scorer = scorer or GPUScorer()

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.SCORING

    def execute(self, data: Any, context: dict[str, Any]) -> StageResult:
        """Compute score.

        Args:
            data: Tuple of (y_true, y_pred, y_prob).
            context: Must contain 'metric' key.

        Returns:
            StageResult with computed score.
        """
        start_time = time.perf_counter()

        try:
            y_true, y_pred, y_prob = data
            metric = context.get("metric", "accuracy")

            needs_proba = context.get("needs_proba", False)
            y_for_scoring = y_prob if (needs_proba and y_prob is not None) else y_pred

            score = self._scorer.score(y_true, y_for_scoring, metric)

            return StageResult(
                stage=self.stage,
                data=score,
                time_seconds=time.perf_counter() - start_time,
            )

        except Exception as e:
            return StageResult(
                stage=self.stage,
                data=float("-inf"),
                time_seconds=time.perf_counter() - start_time,
                success=False,
                error=str(e),
            )




class EvaluationPipeline:
    """GPU-accelerated evaluation pipeline.

    Executes the evaluation workflow through configurable stages:
    Feature Selection Preprocessing Model Fitting Prediction Scoring

    Example:
        >>> pipeline = EvaluationPipeline(PipelineConfig(use_gpu=True))
        >>> result = pipeline.evaluate(model, X_train, y_train, X_val, y_val)
        >>> print(f"Score: {result.score:.4f}")
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        scorer: GPUScorer | None = None,
    ) -> None:
        """Initialize evaluation pipeline.

        Args:
            config: Pipeline configuration.
            scorer: GPU scorer instance.
        """
        self.config = config or PipelineConfig()
        self._scorer = scorer or GPUScorer(
            use_gpu=self.config.use_gpu,
            min_samples_for_gpu=self.config.min_samples_for_gpu,
        )

        # Initialize pipeline stages
        self._stages: list[PipelineStageExecutor] = [
            FeatureSelectionStage(),
            PreprocessingStage(),
            ModelFittingStage(),
            PredictionStage(),
            ScoringStage(self._scorer),
        ]

    def evaluate(
        self,
        model: Any,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        feature_mask: NDArray[np.bool_] | None = None,
        preprocessor: Any = None,
    ) -> EvaluationResult:
        """Evaluate a model configuration.

        Args:
            model: Model instance to evaluate.
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            feature_mask: Optional feature selection mask.
            preprocessor: Optional preprocessor.

        Returns:
            EvaluationResult with score and timing.
        """
        start_time = time.perf_counter()

        context = {
            "model": model,
            "feature_mask": feature_mask,
            "preprocessor": preprocessor,
            "metric": self.config.scoring,
            "needs_proba": self.config.metric_needs_proba,
        }

        data = (X_train, y_train, X_val, y_val)
        stage_results: list[StageResult] = []

        for stage_executor in self._stages:
            if stage_executor.stage == PipelineStage.FEATURE_SELECTION:
                X_train, y_train, X_val, y_val = data
                fs_result_train = stage_executor.execute((X_train, y_train), context)
                if not fs_result_train.success:
                    return self._create_failed_result(
                        fs_result_train.error or "Feature selection failed",
                        start_time,
                        stage_results,
                    )
                X_train_sel, _ = fs_result_train.data

                fs_result_val = stage_executor.execute((X_val, y_val), context)
                if not fs_result_val.success:
                    return self._create_failed_result(
                        fs_result_val.error or "Feature selection failed",
                        start_time,
                        stage_results,
                    )
                X_val_sel, _ = fs_result_val.data

                data = (X_train_sel, y_train, X_val_sel, y_val)
                stage_results.append(
                    StageResult(
                        stage=PipelineStage.FEATURE_SELECTION,
                        data=None,
                        time_seconds=fs_result_train.time_seconds + fs_result_val.time_seconds,
                    )
                )
                continue

            result = stage_executor.execute(data, context)
            stage_results.append(result)

            if not result.success:
                return self._create_failed_result(
                    result.error or f"Stage {result.stage.name} failed",
                    start_time,
                    stage_results,
                )

            data = result.data

        score = data if isinstance(data, float) else float(data)  # pyright: ignore[reportArgumentType]
        total_time = time.perf_counter() - start_time

        # Extract timing from stages
        fit_time = sum(
            r.time_seconds
            for r in stage_results
            if r.stage == PipelineStage.MODEL_FITTING
        )
        score_time = sum(
            r.time_seconds
            for r in stage_results
            if r.stage in (PipelineStage.PREDICTION, PipelineStage.SCORING)
        )

        return EvaluationResult(
            score=score,
            fit_time=fit_time,
            score_time=score_time,
            total_time=total_time,
            stage_results=stage_results,
        )

    def _create_failed_result(
        self,
        error: str,
        start_time: float,
        stage_results: list[StageResult],
    ) -> EvaluationResult:
        """Create a failed evaluation result."""
        # Use worst score based on optimization direction
        score = float("-inf") if self.config.greater_is_better else float("inf")

        return EvaluationResult(
            score=score,
            fit_time=0.0,
            score_time=0.0,
            total_time=time.perf_counter() - start_time,
            stage_results=stage_results,
            success=False,
            error=error,
        )



class BatchEvaluator:
    """Batch evaluation for multiple configurations.

    Efficiently evaluates multiple configurations in batches,
    with support for GPU acceleration and parallel CPU execution.

    Example:
        >>> evaluator = BatchEvaluator(pipeline)
        >>> results = evaluator.evaluate_batch(
        ...     model_builder, configurations, X, y, cv_splitter
        ... )
    """

    def __init__(
        self,
        pipeline: EvaluationPipeline | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        """Initialize batch evaluator.

        Args:
            pipeline: Evaluation pipeline.
            config: Pipeline configuration (used if pipeline not provided).
        """
        self.config = config or (pipeline.config if pipeline else PipelineConfig())
        self._pipeline = pipeline or EvaluationPipeline(self.config)

    def evaluate_cv_fold(
        self,
        model_builder: Callable[[], Any],
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        feature_mask: NDArray[np.bool_] | None = None,
        preprocessor: Any = None,
    ) -> EvaluationResult:
        """Evaluate a single CV fold.

        Args:
            model_builder: Callable that creates a model instance.
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            feature_mask: Optional feature selection mask.
            preprocessor: Optional preprocessor.

        Returns:
            EvaluationResult for this fold.
        """
        model = model_builder()
        return self._pipeline.evaluate(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_mask=feature_mask,
            preprocessor=preprocessor,
        )

    def evaluate_cv(
        self,
        model_builder: Callable[[], Any],
        X: Any,
        y: Any,
        cv_splits: list[tuple[NDArray[np.intp], NDArray[np.intp]]],
        feature_mask: NDArray[np.bool_] | None = None,
        preprocessor: Any = None,
    ) -> EvaluationResult:
        """Evaluate using cross-validation.

        Args:
            model_builder: Callable that creates a model instance.
            X: Feature matrix.
            y: Target array.
            cv_splits: List of (train_idx, val_idx) tuples.
            feature_mask: Optional feature selection mask.
            preprocessor: Optional preprocessor factory.

        Returns:
            EvaluationResult with mean CV score.
        """
        start_time = time.perf_counter()
        fold_scores = []
        total_fit_time = 0.0
        total_score_time = 0.0

        X_np = _ensure_numpy(X)
        y_np = _ensure_numpy(y)

        for train_idx, val_idx in cv_splits:
            X_train, X_val = X_np[train_idx], X_np[val_idx]
            y_train, y_val = y_np[train_idx], y_np[val_idx]

            fold_preprocessor = preprocessor() if callable(preprocessor) else preprocessor

            result = self.evaluate_cv_fold(
                model_builder=model_builder,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                feature_mask=feature_mask,
                preprocessor=fold_preprocessor,
            )

            if result.success:
                fold_scores.append(result.score)
                total_fit_time += result.fit_time
                total_score_time += result.score_time
            else:
                logger.warning("CV fold failed: %s", result.error)
                worst_score = float("-inf") if self.config.greater_is_better else float("inf")
                fold_scores.append(worst_score)

        mean_score = float(np.mean(fold_scores))

        return EvaluationResult(
            score=mean_score,
            fit_time=total_fit_time,
            score_time=total_score_time,
            total_time=time.perf_counter() - start_time,
        )

    def evaluate_batch(
        self,
        model_builders: list[Callable[[], Any]],
        X: Any,
        y: Any,
        cv_splits: list[tuple[NDArray[np.intp], NDArray[np.intp]]],
        feature_masks: list[NDArray[np.bool_] | None] | None = None,
        preprocessors: list[Any] | None = None,
        return_individual: bool = False,
    ) -> BatchEvaluationResult:
        """Evaluate a batch of configurations.

        Args:
            model_builders: List of model builder callables.
            X: Feature matrix.
            y: Target array.
            cv_splits: CV splits for evaluation.
            feature_masks: Optional feature masks for each config.
            preprocessors: Optional preprocessors for each config.
            return_individual: Whether to return individual results.

        Returns:
            BatchEvaluationResult with scores and statistics.
        """
        start_time = time.perf_counter()
        n_configs = len(model_builders)

        scores = np.zeros(n_configs, dtype=np.float64)
        individual_results: list[EvaluationResult] = []
        n_successful = 0
        n_failed = 0

        for i, model_builder in enumerate(model_builders):
            feature_mask = feature_masks[i] if feature_masks else None
            preprocessor = preprocessors[i] if preprocessors else None

            try:
                result = self.evaluate_cv(
                    model_builder=model_builder,
                    X=X,
                    y=y,
                    cv_splits=cv_splits,
                    feature_mask=feature_mask,
                    preprocessor=preprocessor,
                )

                scores[i] = result.score

                if result.success:
                    n_successful += 1
                else:
                    n_failed += 1

                if return_individual:
                    individual_results.append(result)

            except Exception as e:
                logger.warning("Batch evaluation %d failed: %s", i, e)
                worst_score = float("-inf") if self.config.greater_is_better else float("inf")
                scores[i] = worst_score
                n_failed += 1

                if return_individual:
                    individual_results.append(
                        EvaluationResult(
                            score=worst_score,
                            fit_time=0.0,
                            score_time=0.0,
                            total_time=0.0,
                            success=False,
                            error=str(e),
                        )
                    )

        valid_scores = scores[np.isfinite(scores)]
        mean_score = float(np.mean(valid_scores)) if len(valid_scores) > 0 else 0.0
        std_score = float(np.std(valid_scores)) if len(valid_scores) > 0 else 0.0

        # Find best
        if self.config.greater_is_better:
            best_idx = int(np.argmax(scores))
        else:
            best_idx = int(np.argmin(scores))
        best_score = float(scores[best_idx])

        return BatchEvaluationResult(
            scores=scores,
            mean_score=mean_score,
            std_score=std_score,
            best_idx=best_idx,
            best_score=best_score,
            total_time=time.perf_counter() - start_time,
            individual_results=individual_results if return_individual else None,
            n_successful=n_successful,
            n_failed=n_failed,
        )

    def evaluate_configurations(
        self,
        configurations: list[dict[str, Any]],
        model_class: type,
        X: Any,
        y: Any,
        cv_splits: list[tuple[NDArray[np.intp], NDArray[np.intp]]],
        fixed_params: dict[str, Any] | None = None,
        feature_masks: list[NDArray[np.bool_] | None] | None = None,
    ) -> BatchEvaluationResult:
        """Evaluate configurations specified as parameter dictionaries.

        Args:
            configurations: List of parameter dictionaries.
            model_class: Model class to instantiate.
            X: Feature matrix.
            y: Target array.
            cv_splits: CV splits for evaluation.
            fixed_params: Fixed parameters for all configurations.
            feature_masks: Optional feature masks.

        Returns:
            BatchEvaluationResult with scores.
        """
        fixed_params = fixed_params or {}

        def make_builder(params: dict[str, Any]) -> Callable[[], Any]:
            full_params = {**fixed_params, **params}
            return lambda: model_class(**full_params)

        model_builders = [make_builder(cfg) for cfg in configurations]

        return self.evaluate_batch(
            model_builders=model_builders,
            X=X,
            y=y,
            cv_splits=cv_splits,
            feature_masks=feature_masks,
        )




def create_cv_splits(
    n_samples: int,
    n_splits: int = 5,
    y: NDArray | None = None,
    stratified: bool = True,
    shuffle: bool = True,
    random_state: int | None = None,
) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Create cross-validation splits.

    Args:
        n_samples: Number of samples.
        n_splits: Number of CV folds.
        y: Target array (for stratified splitting).
        stratified: Use stratified splitting.
        shuffle: Shuffle before splitting.
        random_state: Random seed.

    Returns:
        List of (train_idx, val_idx) tuples.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    if stratified and y is not None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        indices = list(cv.split(np.zeros(n_samples), y))
    else:
        cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        indices = list(cv.split(np.zeros(n_samples)))

    return [(train_idx.astype(np.intp), val_idx.astype(np.intp)) for train_idx, val_idx in indices]


def get_scorer_function(
    metric: str | ScoringMetric,
    use_gpu: bool = True,
) -> Callable[[Any, Any], float]:
    """Get a scoring function by name.

    Args:
        metric: Metric name or enum.
        use_gpu: Whether to use GPU acceleration.

    Returns:
        Scoring function.
    """
    scorer = GPUScorer(use_gpu=use_gpu)

    def score_fn(y_true: Any, y_pred: Any) -> float:
        return scorer.score(y_true, y_pred, metric)

    return score_fn


def needs_probability_predictions(metric: str | ScoringMetric) -> bool:
    """Check if a metric needs probability predictions.

    Args:
        metric: Metric name or enum.

    Returns:
        True if metric needs probabilities.
    """
    if isinstance(metric, ScoringMetric):
        return metric.needs_proba  # pyright: ignore[reportAttributeAccessIssue]

    proba_metrics = {"roc_auc", "log_loss"}
    return metric.lower() in proba_metrics




class GPUBatchScorer:
    """GPU-optimized batch scoring operations.

    Efficiently scores multiple predictions simultaneously on GPU.
    """

    def __init__(self, use_gpu: bool = True) -> None:
        """Initialize GPU batch scorer.

        Args:
            use_gpu: Whether to use GPU.
        """
        self.use_gpu = use_gpu
        self._gpu_available = self._check_gpu()

    def _check_gpu(self) -> bool:
        """Check GPU availability."""
        if not self.use_gpu:
            return False
        try:
            import cupy as cp

            _ = cp.array([1, 2, 3])
            return True
        except (ImportError, Exception):
            return False

    def batch_accuracy(
        self,
        y_true: Any,
        y_preds: list[Any],
    ) -> NDArray[np.float64]:
        """Compute accuracy for multiple predictions.

        Args:
            y_true: True labels (shared).
            y_preds: List of prediction arrays.

        Returns:
            Array of accuracy scores.
        """
        n_configs = len(y_preds)
        scores = np.zeros(n_configs, dtype=np.float64)

        y_true_np = _ensure_numpy(y_true).ravel()

        for i, y_pred in enumerate(y_preds):
            y_pred_np = _ensure_numpy(y_pred).ravel()
            scores[i] = np.mean(y_true_np == y_pred_np)

        return scores

    def batch_mse(
        self,
        y_true: Any,
        y_preds: list[Any],
    ) -> NDArray[np.float64]:
        """Compute MSE for multiple predictions.

        Args:
            y_true: True values (shared).
            y_preds: List of prediction arrays.

        Returns:
            Array of MSE scores.
        """
        n_configs = len(y_preds)
        scores = np.zeros(n_configs, dtype=np.float64)

        y_true_np = _ensure_numpy(y_true).ravel()

        for i, y_pred in enumerate(y_preds):
            y_pred_np = _ensure_numpy(y_pred).ravel()
            scores[i] = np.mean((y_true_np - y_pred_np) ** 2)

        return scores

    def batch_r2(
        self,
        y_true: Any,
        y_preds: list[Any],
    ) -> NDArray[np.float64]:
        """Compute RÃ‚Â² for multiple predictions.

        Args:
            y_true: True values (shared).
            y_preds: List of prediction arrays.

        Returns:
            Array of RÃ‚Â² scores.
        """
        n_configs = len(y_preds)
        scores = np.zeros(n_configs, dtype=np.float64)

        y_true_np = _ensure_numpy(y_true).ravel()
        ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2)

        if ss_tot == 0:
            return scores

        for i, y_pred in enumerate(y_preds):
            y_pred_np = _ensure_numpy(y_pred).ravel()
            ss_res = np.sum((y_true_np - y_pred_np) ** 2)
            scores[i] = 1 - ss_res / ss_tot

        return scores