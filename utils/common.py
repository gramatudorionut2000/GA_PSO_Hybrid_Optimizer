
from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass




class TaskType(Enum):
    """Machine learning task type."""

    CLASSIFICATION = auto()
    REGRESSION = auto()
    MULTICLASS = auto()


class ScoringMetric(Enum):
    """Supported scoring metrics."""

    # Classification metrics
    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    ROC_AUC = "roc_auc"
    LOG_LOSS = "log_loss"
    COHEN_KAPPA = "cohen_kappa"
    MATTHEWS_CORRCOEF = "matthews_corrcoef"

    # Regression metrics
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"
    MAPE = "mape"
    EXPLAINED_VARIANCE = "explained_variance"
    MAX_ERROR = "max_error"

    @property
    def is_classification(self) -> bool:
        """Check if metric is for classification."""
        classification_metrics = {
            self.ACCURACY,
            self.BALANCED_ACCURACY,
            self.PRECISION,
            self.RECALL,
            self.F1,
            self.ROC_AUC,
            self.LOG_LOSS,
            self.COHEN_KAPPA,
            self.MATTHEWS_CORRCOEF,
        }
        return self in classification_metrics

    @property
    def greater_is_better(self) -> bool:
        """Check if higher values are better."""
        minimize_metrics = {
            self.MSE,
            self.RMSE,
            self.MAE,
            self.LOG_LOSS,
            self.MAPE,
            self.MAX_ERROR,
        }
        return self not in minimize_metrics

    @property
    def needs_proba(self) -> bool:
        """Check if metric needs probability predictions."""
        return self in {self.ROC_AUC, self.LOG_LOSS}




def get_array_module(arr: Any) -> Any:
    """Get the array module (numpy or cupy) for an array.

    Args:
        arr: Input array (numpy or cupy).

    Returns:
        The appropriate array module (np or cp).
    """
    if hasattr(arr, "__cuda_array_interface__"):
        try:
            import cupy as cp

            return cp
        except ImportError:
            pass
    return np


def ensure_numpy(arr: Any) -> NDArray:
    """Convert array to numpy if needed.

    Handles CuPy arrays, lists, and other array-like objects.

    Args:
        arr: Input array-like object.

    Returns:
        NumPy ndarray.
    """
    if hasattr(arr, "get"):  # CuPy array
        return arr.get()
    return np.asarray(arr)


def ensure_contiguous(arr: Any) -> Any:
    """Ensure array is contiguous in memory.

    Args:
        arr: Input array.

    Returns:
        Contiguous array (same module as input).
    """
    xp = get_array_module(arr)
    if not arr.flags.c_contiguous:
        return xp.ascontiguousarray(arr)
    return arr


def is_gpu_array(arr: Any) -> bool:
    """Check if array is a GPU array.

    Args:
        arr: Input array.

    Returns:
        True if array has CUDA interface.
    """
    return hasattr(arr, "__cuda_array_interface__")


def to_device(arr: NDArray, use_gpu: bool = False) -> Any:
    """Transfer array to specified device.

    Args:
        arr: NumPy array.
        use_gpu: If True, transfer to GPU.

    Returns:
        Array on the target device.
    """
    if not use_gpu:
        return ensure_numpy(arr)

    try:
        import cupy as cp

        if is_gpu_array(arr):
            return arr
        return cp.asarray(arr)
    except ImportError:
        return arr


def check_gpu_available() -> bool:
    """Check if GPU (CuPy) is available.

    Returns:
        True if CuPy is available and functional.
    """
    try:
        import cupy as cp

        _ = cp.array([1, 2, 3])
        return True
    except (ImportError, Exception):
        return False




def batch_to_numpy(arrays: list[Any]) -> list[NDArray]:
    """Convert a list of arrays to numpy.

    Args:
        arrays: List of arrays (numpy or cupy).

    Returns:
        List of numpy arrays.
    """
    return [ensure_numpy(arr) for arr in arrays]


def concatenate_arrays(arrays: list[Any], axis: int = 0) -> Any:
    """Concatenate arrays using the appropriate module.

    Args:
        arrays: List of arrays to concatenate.
        axis: Axis along which to concatenate.

    Returns:
        Concatenated array.
    """
    if not arrays:
        return np.array([])

    xp = get_array_module(arrays[0])
    return xp.concatenate(arrays, axis=axis)


def stack_arrays(arrays: list[Any], axis: int = 0) -> Any:
    """Stack arrays using the appropriate module.

    Args:
        arrays: List of arrays to stack.
        axis: Axis along which to stack.

    Returns:
        Stacked array.
    """
    if not arrays:
        return np.array([])

    xp = get_array_module(arrays[0])
    return xp.stack(arrays, axis=axis)