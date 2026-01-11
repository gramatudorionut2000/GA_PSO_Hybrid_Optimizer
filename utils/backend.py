
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

# Configure logging
logger = logging.getLogger(__name__)




class DeviceType(Enum):
    """Enumeration of supported device types."""

    CPU = auto()
    GPU = auto()
    AUTO = auto()


ArrayLike = NDArray | Any

DType = Literal[
    "float32",
    "float64",
    "int32",
    "int64",
    "bool",
    "uint8",
    "uint32",
    "uint64",
]



def _check_cupy_available() -> bool:
    """Check if CuPy is available and functional."""
    try:
        import cupy as cp  # noqa: F401

        # Try a simple operation to ensure GPU is accessible
        _ = cp.array([1, 2, 3])
        return True
    except ImportError:
        logger.debug("CuPy not installed")
        return False
    except Exception as e:
        logger.debug("CuPy available but GPU not functional: %s", e)
        return False


def _get_gpu_memory_info() -> tuple[int, int] | None:
    """Get GPU memory info (free, total) in bytes.

    Returns:
        Tuple of (free_bytes, total_bytes) or None if unavailable.
    """
    try:
        import cupy as cp

        mempool = cp.get_default_memory_pool()  # noqa: F841
        device = cp.cuda.Device()
        free, total = device.mem_info
        return free, total
    except Exception:
        return None


def _get_gpu_device_count() -> int:
    """Get the number of available GPU devices."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


_CUPY_AVAILABLE: bool | None = None


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available.

    Returns:
        True if CuPy and CUDA are available and functional.
    """
    global _CUPY_AVAILABLE  # noqa: PLW0603
    if _CUPY_AVAILABLE is None:
        _CUPY_AVAILABLE = _check_cupy_available()
    return _CUPY_AVAILABLE




@dataclass(frozen=True)
class GPUInfo:
    """Information about GPU availability and status."""

    available: bool
    device_count: int
    current_device: int
    free_memory: int
    total_memory: int
    used_memory: int
    memory_utilization: float
    device_name: str

    @classmethod
    def query(cls) -> GPUInfo:
        """Query current GPU information."""
        if not is_gpu_available():
            return cls(
                available=False,
                device_count=0,
                current_device=-1,
                free_memory=0,
                total_memory=0,
                used_memory=0,
                memory_utilization=0.0,
                device_name="N/A",
            )

        try:
            import cupy as cp

            device = cp.cuda.Device()
            free, total = device.mem_info
            used = total - free
            utilization = used / total if total > 0 else 0.0

            # Get device name
            props = cp.cuda.runtime.getDeviceProperties(device.id)
            device_name = props.get("name", b"Unknown").decode("utf-8")

            return cls(
                available=True,
                device_count=_get_gpu_device_count(),
                current_device=device.id,
                free_memory=free,
                total_memory=total,
                used_memory=used,
                memory_utilization=utilization,
                device_name=device_name,
            )
        except Exception as e:
            logger.warning("Failed to query GPU info: %s", e)
            return cls(
                available=False,
                device_count=0,
                current_device=-1,
                free_memory=0,
                total_memory=0,
                used_memory=0,
                memory_utilization=0.0,
                device_name="Error",
            )

    def __str__(self) -> str:
        """Human-readable string representation."""
        if not self.available:
            return "GPU: Not available"

        free_gb = self.free_memory / (1024**3)
        total_gb = self.total_memory / (1024**3)
        return (
            f"GPU: {self.device_name} (Device {self.current_device})\n"
            f"  Memory: {free_gb:.2f} GB free / {total_gb:.2f} GB total "
            f"({self.memory_utilization:.1%} used)"
        )




class ArrayBackend(ABC):
    """Abstract base class for array computation backends.

    This interface defines all operations that can be performed on arrays,
    allowing transparent switching between NumPy (CPU) and CuPy (GPU).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name ('numpy' or 'cupy')."""

    @property
    @abstractmethod
    def device_type(self) -> DeviceType:
        """Device type (CPU or GPU)."""

    @property
    @abstractmethod
    def xp(self) -> Any:
        """Get the array module (numpy or cupy)."""



    @abstractmethod
    def zeros(
        self,
        shape: tuple[int, ...] | int,
        dtype: DType = "float64",
    ) -> ArrayLike:
        """Create array filled with zeros."""

    @abstractmethod
    def ones(
        self,
        shape: tuple[int, ...] | int,
        dtype: DType = "float64",
    ) -> ArrayLike:
        """Create array filled with ones."""

    @abstractmethod
    def full(
        self,
        shape: tuple[int, ...] | int,
        fill_value: float | int | bool,
        dtype: DType = "float64",
    ) -> ArrayLike:
        """Create array filled with a value."""

    @abstractmethod
    def empty(
        self,
        shape: tuple[int, ...] | int,
        dtype: DType = "float64",
    ) -> ArrayLike:
        """Create uninitialized array."""

    @abstractmethod
    def arange(
        self,
        start: float,
        stop: float | None = None,
        step: float = 1.0,
        dtype: DType | None = None,
    ) -> ArrayLike:
        """Create array with evenly spaced values."""

    @abstractmethod
    def linspace(
        self,
        start: float,
        stop: float,
        num: int,
        dtype: DType = "float64",
    ) -> ArrayLike:
        """Create array with linearly spaced values."""

    @abstractmethod
    def eye(self, n: int, dtype: DType = "float64") -> ArrayLike:
        """Create identity matrix."""

    @abstractmethod
    def asarray(
        self,
        a: ArrayLike | Sequence[Any],
        dtype: DType | None = None,
    ) -> ArrayLike:
        """Convert input to array on this backend."""


    @abstractmethod
    def set_seed(self, seed: int | None) -> None:
        """Set random seed for reproducibility."""

    @abstractmethod
    def random_uniform(
        self,
        low: float,
        high: float,
        size: tuple[int, ...] | int | None = None,
    ) -> ArrayLike:
        """Generate uniform random values in [low, high)."""

    @abstractmethod
    def random_normal(
        self,
        mean: float,
        std: float,
        size: tuple[int, ...] | int | None = None,
    ) -> ArrayLike:
        """Generate normal (Gaussian) random values."""

    @abstractmethod
    def random_integers(
        self,
        low: int,
        high: int,
        size: tuple[int, ...] | int | None = None,
    ) -> ArrayLike:
        """Generate random integers in [low, high]."""

    @abstractmethod
    def random_choice(
        self,
        n: int,
        size: int | None = None,
        p: ArrayLike | None = None,
        *,
        replace: bool = True,
    ) -> ArrayLike:
        """Random choice from range(n)."""

    @abstractmethod
    def random_permutation(self, n: int) -> ArrayLike:
        """Generate random permutation of range(n)."""


    @abstractmethod
    def exp(self, x: ArrayLike) -> ArrayLike:
        """Element-wise exponential."""

    @abstractmethod
    def log(self, x: ArrayLike) -> ArrayLike:
        """Element-wise natural logarithm."""

    @abstractmethod
    def sqrt(self, x: ArrayLike) -> ArrayLike:
        """Element-wise square root."""

    @abstractmethod
    def power(self, x: ArrayLike, p: float | ArrayLike) -> ArrayLike:
        """Element-wise power."""

    @abstractmethod
    def abs(self, x: ArrayLike) -> ArrayLike:
        """Element-wise absolute value."""

    @abstractmethod
    def clip(
        self,
        x: ArrayLike,
        a_min: float | ArrayLike | None,
        a_max: float | ArrayLike | None,
    ) -> ArrayLike:
        """Clip values to range [a_min, a_max]."""

    @abstractmethod
    def sin(self, x: ArrayLike) -> ArrayLike:
        """Element-wise sine."""

    @abstractmethod
    def cos(self, x: ArrayLike) -> ArrayLike:
        """Element-wise cosine."""



    @abstractmethod
    def sum(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> ArrayLike:
        """Sum of array elements."""

    @abstractmethod
    def mean(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> ArrayLike:
        """Mean of array elements."""

    @abstractmethod
    def std(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> ArrayLike:
        """Standard deviation of array elements."""

    @abstractmethod
    def var(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> ArrayLike:
        """Variance of array elements."""

    @abstractmethod
    def min(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> ArrayLike:
        """Minimum of array elements."""

    @abstractmethod
    def max(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> ArrayLike:
        """Maximum of array elements."""

    @abstractmethod
    def argmin(self, x: ArrayLike, axis: int | None = None) -> ArrayLike:
        """Indices of minimum values."""

    @abstractmethod
    def argmax(self, x: ArrayLike, axis: int | None = None) -> ArrayLike:
        """Indices of maximum values."""

    @abstractmethod
    def median(self, x: ArrayLike, axis: int | None = None) -> ArrayLike:
        """Median of array elements."""

    @abstractmethod
    def percentile(
        self,
        x: ArrayLike,
        q: float | ArrayLike,
        axis: int | None = None,
    ) -> ArrayLike:
        """Compute percentiles."""



    @abstractmethod
    def dot(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        """Dot product of two arrays."""

    @abstractmethod
    def matmul(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        """Matrix multiplication."""

    @abstractmethod
    def norm(
        self,
        x: ArrayLike,
        ord: int | float | str | None = None,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> ArrayLike:
        """Matrix or vector norm."""



    @abstractmethod
    def where(
        self,
        condition: ArrayLike,
        x: ArrayLike | float,
        y: ArrayLike | float,
    ) -> ArrayLike:
        """Return elements chosen from x or y depending on condition."""

    @abstractmethod
    def logical_and(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        """Element-wise logical AND."""

    @abstractmethod
    def logical_or(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        """Element-wise logical OR."""

    @abstractmethod
    def logical_not(self, x: ArrayLike) -> ArrayLike:
        """Element-wise logical NOT."""

    @abstractmethod
    def logical_xor(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        """Element-wise logical XOR."""

    @abstractmethod
    def all(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
    ) -> ArrayLike:
        """Test whether all elements are True."""

    @abstractmethod
    def any(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
    ) -> ArrayLike:
        """Test whether any element is True."""


    @abstractmethod
    def reshape(self, x: ArrayLike, shape: tuple[int, ...]) -> ArrayLike:
        """Reshape array."""

    @abstractmethod
    def transpose(
        self,
        x: ArrayLike,
        axes: tuple[int, ...] | None = None,
    ) -> ArrayLike:
        """Transpose array."""

    @abstractmethod
    def concatenate(
        self,
        arrays: Sequence[ArrayLike],
        axis: int = 0,
    ) -> ArrayLike:
        """Concatenate arrays along an axis."""

    @abstractmethod
    def stack(self, arrays: Sequence[ArrayLike], axis: int = 0) -> ArrayLike:
        """Stack arrays along a new axis."""

    @abstractmethod
    def vstack(self, arrays: Sequence[ArrayLike]) -> ArrayLike:
        """Stack arrays vertically."""

    @abstractmethod
    def hstack(self, arrays: Sequence[ArrayLike]) -> ArrayLike:
        """Stack arrays horizontally."""

    @abstractmethod
    def split(
        self,
        x: ArrayLike,
        indices_or_sections: int | Sequence[int],
        axis: int = 0,
    ) -> list[ArrayLike]:
        """Split array into multiple sub-arrays."""

    @abstractmethod
    def unique(
        self,
        x: ArrayLike,
        return_counts: bool = False,
    ) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
        """Find unique elements."""

    @abstractmethod
    def argsort(self, x: ArrayLike, axis: int = -1) -> ArrayLike:
        """Returns indices that would sort an array."""

    @abstractmethod
    def sort(self, x: ArrayLike, axis: int = -1) -> ArrayLike:
        """Sort array."""

    @abstractmethod
    def copy(self, x: ArrayLike) -> ArrayLike:
        """Return a copy of the array."""


    @abstractmethod
    def triu_indices(self, n: int, k: int = 0) -> tuple[ArrayLike, ArrayLike]:
        """Return indices for upper triangle."""

    @abstractmethod
    def tril_indices(self, n: int, k: int = 0) -> tuple[ArrayLike, ArrayLike]:
        """Return indices for lower triangle."""



    @abstractmethod
    def to_cpu(self, x: ArrayLike) -> NDArray:
        """Transfer array to CPU (NumPy array)."""

    @abstractmethod
    def to_device(self, x: ArrayLike) -> ArrayLike:
        """Transfer array to this backend's device."""



    @abstractmethod
    def rint(self, x: ArrayLike) -> ArrayLike:
        """Round to nearest integer."""

    @abstractmethod
    def floor(self, x: ArrayLike) -> ArrayLike:
        """Floor of input."""

    @abstractmethod
    def ceil(self, x: ArrayLike) -> ArrayLike:
        """Ceiling of input."""


    def is_array(self, x: Any) -> bool:
        """Check if x is an array of this backend's type."""
        return isinstance(x, self.xp.ndarray)

    def to_scalar(self, x: ArrayLike) -> float | int | bool:
        """Convert 0-d array or scalar to Python scalar."""
        if hasattr(x, "item"):
            return x.item()
        return x  # pyright: ignore[reportReturnType]

    def ensure_2d(self, x: ArrayLike) -> ArrayLike:
        """Ensure array is at least 2D."""
        x = self.asarray(x)
        if x.ndim == 1:
            return self.reshape(x, (-1, 1))
        return x





class NumpyBackend(ArrayBackend):
    """NumPy-based backend for CPU computations."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize NumPy backend.

        Args:
            seed: Random seed for reproducibility.
        """
        self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "numpy"

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.CPU

    @property
    def xp(self) -> Any:
        return np


    def zeros(
        self,
        shape: tuple[int, ...] | int,
        dtype: DType = "float64",
    ) -> NDArray:
        return np.zeros(shape, dtype=dtype)

    def ones(
        self,
        shape: tuple[int, ...] | int,
        dtype: DType = "float64",
    ) -> NDArray:
        return np.ones(shape, dtype=dtype)

    def full(
        self,
        shape: tuple[int, ...] | int,
        fill_value: float | int | bool,
        dtype: DType = "float64",
    ) -> NDArray:
        return np.full(shape, fill_value, dtype=dtype)

    def empty(
        self,
        shape: tuple[int, ...] | int,
        dtype: DType = "float64",
    ) -> NDArray:
        return np.empty(shape, dtype=dtype)

    def arange(
        self,
        start: float,
        stop: float | None = None,
        step: float = 1.0,
        dtype: DType | None = None,
    ) -> NDArray:
        return np.arange(start, stop, step, dtype=dtype)

    def linspace(
        self,
        start: float,
        stop: float,
        num: int,
        dtype: DType = "float64",
    ) -> NDArray:
        return np.linspace(start, stop, num, dtype=dtype)

    def eye(self, n: int, dtype: DType = "float64") -> NDArray:
        return np.eye(n, dtype=dtype)

    def asarray(
        self,
        a: ArrayLike | Sequence[Any],
        dtype: DType | None = None,
    ) -> NDArray:
        return np.asarray(a, dtype=dtype)



    def set_seed(self, seed: int | None) -> None:
        self._rng = np.random.default_rng(seed)

    def random_uniform(
        self,
        low: float,
        high: float,
        size: tuple[int, ...] | int | None = None,
    ) -> NDArray:
        return self._rng.uniform(low, high, size)

    def random_normal(
        self,
        mean: float,
        std: float,
        size: tuple[int, ...] | int | None = None,
    ) -> NDArray:
        return self._rng.normal(mean, std, size)

    def random_integers(
        self,
        low: int,
        high: int,
        size: tuple[int, ...] | int | None = None,
    ) -> NDArray:
        return self._rng.integers(low, high + 1, size)

    def random_choice(
        self,
        n: int,
        size: int | None = None,
        p: ArrayLike | None = None,
        *,
        replace: bool = True,
    ) -> NDArray:
        return self._rng.choice(n, size=size, p=p, replace=replace)  # pyright: ignore[reportReturnType]

    def random_permutation(self, n: int) -> NDArray:
        return self._rng.permutation(n)



    def exp(self, x: ArrayLike) -> NDArray:
        return np.exp(x)

    def log(self, x: ArrayLike) -> NDArray:
        return np.log(x)

    def sqrt(self, x: ArrayLike) -> NDArray:
        return np.sqrt(x)

    def power(self, x: ArrayLike, p: float | ArrayLike) -> NDArray:
        return np.power(x, p)

    def abs(self, x: ArrayLike) -> NDArray:
        return np.abs(x)

    def clip(
        self,
        x: ArrayLike,
        a_min: float | ArrayLike | None,
        a_max: float | ArrayLike | None,
    ) -> NDArray:
        return np.clip(x, a_min, a_max)

    def sin(self, x: ArrayLike) -> NDArray:
        return np.sin(x)

    def cos(self, x: ArrayLike) -> NDArray:
        return np.cos(x)


    def sum(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> NDArray:
        return np.sum(x, axis=axis, keepdims=keepdims)

    def mean(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> NDArray:
        return np.mean(x, axis=axis, keepdims=keepdims)

    def std(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> NDArray:
        return np.std(x, axis=axis, ddof=ddof, keepdims=keepdims)

    def var(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> NDArray:
        return np.var(x, axis=axis, ddof=ddof, keepdims=keepdims)

    def min(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> NDArray:
        return np.min(x, axis=axis, keepdims=keepdims)

    def max(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> NDArray:
        return np.max(x, axis=axis, keepdims=keepdims)

    def argmin(self, x: ArrayLike, axis: int | None = None) -> NDArray:
        return np.argmin(x, axis=axis)

    def argmax(self, x: ArrayLike, axis: int | None = None) -> NDArray:
        return np.argmax(x, axis=axis)

    def median(self, x: ArrayLike, axis: int | None = None) -> NDArray:
        return np.median(x, axis=axis)

    def percentile(
        self,
        x: ArrayLike,
        q: float | ArrayLike,
        axis: int | None = None,
    ) -> NDArray:
        return np.percentile(x, q, axis=axis)



    def dot(self, a: ArrayLike, b: ArrayLike) -> NDArray:
        return np.dot(a, b)

    def matmul(self, a: ArrayLike, b: ArrayLike) -> NDArray:
        return np.matmul(a, b)

    def norm(
        self,
        x: ArrayLike,
        ord: int | float | str | None = None,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> NDArray:
        return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)  # pyright: ignore[reportCallIssue, reportArgumentType]


    def where(
        self,
        condition: ArrayLike,
        x: ArrayLike | float,
        y: ArrayLike | float,
    ) -> NDArray:
        return np.where(condition, x, y)

    def logical_and(self, x1: ArrayLike, x2: ArrayLike) -> NDArray:
        return np.logical_and(x1, x2)

    def logical_or(self, x1: ArrayLike, x2: ArrayLike) -> NDArray:
        return np.logical_or(x1, x2)

    def logical_not(self, x: ArrayLike) -> NDArray:
        return np.logical_not(x)

    def logical_xor(self, x1: ArrayLike, x2: ArrayLike) -> NDArray:
        return np.logical_xor(x1, x2)

    def all(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
    ) -> NDArray:
        return np.all(x, axis=axis)

    def any(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
    ) -> NDArray:
        return np.any(x, axis=axis)



    def reshape(self, x: ArrayLike, shape: tuple[int, ...]) -> NDArray:
        return np.reshape(x, shape)

    def transpose(
        self,
        x: ArrayLike,
        axes: tuple[int, ...] | None = None,
    ) -> NDArray:
        return np.transpose(x, axes)

    def concatenate(
        self,
        arrays: Sequence[ArrayLike],
        axis: int = 0,
    ) -> NDArray:
        return np.concatenate(arrays, axis=axis)

    def stack(self, arrays: Sequence[ArrayLike], axis: int = 0) -> NDArray:
        return np.stack(arrays, axis=axis)

    def vstack(self, arrays: Sequence[ArrayLike]) -> NDArray:
        return np.vstack(arrays)

    def hstack(self, arrays: Sequence[ArrayLike]) -> NDArray:
        return np.hstack(arrays)

    def split(
        self,
        x: ArrayLike,
        indices_or_sections: int | Sequence[int],
        axis: int = 0,
    ) -> list[NDArray]:
        return list(np.split(x, indices_or_sections, axis=axis))

    def unique(
        self,
        x: ArrayLike,
        return_counts: bool = False,
    ) -> NDArray | tuple[NDArray, NDArray]:
        return np.unique(x, return_counts=return_counts)

    def argsort(self, x: ArrayLike, axis: int = -1) -> NDArray:
        return np.argsort(x, axis=axis)

    def sort(self, x: ArrayLike, axis: int = -1) -> NDArray:
        return np.sort(x, axis=axis)

    def copy(self, x: ArrayLike) -> NDArray:
        return np.copy(x)



    def triu_indices(self, n: int, k: int = 0) -> tuple[NDArray, NDArray]:
        return np.triu_indices(n, k)

    def tril_indices(self, n: int, k: int = 0) -> tuple[NDArray, NDArray]:
        return np.tril_indices(n, k)


    def to_cpu(self, x: ArrayLike) -> NDArray:
        return np.asarray(x)

    def to_device(self, x: ArrayLike) -> NDArray:
        return np.asarray(x)



    def rint(self, x: ArrayLike) -> NDArray:
        return np.rint(x)

    def floor(self, x: ArrayLike) -> NDArray:
        return np.floor(x)

    def ceil(self, x: ArrayLike) -> NDArray:
        return np.ceil(x)



    def pairwise_distances(
        self,
        x: ArrayLike,
        y: ArrayLike | None = None,
        metric: str = "euclidean",
    ) -> NDArray:
        """Compute pairwise distances."""
        if y is None:
            y = x
        return cdist(x, y, metric=metric)  # pyright: ignore[reportCallIssue, reportArgumentType]

    def rankdata(self, x: ArrayLike, method: str = "average") -> NDArray:
        """Compute ranks."""
        return scipy_stats.rankdata(x, method=method)




class CupyBackend(ArrayBackend):
    """CuPy-based backend for GPU computations.

    This backend requires CuPy and CUDA to be installed and functional.
    Falls back to NumPy operations for unsupported operations.
    """

    def __init__(
        self,
        seed: int | None = None,
        device_id: int = 0,
    ) -> None:
        """Initialize CuPy backend.

        Args:
            seed: Random seed for reproducibility.
            device_id: CUDA device ID to use.

        Raises:
            RuntimeError: If CuPy/CUDA is not available.
        """
        if not is_gpu_available():
            msg = "CuPy/CUDA is not available"
            raise RuntimeError(msg)

        import cupy as cp

        self._cp = cp
        self._device_id = device_id

        # Set device
        cp.cuda.Device(device_id).use()

        # Initialize RNG
        self._rng = cp.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "cupy"

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.GPU

    @property
    def xp(self) -> Any:
        return self._cp

    @property
    def device_id(self) -> int:
        """Current CUDA device ID."""
        return self._device_id


    def zeros(
        self,
        shape: tuple[int, ...] | int,
        dtype: DType = "float64",
    ) -> ArrayLike:
        return self._cp.zeros(shape, dtype=dtype)

    def ones(
        self,
        shape: tuple[int, ...] | int,
        dtype: DType = "float64",
    ) -> ArrayLike:
        return self._cp.ones(shape, dtype=dtype)

    def full(
        self,
        shape: tuple[int, ...] | int,
        fill_value: float | int | bool,
        dtype: DType = "float64",
    ) -> ArrayLike:
        return self._cp.full(shape, fill_value, dtype=dtype)

    def empty(
        self,
        shape: tuple[int, ...] | int,
        dtype: DType = "float64",
    ) -> ArrayLike:
        return self._cp.empty(shape, dtype=dtype)

    def arange(
        self,
        start: float,
        stop: float | None = None,
        step: float = 1.0,
        dtype: DType | None = None,
    ) -> ArrayLike:
        return self._cp.arange(start, stop, step, dtype=dtype)  # pyright: ignore[reportArgumentType]

    def linspace(
        self,
        start: float,
        stop: float,
        num: int,
        dtype: DType = "float64",
    ) -> ArrayLike:
        return self._cp.linspace(start, stop, num, dtype=dtype)

    def eye(self, n: int, dtype: DType = "float64") -> ArrayLike:
        return self._cp.eye(n, dtype=dtype)

    def asarray(
        self,
        a: ArrayLike | Sequence[Any],
        dtype: DType | None = None,
    ) -> ArrayLike:
        return self._cp.asarray(a, dtype=dtype)



    def set_seed(self, seed: int | None) -> None:
        self._rng = self._cp.random.default_rng(seed)

    def random_uniform(
        self,
        low: float,
        high: float,
        size: tuple[int, ...] | int | None = None,
    ) -> ArrayLike:
        return self._rng.uniform(low, high, size)  # pyright: ignore[reportOptionalMemberAccess]

    def random_normal(
        self,
        mean: float,
        std: float,
        size: tuple[int, ...] | int | None = None,
    ) -> ArrayLike:
        return self._rng.normal(mean, std, size)  # pyright: ignore[reportOptionalMemberAccess]

    def random_integers(
        self,
        low: int,
        high: int,
        size: tuple[int, ...] | int | None = None,
    ) -> ArrayLike:
        return self._rng.integers(low, high + 1, size)  # pyright: ignore[reportOptionalMemberAccess]

    def random_choice(
        self,
        n: int,
        size: int | None = None,
        p: ArrayLike | None = None,
        *,
        replace: bool = True,
    ) -> ArrayLike:
        # Convert p to CuPy array if provided
        if p is not None:
            p = self._cp.asarray(p)
        return self._rng.choice(n, size=size, p=p, replace=replace)  # pyright: ignore[reportOptionalMemberAccess]

    def random_permutation(self, n: int) -> ArrayLike:
        return self._rng.permutation(n)  # pyright: ignore[reportOptionalMemberAccess]


    def exp(self, x: ArrayLike) -> ArrayLike:
        return self._cp.exp(x)

    def log(self, x: ArrayLike) -> ArrayLike:
        return self._cp.log(x)

    def sqrt(self, x: ArrayLike) -> ArrayLike:
        return self._cp.sqrt(x)

    def power(self, x: ArrayLike, p: float | ArrayLike) -> ArrayLike:
        return self._cp.power(x, p)

    def abs(self, x: ArrayLike) -> ArrayLike:
        return self._cp.abs(x)

    def clip(
        self,
        x: ArrayLike,
        a_min: float | ArrayLike | None,
        a_max: float | ArrayLike | None,
    ) -> ArrayLike:
        return self._cp.clip(x, a_min, a_max)

    def sin(self, x: ArrayLike) -> ArrayLike:
        return self._cp.sin(x)

    def cos(self, x: ArrayLike) -> ArrayLike:
        return self._cp.cos(x)



    def sum(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> ArrayLike:
        return self._cp.sum(x, axis=axis, keepdims=keepdims)

    def mean(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> ArrayLike:
        return self._cp.mean(x, axis=axis, keepdims=keepdims)

    def std(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> ArrayLike:
        return self._cp.std(x, axis=axis, ddof=ddof, keepdims=keepdims)

    def var(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> ArrayLike:
        return self._cp.var(x, axis=axis, ddof=ddof, keepdims=keepdims)

    def min(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> ArrayLike:
        return self._cp.min(x, axis=axis, keepdims=keepdims)

    def max(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> ArrayLike:
        return self._cp.max(x, axis=axis, keepdims=keepdims)

    def argmin(self, x: ArrayLike, axis: int | None = None) -> ArrayLike:
        return self._cp.argmin(x, axis=axis)

    def argmax(self, x: ArrayLike, axis: int | None = None) -> ArrayLike:
        return self._cp.argmax(x, axis=axis)

    def median(self, x: ArrayLike, axis: int | None = None) -> ArrayLike:
        return self._cp.median(x, axis=axis)

    def percentile(
        self,
        x: ArrayLike,
        q: float | ArrayLike,
        axis: int | None = None,
    ) -> ArrayLike:
        return self._cp.percentile(x, q, axis=axis)


    def dot(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        return self._cp.dot(a, b)

    def matmul(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        return self._cp.matmul(a, b)

    def norm(
        self,
        x: ArrayLike,
        ord: int | float | str | None = None,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> ArrayLike:
        return self._cp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)



    def where(
        self,
        condition: ArrayLike,
        x: ArrayLike | float,
        y: ArrayLike | float,
    ) -> ArrayLike:
        return self._cp.where(condition, x, y)

    def logical_and(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        return self._cp.logical_and(x1, x2)

    def logical_or(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        return self._cp.logical_or(x1, x2)

    def logical_not(self, x: ArrayLike) -> ArrayLike:
        return self._cp.logical_not(x)

    def logical_xor(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        return self._cp.logical_xor(x1, x2)

    def all(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
    ) -> ArrayLike:
        return self._cp.all(x, axis=axis)

    def any(
        self,
        x: ArrayLike,
        axis: int | tuple[int, ...] | None = None,
    ) -> ArrayLike:
        return self._cp.any(x, axis=axis)


    def reshape(self, x: ArrayLike, shape: tuple[int, ...]) -> ArrayLike:
        return self._cp.reshape(x, shape)

    def transpose(
        self,
        x: ArrayLike,
        axes: tuple[int, ...] | None = None,
    ) -> ArrayLike:
        return self._cp.transpose(x, axes)

    def concatenate(
        self,
        arrays: Sequence[ArrayLike],
        axis: int = 0,
    ) -> ArrayLike:
        return self._cp.concatenate(arrays, axis=axis)

    def stack(self, arrays: Sequence[ArrayLike], axis: int = 0) -> ArrayLike:
        return self._cp.stack(arrays, axis=axis)

    def vstack(self, arrays: Sequence[ArrayLike]) -> ArrayLike:
        return self._cp.vstack(arrays)

    def hstack(self, arrays: Sequence[ArrayLike]) -> ArrayLike:
        return self._cp.hstack(arrays)

    def split(
        self,
        x: ArrayLike,
        indices_or_sections: int | Sequence[int],
        axis: int = 0,
    ) -> list[ArrayLike]:
        return list(self._cp.split(x, indices_or_sections, axis=axis))

    def unique(
        self,
        x: ArrayLike,
        return_counts: bool = False,
    ) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
        return self._cp.unique(x, return_counts=return_counts)

    def argsort(self, x: ArrayLike, axis: int = -1) -> ArrayLike:
        return self._cp.argsort(x, axis=axis)

    def sort(self, x: ArrayLike, axis: int = -1) -> ArrayLike:
        return self._cp.sort(x, axis=axis)

    def copy(self, x: ArrayLike) -> ArrayLike:
        return self._cp.copy(x)



    def triu_indices(self, n: int, k: int = 0) -> tuple[ArrayLike, ArrayLike]:
        return self._cp.triu_indices(n, k)

    def tril_indices(self, n: int, k: int = 0) -> tuple[ArrayLike, ArrayLike]:
        return self._cp.tril_indices(n, k)



    def to_cpu(self, x: ArrayLike) -> NDArray:
        """Transfer array to CPU (NumPy array)."""
        if isinstance(x, self._cp.ndarray):
            return x.get()  # pyright: ignore[reportAttributeAccessIssue]
        return np.asarray(x)

    def to_device(self, x: ArrayLike) -> ArrayLike:
        """Transfer array to GPU."""
        return self._cp.asarray(x)


    def rint(self, x: ArrayLike) -> ArrayLike:
        return self._cp.rint(x)

    def floor(self, x: ArrayLike) -> ArrayLike:
        return self._cp.floor(x)

    def ceil(self, x: ArrayLike) -> ArrayLike:
        return self._cp.ceil(x)


    def free_memory_pool(self) -> None:
        """Free unused memory from the memory pool."""
        mempool = self._cp.get_default_memory_pool()
        mempool.free_all_blocks()

    def get_memory_info(self) -> dict[str, int]:
        """Get current GPU memory usage."""
        mempool = self._cp.get_default_memory_pool()
        device = self._cp.cuda.Device(self._device_id)
        free, total = device.mem_info
        return {
            "pool_used": mempool.used_bytes(),
            "pool_total": mempool.total_bytes(),
            "device_free": free,
            "device_total": total,
        }

    def synchronize(self) -> None:
        """Synchronize GPU device."""
        self._cp.cuda.Stream.null.synchronize()



    def pairwise_distances(
        self,
        x: ArrayLike,
        y: ArrayLike | None = None,
        metric: str = "euclidean",
    ) -> ArrayLike:
        """Compute pairwise distances on GPU.

        For 'euclidean' metric, uses efficient GPU implementation.
        For other metrics, falls back to CPU (SciPy).
        """
        if y is None:
            y = x

        if metric == "euclidean":
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            x_sq = self.sum(x**2, axis=1, keepdims=True)
            y_sq = self.sum(y**2, axis=1, keepdims=True)  # pyright: ignore[reportOptionalOperand]
            xy = self.matmul(x, self.transpose(y))
            dist_sq = x_sq + self.transpose(y_sq) - 2 * xy
            # Clamp negative values due to numerical errors
            dist_sq = self.clip(dist_sq, 0, None)
            return self.sqrt(dist_sq)

        # Fall back to CPU for other metrics
        x_cpu = self.to_cpu(x)
        y_cpu = self.to_cpu(y)
        result = cdist(x_cpu, y_cpu, metric=metric)  # pyright: ignore[reportCallIssue, reportArgumentType]
        return self.asarray(result)

    def rankdata(self, x: ArrayLike, method: str = "average") -> ArrayLike:
        """Compute ranks on GPU.

        Falls back to CPU for complex ranking methods.
        """
        x_cpu = self.to_cpu(x)
        ranks = scipy_stats.rankdata(x_cpu, method=method)
        return self.asarray(ranks)




@dataclass
class BackendConfig:
    """Configuration for backend selection and behavior."""

    use_gpu: bool = True
    device_id: int = 0
    fallback_to_cpu: bool = True
    min_array_size_for_gpu: int = 1000
    memory_limit: float = 0.8
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 < self.memory_limit <= 1.0:
            msg = f"memory_limit must be in (0, 1], got {self.memory_limit}"
            raise ValueError(msg)
        if self.min_array_size_for_gpu < 0:
            msg = "min_array_size_for_gpu must be non-negative"
            raise ValueError(msg)


class BackendManager:
    """Manages backend selection and provides unified access to array operations.

    This class handles:
    - Automatic backend selection based on GPU availability
    - Graceful fallback to CPU when GPU operations fail
    - Memory monitoring and management
    - Consistent random state across backends

    Example:
        >>> manager = BackendManager()
        >>> backend = manager.get_backend()
        >>> arr = backend.zeros((100, 100))
    """

    _instance: BackendManager | None = None
    _backends: dict[str, ArrayBackend] = {}

    def __init__(self, config: BackendConfig | None = None) -> None:
        """Initialize backend manager.

        Args:
            config: Backend configuration. If None, uses defaults.
        """
        self._config = config or BackendConfig()
        self._current_backend: ArrayBackend | None = None
        self._initialize_backends()

    def _initialize_backends(self) -> None:
        """Initialize available backends."""
        self._backends["numpy"] = NumpyBackend(seed=self._config.seed)

        # Try to initialize CuPy backend if requested
        if self._config.use_gpu and is_gpu_available():
            try:
                self._backends["cupy"] = CupyBackend(
                    seed=self._config.seed,
                    device_id=self._config.device_id,
                )
                logger.info(
                    "CuPy backend initialized on device %d",
                    self._config.device_id,
                )
            except Exception as e:
                logger.warning("Failed to initialize CuPy backend: %s", e)
                if not self._config.fallback_to_cpu:
                    raise

        if "cupy" in self._backends and self._config.use_gpu:
            self._current_backend = self._backends["cupy"]
        else:
            self._current_backend = self._backends["numpy"]

    @classmethod
    def get_instance(cls, config: BackendConfig | None = None) -> BackendManager:
        """Get  instance of BackendManager.

        Args:
            config: Configuration (only used on first call).

        Returns:
            BackendManager  instance.
        """
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance."""
        cls._instance = None
        cls._backends = {}

    def get_backend(self, name: str | None = None) -> ArrayBackend:
        """Get a specific backend or the current default.

        Args:
            name: Backend name ('numpy' or 'cupy'). If None, returns current.

        Returns:
            Requested ArrayBackend instance.

        Raises:
            ValueError: If requested backend is not available.
        """
        if name is None:
            if self._current_backend is None:
                msg = "No backend initialized"
                raise RuntimeError(msg)
            return self._current_backend

        if name not in self._backends:
            available = list(self._backends.keys())
            msg = f"Backend '{name}' not available. Available: {available}"
            raise ValueError(msg)

        return self._backends[name]

    def set_backend(self, name: str) -> None:
        """Set the current default backend.

        Args:
            name: Backend name to use.
        """
        self._current_backend = self.get_backend(name)
        logger.info("Switched to %s backend", name)

    def use_gpu(self) -> bool:
        """Check if GPU backend is active."""
        return (
            self._current_backend is not None
            and self._current_backend.device_type == DeviceType.GPU
        )

    def set_seed(self, seed: int | None) -> None:
        """Set random seed across all backends."""
        self._config.seed = seed
        for backend in self._backends.values():
            backend.set_seed(seed)

    @property
    def config(self) -> BackendConfig:
        """Get current configuration."""
        return self._config

    @property
    def available_backends(self) -> list[str]:
        """List of available backend names."""
        return list(self._backends.keys())

    def should_use_gpu(self, array_size: int) -> bool:
        """Determine if GPU should be used based on array size.

        Args:
            array_size: Total number of elements in the array.

        Returns:
            True if GPU should be used for this operation.
        """
        if not self.use_gpu():
            return False
        return array_size >= self._config.min_array_size_for_gpu

    @contextmanager
    def backend_context(
        self,
        name: str,
    ) -> Generator[ArrayBackend, None, None]:
        """Context manager for temporary backend switching.

        Args:
            name: Backend name to use within context.

        Yields:
            The requested backend.

        Example:
            >>> with manager.backend_context('numpy') as backend:
            ...     arr = backend.zeros((100,))
        """
        previous = self._current_backend
        try:
            self.set_backend(name)
            yield self.get_backend()
        finally:
            if previous is not None:
                self._current_backend = previous

    def get_gpu_info(self) -> GPUInfo:
        """Get current GPU information."""
        return GPUInfo.query()

    def __repr__(self) -> str:
        """String representation."""
        current = self._current_backend.name if self._current_backend else "None"
        available = ", ".join(self.available_backends)
        return f"BackendManager(current={current}, available=[{available}])"




_default_manager: BackendManager | None = None


def initialize_backend(
    use_gpu: bool = True,
    device_id: int = 0,
    seed: int | None = None,
    fallback_to_cpu: bool = True,
) -> BackendManager:
    """Initialize the default backend manager.

    Args:
        use_gpu: Whether to try using GPU.
        device_id: CUDA device ID.
        seed: Random seed.
        fallback_to_cpu: Fall back to CPU if GPU fails.

    Returns:
        Initialized BackendManager.
    """
    global _default_manager  # noqa: PLW0603
    config = BackendConfig(
        use_gpu=use_gpu,
        device_id=device_id,
        seed=seed,
        fallback_to_cpu=fallback_to_cpu,
    )
    _default_manager = BackendManager(config)
    return _default_manager


def get_backend() -> ArrayBackend:
    """Get the current default backend.

    Returns:
        Current ArrayBackend instance.
    """
    global _default_manager  # noqa: PLW0603
    if _default_manager is None:
        _default_manager = BackendManager()
    return _default_manager.get_backend()


def get_backend_manager() -> BackendManager:
    """Get the default backend manager.

    Returns:
        BackendManager singleton.
    """
    global _default_manager  # noqa: PLW0603
    if _default_manager is None:
        _default_manager = BackendManager()
    return _default_manager


def set_seed(seed: int | None) -> None:
    """Set random seed for all backends.

    Args:
        seed: Random seed value.
    """
    manager = get_backend_manager()
    manager.set_seed(seed)


def gpu_available() -> bool:
    """Check if GPU is available.

    Returns:
        True if GPU can be used.
    """
    return is_gpu_available()


def current_device() -> str:
    """Get current device type.

    Returns:
        'cpu' or 'gpu'.
    """
    backend = get_backend()
    return "gpu" if backend.device_type == DeviceType.GPU else "cpu"


@contextmanager
def use_backend(name: str) -> Generator[ArrayBackend, None, None]:
    """Context manager for temporary backend switching.

    Args:
        name: Backend name ('numpy' or 'cupy').

    Yields:
        The requested backend.

    Example:
        >>> with use_backend('numpy') as backend:
        ...     arr = backend.zeros((100,))
    """
    manager = get_backend_manager()
    with manager.backend_context(name) as backend:
        yield backend




def to_numpy(x: ArrayLike) -> NDArray:
    """Convert any array to NumPy array.

    Args:
        x: Input array (NumPy or CuPy).

    Returns:
        NumPy array.
    """
    if hasattr(x, "get"):  # CuPy array
        return x.get()  # pyright: ignore[reportAttributeAccessIssue]
    return np.asarray(x)


def to_backend(x: ArrayLike, backend: ArrayBackend | None = None) -> ArrayLike:
    """Convert array to specified backend.

    Args:
        x: Input array.
        backend: Target backend. If None, uses current default.

    Returns:
        Array on the target backend.
    """
    if backend is None:
        backend = get_backend()
    return backend.to_device(x)


def ensure_cpu(x: ArrayLike) -> NDArray:
    """Ensure array is on CPU for operations requiring NumPy.

    This is useful when calling functions that don't support CuPy,
    such as scikit-learn functions.

    Args:
        x: Input array.

    Returns:
        NumPy array.
    """
    return to_numpy(x)


def ensure_device(x: ArrayLike) -> ArrayLike:
    """Ensure array is on the current default device.

    Args:
        x: Input array.

    Returns:
        Array on current device.
    """
    return to_backend(x)

