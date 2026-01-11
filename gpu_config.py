from __future__ import annotations

import contextlib
import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


@dataclass(frozen=False)
class GPUConfig:
    """Comprehensive GPU configuration for optimization.

    Controls all aspects of GPU usage including device selection,
    memory management, and fallback behavior.

    Attributes:
        use_gpu: Whether to attempt GPU acceleration.
        device_id: CUDA device ID to use.
        memory_limit: Fraction of GPU memory to use (0.0-1.0).
        min_array_size_for_gpu: Minimum array size for GPU transfer.
        fallback_to_cpu: Fall back to CPU on GPU errors.
        use_float32: Use float32 for memory efficiency.
        async_transfers: Enable asynchronous data transfers.
        use_memory_pool: Enable CuPy memory pooling.
        prefer_cuml: Prefer cuML models when available.
        verbose: Print GPU status information.

    Example:
        >>> config = GPUConfig(
        ...     use_gpu=True,
        ...     device_id=0,
        ...     memory_limit=0.8,
        ...     fallback_to_cpu=True,
        ... )
        >>> optimizer = NestedCVOptimizer(gpu_config=config)
    """

    use_gpu: bool = True
    device_id: int = 0
    memory_limit: float = 0.8
    min_array_size_for_gpu: int = 1000
    fallback_to_cpu: bool = True
    use_float32: bool = False
    async_transfers: bool = True
    use_memory_pool: bool = True
    prefer_cuml: bool = True
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 < self.memory_limit <= 1.0:
            msg = f"memory_limit must be in (0, 1], got {self.memory_limit}"
            raise ValueError(msg)

        if self.device_id < 0:
            msg = f"device_id must be non-negative, got {self.device_id}"
            raise ValueError(msg)

        if self.min_array_size_for_gpu < 0:
            msg = f"min_array_size_for_gpu must be non-negative, got {self.min_array_size_for_gpu}"
            raise ValueError(msg)

    @classmethod
    def cpu_only(cls) -> GPUConfig:
        """Create CPU-only configuration.

        Returns:
            GPUConfig configured for CPU-only execution.

        Example:
            >>> config = GPUConfig.cpu_only()
            >>> assert not config.use_gpu
        """
        return cls(use_gpu=False, fallback_to_cpu=False)

    @classmethod
    def auto(cls) -> GPUConfig:
        """Create auto-detecting GPU configuration.

        Automatically detects GPU availability and configures
        appropriate settings.

        Returns:
            GPUConfig with auto-detected settings.

        Example:
            >>> config = GPUConfig.auto()
            >>> if gpu_info.is_available():
            ...     assert config.use_gpu
        """
        use_gpu = gpu_info.is_available()
        return cls(
            use_gpu=use_gpu,
            fallback_to_cpu=True,
            verbose=False,
        )

    @classmethod
    def high_performance(cls, device_id: int = 0) -> GPUConfig:
        """Create high-performance GPU configuration.

        Optimized for maximum throughput with larger memory allocation.

        Args:
            device_id: CUDA device ID to use.

        Returns:
            GPUConfig optimized for performance.

        Example:
            >>> config = GPUConfig.high_performance()
        """
        return cls(
            use_gpu=True,
            device_id=device_id,
            memory_limit=0.9,
            min_array_size_for_gpu=500,
            fallback_to_cpu=True,
            use_float32=False,
            async_transfers=True,
            use_memory_pool=True,
            prefer_cuml=True,
        )

    @classmethod
    def memory_efficient(cls, device_id: int = 0) -> GPUConfig:
        """Create memory-efficient GPU configuration.

        Optimized for limited GPU memory scenarios.

        Args:
            device_id: CUDA device ID to use.

        Returns:
            GPUConfig optimized for memory efficiency.

        Example:
            >>> config = GPUConfig.memory_efficient()
        """
        return cls(
            use_gpu=True,
            device_id=device_id,
            memory_limit=0.6,
            min_array_size_for_gpu=2000,
            fallback_to_cpu=True,
            use_float32=True,
            async_transfers=False,
            use_memory_pool=True,
            prefer_cuml=True,
        )

    def validate_resources(self) -> tuple[bool, str]:
        """Validate that required GPU resources are available.

        Returns:
            Tuple of (success, message).

        Example:
            >>> config = GPUConfig(use_gpu=True)
            >>> valid, msg = config.validate_resources()
            >>> if not valid:
            ...     print(f"GPU validation failed: {msg}")
        """
        if not self.use_gpu:
            return True, "CPU mode - no validation needed"

        if not gpu_info.is_available():
            if self.fallback_to_cpu:
                return True, "GPU unavailable, will fall back to CPU"
            return False, "GPU requested but not available"

        device_count = gpu_info.device_count()
        if self.device_id >= device_count:
            return False, f"Device {self.device_id} not available (only {device_count} devices)"

        memory = gpu_info.memory_info(self.device_id)
        if memory is None:
            return False, f"Could not query memory for device {self.device_id}"

        required_memory = memory["total"] * self.memory_limit
        if memory["free"] < required_memory * 0.5:
            msg = (
                f"Insufficient GPU memory: {memory['free'] / 1e9:.2f}GB free, "
                f"recommended {required_memory / 1e9:.2f}GB"
            )
            if self.fallback_to_cpu:
                return True, msg + " - will fall back to CPU if needed"
            return False, msg

        return True, f"GPU {self.device_id} validated successfully"

    def get_effective_config(self) -> GPUConfig:
        """Get effective configuration after validation.

        Returns configuration with use_gpu set to False if GPU
        is unavailable or validation fails.

        Returns:
            Validated GPUConfig.
        """
        if not self.use_gpu:
            return self

        valid, msg = self.validate_resources()
        if not valid:
            logger.warning("GPU validation failed: %s. Using CPU.", msg)
            return GPUConfig(
                use_gpu=False,
                fallback_to_cpu=False,
                use_float32=self.use_float32,
                prefer_cuml=False,
            )

        if self.verbose:
            logger.info("GPU config: %s", msg)

        return self

    def estimate_memory_usage(
        self,
        n_samples: int,
        n_features: int,
        population_size: int,
        swarm_size: int,
        n_discrete: int = 0,
        n_continuous: int = 0,
    ) -> dict[str, float]:
        """Estimate GPU memory usage for given problem size.

        Args:
            n_samples: Number of data samples.
            n_features: Number of features.
            population_size: GA population size.
            swarm_size: PSO swarm size.
            n_discrete: Number of discrete parameters.
            n_continuous: Number of continuous parameters.

        Returns:
            Dictionary with memory estimates in bytes.

        Example:
            >>> config = GPUConfig()
            >>> mem = config.estimate_memory_usage(
            ...     n_samples=10000, n_features=500,
            ...     population_size=100, swarm_size=50,
            ... )
            >>> print(f"Estimated: {mem['total'] / 1e9:.2f} GB")
        """
        dtype_size = 4 if self.use_float32 else 8

        data_memory = n_samples * n_features * dtype_size * 2

        ga_memory = population_size * (n_features + n_discrete) * dtype_size
        ga_memory += population_size * dtype_size  # fitness

        pso_memory = swarm_size * n_continuous * dtype_size * 4
        pso_memory += n_continuous * dtype_size  # global best

        # Temporary allocations
        temp_memory = max(
            population_size * population_size * dtype_size,  # diversity matrix
            n_samples * dtype_size * 10,  # CV predictions
        )

        total = data_memory + ga_memory + pso_memory + temp_memory

        return {
            "data": float(data_memory),
            "ga_population": float(ga_memory),
            "pso_swarm": float(pso_memory),
            "temporary": float(temp_memory),
            "total": float(total),
            "total_gb": total / 1e9,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of config.
        """
        return {
            "use_gpu": self.use_gpu,
            "device_id": self.device_id,
            "memory_limit": self.memory_limit,
            "min_array_size_for_gpu": self.min_array_size_for_gpu,
            "fallback_to_cpu": self.fallback_to_cpu,
            "use_float32": self.use_float32,
            "async_transfers": self.async_transfers,
            "use_memory_pool": self.use_memory_pool,
            "prefer_cuml": self.prefer_cuml,
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GPUConfig:
        """Create configuration from dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            GPUConfig instance.
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class _GPUInfo:
    """GPU information and status query interface.

    Singleton class providing GPU status queries. Access via the
    `gpu_info` module-level instance.

    Example:
        >>> from gpu_config import gpu_info
        >>>
        >>> if gpu_info.is_available():
        ...     print(f"GPUs: {gpu_info.device_count()}")
        ...     print(f"Memory: {gpu_info.memory_info()}")
        ...     print(f"Backend: {gpu_info.current_backend()}")
    """

    _instance: _GPUInfo | None = None
    _cached_available: bool | None = None
    _cached_device_count: int | None = None

    def __new__(cls) -> _GPUInfo:
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def is_available(self) -> bool:
        """Check if GPU (CuPy/CUDA) is available.

        Results are cached for performance.

        Returns:
            True if GPU is available and functional.

        Example:
            >>> if gpu_info.is_available():
            ...     print("GPU ready!")
        """
        if self._cached_available is not None:
            return self._cached_available

        try:
            import cupy as cp

            arr = cp.array([1, 2, 3])
            _ = arr.sum()
            self._cached_available = True
        except (ImportError, Exception):
            self._cached_available = False

        return self._cached_available

    def device_count(self) -> int:
        """Get number of available CUDA devices.

        Returns:
            Number of CUDA devices, 0 if none available.

        Example:
            >>> n_gpus = gpu_info.device_count()
            >>> print(f"Found {n_gpus} GPU(s)")
        """
        if self._cached_device_count is not None:
            return self._cached_device_count

        if not self.is_available():
            self._cached_device_count = 0
            return 0

        try:
            import cupy as cp

            self._cached_device_count = cp.cuda.runtime.getDeviceCount()
        except Exception:
            self._cached_device_count = 0

        return self._cached_device_count  # pyright: ignore[reportReturnType]

    def memory_info(self, device_id: int = 0) -> dict[str, Any] | None:
        """Get GPU memory information.

        Args:
            device_id: CUDA device ID to query.

        Returns:
            Dictionary with 'free', 'total', 'used' bytes, or None if unavailable.

        Example:
            >>> mem = gpu_info.memory_info()
            >>> if mem:
            ...     print(f"Free: {mem['free'] / 1e9:.2f} GB")
            ...     print(f"Total: {mem['total'] / 1e9:.2f} GB")
        """
        if not self.is_available():
            return None

        try:
            import cupy as cp

            with cp.cuda.Device(device_id):
                free, total = cp.cuda.runtime.memGetInfo()
                return {
                    "free": free,
                    "total": total,
                    "used": total - free,
                    "free_gb": free / 1e9,
                    "total_gb": total / 1e9,
                    "used_gb": (total - free) / 1e9,
                    "utilization": (total - free) / total,
                }
        except Exception as e:
            logger.debug("Could not get memory info: %s", e)
            return None

    def current_backend(self) -> str:
        """Get the current array backend name.

        Returns:
            'cupy' if GPU available and CuPy loaded, else 'numpy'.

        Example:
            >>> backend = gpu_info.current_backend()
            >>> print(f"Using {backend} backend")
        """
        if self.is_available():
            try:
                import cupy  # noqa: F401

                return "cupy"
            except ImportError:
                pass
        return "numpy"

    def device_name(self, device_id: int = 0) -> str | None:
        """Get GPU device name.

        Args:
            device_id: CUDA device ID to query.

        Returns:
            Device name string or None if unavailable.

        Example:
            >>> name = gpu_info.device_name()
            >>> if name:
            ...     print(f"GPU: {name}")
        """
        if not self.is_available():
            return None

        try:
            import cupy as cp

            with cp.cuda.Device(device_id):
                props = cp.cuda.runtime.getDeviceProperties(device_id)
                return props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        except Exception:
            return None

    def device_properties(self, device_id: int = 0) -> dict[str, Any] | None:
        """Get detailed GPU device properties.

        Args:
            device_id: CUDA device ID to query.

        Returns:
            Dictionary with device properties or None if unavailable.

        Example:
            >>> props = gpu_info.device_properties()
            >>> if props:
            ...     print(f"Compute capability: {props['compute_capability']}")
        """
        if not self.is_available():
            return None

        try:
            import cupy as cp

            with cp.cuda.Device(device_id):
                props = cp.cuda.runtime.getDeviceProperties(device_id)
                return {
                    "name": props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
                    "compute_capability": f"{props['major']}.{props['minor']}",
                    "total_memory": props["totalGlobalMem"],
                    "total_memory_gb": props["totalGlobalMem"] / 1e9,
                    "multiprocessors": props["multiProcessorCount"],
                    "max_threads_per_block": props["maxThreadsPerBlock"],
                    "warp_size": props["warpSize"],
                }
        except Exception as e:
            logger.debug("Could not get device properties: %s", e)
            return None

    def cuda_version(self) -> str | None:
        """Get CUDA runtime version.

        Returns:
            CUDA version string or None if unavailable.

        Example:
            >>> version = gpu_info.cuda_version()
            >>> if version:
            ...     print(f"CUDA {version}")
        """
        if not self.is_available():
            return None

        try:
            import cupy as cp

            version = cp.cuda.runtime.runtimeGetVersion()
            major = version // 1000
            minor = (version % 1000) // 10
            return f"{major}.{minor}"
        except Exception:
            return None

    def cupy_version(self) -> str | None:
        """Get CuPy version.

        Returns:
            CuPy version string or None if unavailable.
        """
        try:
            import cupy

            return cupy.__version__
        except ImportError:
            return None

    def summary(self) -> str:
        """Get a summary of GPU status.

        Returns:
            Multi-line string with GPU status summary.

        Example:
            >>> print(gpu_info.summary())
        """
        lines = ["GPU Status Summary", "=" * 40]

        if not self.is_available():
            lines.append("GPU: Not available")
            lines.append("Backend: numpy (CPU)")
            return "\n".join(lines)

        lines.append(f"GPU: Available ({self.device_count()} device(s))")
        lines.append(f"Backend: {self.current_backend()}")

        cuda_ver = self.cuda_version()
        if cuda_ver:
            lines.append(f"CUDA Version: {cuda_ver}")

        cupy_ver = self.cupy_version()
        if cupy_ver:
            lines.append(f"CuPy Version: {cupy_ver}")

        for dev_id in range(self.device_count()):
            lines.append(f"\nDevice {dev_id}:")
            name = self.device_name(dev_id)
            if name:
                lines.append(f"  Name: {name}")

            mem = self.memory_info(dev_id)
            if mem:
                lines.append(f"  Memory: {mem['used_gb']:.2f} / {mem['total_gb']:.2f} GB")
                lines.append(f"  Utilization: {mem['utilization']:.1%}")

            props = self.device_properties(dev_id)
            if props:
                lines.append(f"  Compute: {props['compute_capability']}")
                lines.append(f"  SMs: {props['multiprocessors']}")

        return "\n".join(lines)

    def clear_cache(self) -> None:
        """Clear cached GPU information.

        Call this if GPU state may have changed.
        """
        self._cached_available = None
        self._cached_device_count = None


gpu_info = _GPUInfo()




@contextlib.contextmanager
def gpu_context(
    device: int = 0,
    memory_limit: float = 0.8,
    use_float32: bool = False,
) -> Generator[GPUConfig, None, None]:
    """Context manager for temporary GPU configuration.

    Provides a context where GPU settings are temporarily applied.
    Settings are restored after the context exits.

    Args:
        device: CUDA device ID to use.
        memory_limit: Fraction of GPU memory to limit to.
        use_float32: Use float32 for memory efficiency.

    Yields:
        GPUConfig for the context.

    Example:
        >>> with gpu_context(device=0, memory_limit=0.8) as config:
        ...     result = optimize(X, y, gpu_config=config)
    """
    if not gpu_info.is_available():
        logger.warning("GPU not available, context will use CPU")
        yield GPUConfig.cpu_only()
        return

    config = GPUConfig(
        use_gpu=True,
        device_id=device,
        memory_limit=memory_limit,
        use_float32=use_float32,
        fallback_to_cpu=True,
    )

    try:
        import cupy as cp

        original_device = cp.cuda.Device().id
        cp.cuda.Device(device).use()

        if memory_limit < 1.0:
            mempool = cp.get_default_memory_pool()
            mem_info = gpu_info.memory_info(device)
            if mem_info:
                limit_bytes = int(mem_info["total"] * memory_limit)
                mempool.set_limit(size=limit_bytes)

        yield config

    finally:
        try:
            cp.cuda.Device(original_device).use()
        except Exception:
            pass


@contextlib.contextmanager
def cpu_context() -> Generator[GPUConfig, None, None]:
    """Context manager for CPU-only execution.

    Forces CPU execution regardless of GPU availability.

    Yields:
        CPU-only GPUConfig.

    Example:
        >>> with cpu_context():
        ...     result = optimize(X, y)  # Always uses CPU
    """
    yield GPUConfig.cpu_only()


def estimate_problem_memory(
    n_samples: int,
    n_features: int,
    population_size: int = 100,
    swarm_size: int = 50,
    n_discrete: int = 5,
    n_continuous: int = 5,
    use_float32: bool = False,
) -> dict[str, Any]:
    """Estimate memory requirements for an optimization problem.

    Helper function to determine if a problem will fit in GPU memory.

    Args:
        n_samples: Number of data samples.
        n_features: Number of features.
        population_size: GA population size.
        swarm_size: PSO swarm size.
        n_discrete: Number of discrete parameters.
        n_continuous: Number of continuous parameters.
        use_float32: Use float32 for calculations.

    Returns:
        Dictionary with memory estimates and recommendations.

    Example:
        >>> mem = estimate_problem_memory(
        ...     n_samples=50000, n_features=1000,
        ...     population_size=200, swarm_size=100,
        ... )
        >>> print(f"Estimated memory: {mem['total_gb']:.2f} GB")
        >>> print(f"Recommendation: {mem['recommendation']}")
    """
    config = GPUConfig(use_float32=use_float32)
    estimates = config.estimate_memory_usage(
        n_samples=n_samples,
        n_features=n_features,
        population_size=population_size,
        swarm_size=swarm_size,
        n_discrete=n_discrete,
        n_continuous=n_continuous,
    )

    total_gb = estimates["total_gb"]
    available_memory = None
    recommendation = "CPU recommended"

    if gpu_info.is_available():
        mem_info = gpu_info.memory_info()
        if mem_info:
            available_memory = mem_info["free_gb"]
            if total_gb < available_memory * 0.7:
                recommendation = "GPU recommended"
            elif total_gb < available_memory * 0.9:
                recommendation = "GPU possible with memory_efficient config"
            else:
                recommendation = "CPU recommended (insufficient GPU memory)"

    estimates["available_gpu_memory_gb"] = available_memory  # pyright: ignore[reportArgumentType]
    estimates["recommendation"] = recommendation  # pyright: ignore[reportArgumentType]
    estimates["fits_in_gpu"] = (
        available_memory is not None and total_gb < available_memory * 0.9
    )

    return estimates


def get_optimal_batch_size(
    n_features: int,
    available_memory_gb: float | None = None,
    use_float32: bool = False,
) -> int:
    """Calculate optimal batch size for GPU operations.

    Args:
        n_features: Number of features per sample.
        available_memory_gb: Available GPU memory in GB.
        use_float32: Use float32 calculations.

    Returns:
        Recommended batch size.

    Example:
        >>> batch_size = get_optimal_batch_size(n_features=1000)
        >>> print(f"Optimal batch size: {batch_size}")
    """
    dtype_size = 4 if use_float32 else 8

    if available_memory_gb is None:
        if gpu_info.is_available():
            mem = gpu_info.memory_info()
            if mem:
                available_memory_gb = mem["free_gb"] * 0.5
            else:
                available_memory_gb = 2.0 
        else:
            available_memory_gb = 4.0

    # Memory per sample (features + overhead)
    memory_per_sample = n_features * dtype_size * 3  # Original + transformed + temp

    # Calculate batch size with 20% safety margin
    max_samples = int((available_memory_gb * 1e9 * 0.8) / memory_per_sample)  # pyright: ignore[reportOptionalOperand]

    # Clamp to reasonable range
    return max(32, min(max_samples, 10000))

def validate_gpu_setup() -> dict[str, Any]:
    """Validate complete GPU setup and return status.

    Performs comprehensive GPU validation including driver,
    CUDA runtime, CuPy, and memory checks.

    Returns:
        Dictionary with validation results.

    Example:
        >>> status = validate_gpu_setup()
        >>> if status["valid"]:
        ...     print("GPU setup valid!")
        >>> else:
        ...     for issue in status["issues"]:
        ...         print(f"Issue: {issue}")
    """
    results: dict[str, Any] = {
        "valid": True,
        "issues": [],
        "warnings": [],
        "info": {},
    }

    try:
        import cupy as cp

        results["info"]["cupy_version"] = cp.__version__
    except ImportError:
        results["valid"] = False
        results["issues"].append("CuPy not installed")
        return results

    if not gpu_info.is_available():
        results["valid"] = False
        results["issues"].append("CUDA not available or not functional")
        return results

    device_count = gpu_info.device_count()
    results["info"]["device_count"] = device_count
    if device_count == 0:
        results["valid"] = False
        results["issues"].append("No CUDA devices found")
        return results

    for dev_id in range(device_count):
        props = gpu_info.device_properties(dev_id)
        mem = gpu_info.memory_info(dev_id)

        if props:
            results["info"][f"device_{dev_id}_name"] = props["name"]
            results["info"][f"device_{dev_id}_compute"] = props["compute_capability"]

            # Check compute capability
            major = int(props["compute_capability"].split(".")[0])
            if major < 5:
                results["warnings"].append(
                    f"Device {dev_id} has old compute capability {props['compute_capability']}"
                )

        if mem:
            results["info"][f"device_{dev_id}_memory_gb"] = mem["total_gb"]
            if mem["free_gb"] < 1.0:
                results["warnings"].append(
                    f"Device {dev_id} has low free memory: {mem['free_gb']:.2f} GB"
                )

    cuda_ver = gpu_info.cuda_version()
    if cuda_ver:
        results["info"]["cuda_version"] = cuda_ver
        major = int(cuda_ver.split(".")[0])
        if major < 11:
            results["warnings"].append(f"CUDA version {cuda_ver} is old, consider upgrading")

    return results




__all__ = [
    "GPUConfig",
    "gpu_info",
    "gpu_context",
    "cpu_context",
    "estimate_problem_memory",
    "get_optimal_batch_size",
    "validate_gpu_setup",
]