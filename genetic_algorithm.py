from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

from .utils.backend import (
    ArrayBackend,
    ArrayLike,
    get_backend,
    is_gpu_available,
    to_numpy,
)
from .utils.data import DeviceInfo
from .utils.performance import GPUKernels, create_gpu_kernels, get_stream_manager

if TYPE_CHECKING:
    from collections.abc import Sequence

# Configure logging
logger = logging.getLogger(__name__)

_gpu_kernels: GPUKernels | None = None


def get_gpu_kernels() -> GPUKernels:
    """Get or create global GPUKernels instance."""
    global _gpu_kernels
    if _gpu_kernels is None:
        _gpu_kernels = create_gpu_kernels()
    return _gpu_kernels


@dataclass
class GPUPopulation:
    """GPU-accelerated population representation for Genetic Algorithm.

    Stores all population data as contiguous arrays for efficient GPU operations.
    All genetic operations (selection, crossover, mutation) operate on these arrays
    in a vectorized manner.

    Attributes:
        feature_genes: Boolean array of shape (pop_size, n_features).
                      True indicates feature is selected.
        discrete_genes: Integer array of shape (pop_size, n_discrete_params).
                       Stores indices into discrete choice arrays.
        fitness: Float array of shape (pop_size,).
                Contains fitness values (-inf for unevaluated).
        needs_evaluation: Boolean array of shape (pop_size,).
                         True if individual needs fitness evaluation.
        generation: Integer array of shape (pop_size,).
                   Generation when each individual was created.
        device_info: Information about storage device (CPU/GPU).
        individual_ids: List of unique IDs for each individual (CPU).
        metadata: List of metadata dicts for each individual (CPU, sparse).
    """

    feature_genes: ArrayLike
    discrete_genes: ArrayLike
    fitness: ArrayLike
    needs_evaluation: ArrayLike
    generation: ArrayLike
    device_info: DeviceInfo = field(default_factory=DeviceInfo.cpu)
    individual_ids: list[str] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate population data and initialize IDs if needed."""
        # Validate dimensions
        if self.feature_genes.ndim != 2:
            msg = f"feature_genes must be 2D, got {self.feature_genes.ndim}D"
            raise ValueError(msg)

        if self.discrete_genes.ndim != 2:
            msg = f"discrete_genes must be 2D, got {self.discrete_genes.ndim}D"
            raise ValueError(msg)

        pop_size = self.feature_genes.shape[0]

        if self.discrete_genes.shape[0] != pop_size:
            msg = "feature_genes and discrete_genes must have same population size"
            raise ValueError(msg)

        if len(self.fitness) != pop_size:
            msg = f"fitness length ({len(self.fitness)}) must match pop_size ({pop_size})"
            raise ValueError(msg)

        if len(self.needs_evaluation) != pop_size:
            msg = "needs_evaluation length must match pop_size"
            raise ValueError(msg)

        if len(self.generation) != pop_size:
            msg = "generation length must match pop_size"
            raise ValueError(msg)

        if not self.individual_ids:
            self.individual_ids = [self._generate_id() for _ in range(pop_size)]

        if not self.metadata:
            self.metadata = [{} for _ in range(pop_size)]

    @staticmethod
    def _generate_id() -> str:
        """Generate a unique individual ID."""
        return str(uuid.uuid4())[:8]

    @property
    def pop_size(self) -> int:
        """Population size."""
        return self.feature_genes.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.feature_genes.shape[1]

    @property
    def n_discrete(self) -> int:
        """Number of discrete parameters."""
        return self.discrete_genes.shape[1]

    @property
    def is_gpu(self) -> bool:
        """Check if population is on GPU."""
        return self.device_info.is_gpu

    @property
    def is_cpu(self) -> bool:
        """Check if population is on CPU."""
        return self.device_info.is_cpu

    def get_backend(self) -> ArrayBackend:
        """Get the appropriate backend for this population."""
        return get_backend()

    def to_cpu(self) -> GPUPopulation:
        """Transfer all array data to CPU.

        Returns:
            New GPUPopulation with all arrays on CPU.
        """
        if self.is_cpu:
            return self

        return GPUPopulation(
            feature_genes=to_numpy(self.feature_genes),
            discrete_genes=to_numpy(self.discrete_genes),
            fitness=to_numpy(self.fitness),
            needs_evaluation=to_numpy(self.needs_evaluation),
            generation=to_numpy(self.generation),
            device_info=DeviceInfo.cpu(),
            individual_ids=list(self.individual_ids),
            metadata=[dict(m) for m in self.metadata],
        )

    def to_gpu(self, device_id: int = 0) -> GPUPopulation:
        """Transfer all array data to GPU.

        Args:
            device_id: CUDA device ID.

        Returns:
            New GPUPopulation with all arrays on GPU.

        Raises:
            RuntimeError: If GPU is not available.
        """
        if not is_gpu_available():
            msg = "GPU not available"
            raise RuntimeError(msg)

        if self.is_gpu:
            return self

        backend = get_backend()
        return GPUPopulation(
            feature_genes=backend.asarray(self.feature_genes),
            discrete_genes=backend.asarray(self.discrete_genes),
            fitness=backend.asarray(self.fitness),
            needs_evaluation=backend.asarray(self.needs_evaluation),
            generation=backend.asarray(self.generation),
            device_info=DeviceInfo.gpu(device_id),
            individual_ids=list(self.individual_ids),
            metadata=[dict(m) for m in self.metadata],
        )

    def to_gpu_async(self, device_id: int = 0) -> GPUPopulation:
        """Transfer all array data to GPU asynchronously using streams.

        Uses multiple CUDA streams for concurrent data transfer.

        Args:
            device_id: CUDA device ID.

        Returns:
            New GPUPopulation with all arrays on GPU.
        """
        if not is_gpu_available():
            msg = "GPU not available"
            raise RuntimeError(msg)

        if self.is_gpu:
            return self

        try:
            import cupy as cp

            stream_manager = get_stream_manager()

            # Use multiple streams for concurrent transfer
            with stream_manager.stream_context(0) as s0:
                feature_genes_gpu = cp.asarray(self.feature_genes)

            with stream_manager.stream_context(1) as s1:
                discrete_genes_gpu = cp.asarray(self.discrete_genes)

            with stream_manager.stream_context(2) as s2:
                fitness_gpu = cp.asarray(self.fitness)
                needs_eval_gpu = cp.asarray(self.needs_evaluation)
                generation_gpu = cp.asarray(self.generation)

            # Synchronize all streams
            stream_manager.synchronize_all()

            return GPUPopulation(
                feature_genes=feature_genes_gpu,
                discrete_genes=discrete_genes_gpu,
                fitness=fitness_gpu,
                needs_evaluation=needs_eval_gpu,
                generation=generation_gpu,
                device_info=DeviceInfo.gpu(device_id),
                individual_ids=list(self.individual_ids),
                metadata=[dict(m) for m in self.metadata],
            )

        except Exception as e:
            logger.warning("Async GPU transfer failed, using sync: %s", e)
            return self.to_gpu(device_id)

    def to_device(self, device_info: DeviceInfo) -> GPUPopulation:
        """Transfer to specified device.

        Args:
            device_info: Target device information.

        Returns:
            GPUPopulation on the target device.
        """
        if device_info.is_gpu:
            return self.to_gpu(device_info.device_id)
        return self.to_cpu()

    def get_unevaluated_indices(self) -> ArrayLike:
        """Get indices of individuals needing evaluation.

        Returns:
            Array of indices where needs_evaluation is True.
        """
        if self.is_gpu:
            backend = self.get_backend()
            return backend.xp.where(self.needs_evaluation)[0]
        return np.where(self.needs_evaluation)[0]

    def get_unevaluated_count(self) -> int:
        """Get count of individuals needing evaluation."""
        if self.is_gpu:
            backend = self.get_backend()
            count = backend.sum(self.needs_evaluation.astype(np.int32))
            return int(backend.to_scalar(count))
        return int(np.sum(self.needs_evaluation))

    def get_evaluated_mask(self) -> ArrayLike:
        """Get boolean mask of evaluated individuals."""
        if self.is_gpu:
            backend = self.get_backend()
            return backend.logical_not(self.needs_evaluation)
        return np.logical_not(self.needs_evaluation)

    def get_best_index(self) -> int:
        """Get index of the best individual (highest fitness).

        Returns:
            Index of individual with highest fitness.
        """
        evaluated_mask = self.get_evaluated_mask()

        if self.is_gpu:
            backend = self.get_backend()
            # Create masked fitness (unevaluated get -inf)
            masked_fitness = backend.where(
                evaluated_mask, self.fitness, backend.full(self.fitness.shape, float("-inf"))
            )
            best_idx = backend.argmax(masked_fitness)
            return int(backend.to_scalar(best_idx))

        masked_fitness = np.where(evaluated_mask, self.fitness, -np.inf)
        return int(np.argmax(masked_fitness))

    def get_best_fitness(self) -> float:
        """Get the best fitness value."""
        best_idx = self.get_best_index()
        if self.is_gpu:
            backend = self.get_backend()
            return float(backend.to_scalar(self.fitness[best_idx]))
        return float(self.fitness[best_idx])

    def get_fitness_statistics(self) -> dict[str, float]:
        """Compute statistics for evaluated individuals.

        All computations are performed on GPU when available,
        only transferring final scalar results to CPU.

        Returns:
            Dictionary with min, max, mean, std, median statistics.
        """
        evaluated_mask = self.get_evaluated_mask()

        if self.is_gpu:
            kernels = get_gpu_kernels()
            return kernels.compute_fitness_statistics_gpu(self.fitness, evaluated_mask)  # pyright: ignore[reportAttributeAccessIssue]

        mask_cpu = np.asarray(evaluated_mask)
        fitness_cpu = np.asarray(self.fitness)
        valid_fitness = fitness_cpu[mask_cpu]

        if len(valid_fitness) == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "n_evaluated": 0,
            }

        return {
            "min": float(np.min(valid_fitness)),
            "max": float(np.max(valid_fitness)),
            "mean": float(np.mean(valid_fitness)),
            "std": float(np.std(valid_fitness)),
            "median": float(np.median(valid_fitness)),
            "n_evaluated": int(np.sum(mask_cpu)),
        }

    def get_feature_selection_frequency(self) -> NDArray:
        """Compute how often each feature is selected across population.

        Computation is performed on GPU when available, transferring
        only the final frequency array to CPU.

        Returns:
            Array of shape (n_features,) with selection frequencies.
        """
        if self.is_gpu:
            backend = self.get_backend()
            xp = backend.xp
            # Compute entirely on GPU
            counts = xp.sum(self.feature_genes.astype(xp.float64), axis=0)
            frequencies = counts / self.pop_size
            # Single transfer of result
            return to_numpy(frequencies)

        # CPU path
        counts = np.sum(self.feature_genes.astype(np.float64), axis=0)
        return counts / self.pop_size

    def get_diversity_metrics(self) -> dict[str, float]:
        """Compute population diversity metrics on GPU.

        Returns:
            Dictionary with diversity metrics.
        """
        if self.is_gpu:
            backend = self.get_backend()
            xp = backend.xp

            # Feature diversity: average pairwise Hamming distance
            features_float = self.feature_genes.astype(xp.float64)

            # Compute mean feature selection rate per feature
            mean_selection = xp.mean(features_float, axis=0)

            # Diversity as entropy-like measure
            # For each feature: -p*log(p) - (1-p)*log(1-p)
            eps = 1e-10
            p = xp.clip(mean_selection, eps, 1 - eps)
            entropy = -p * xp.log2(p) - (1 - p) * xp.log2(1 - p)
            mean_entropy = float(xp.mean(entropy))

            # Fitness diversity
            evaluated_mask = self.get_evaluated_mask()
            valid_fitness = self.fitness[evaluated_mask]
            if len(valid_fitness) > 1:
                fitness_std = float(xp.std(valid_fitness))
                fitness_range = float(xp.max(valid_fitness) - xp.min(valid_fitness))
            else:
                fitness_std = 0.0
                fitness_range = 0.0

            return {
                "feature_entropy": mean_entropy,
                "fitness_std": fitness_std,
                "fitness_range": fitness_range,
                "unique_ratio": self._compute_unique_ratio_gpu(),
            }

        # CPU path
        features_float = self.feature_genes.astype(np.float64)
        mean_selection = np.mean(features_float, axis=0)

        eps = 1e-10
        p = np.clip(mean_selection, eps, 1 - eps)
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        mean_entropy = float(np.mean(entropy))

        evaluated_mask = self.get_evaluated_mask()
        valid_fitness = self.fitness[evaluated_mask]
        if len(valid_fitness) > 1:
            fitness_std = float(np.std(valid_fitness))
            fitness_range = float(np.max(valid_fitness) - np.min(valid_fitness))
        else:
            fitness_std = 0.0
            fitness_range = 0.0

        return {
            "feature_entropy": mean_entropy,
            "fitness_std": fitness_std,
            "fitness_range": fitness_range,
            "unique_ratio": self._compute_unique_ratio_cpu(),
        }

    def _compute_unique_ratio_gpu(self) -> float:
        """Compute ratio of unique individuals (GPU)."""
        backend = self.get_backend()
        xp = backend.xp

        # Convert features to hashable representation
        # Use sum of weighted features as a simple hash
        weights = xp.arange(self.n_features, dtype=xp.float64)
        hashes = xp.dot(self.feature_genes.astype(xp.float64), weights)

        unique_count = len(xp.unique(hashes))
        return unique_count / self.pop_size

    def _compute_unique_ratio_cpu(self) -> float:
        """Compute ratio of unique individuals (CPU)."""
        unique_individuals = set()
        for i in range(self.pop_size):
            key = (
                tuple(self.feature_genes[i].astype(bool)),
                tuple(self.discrete_genes[i]),
            )
            unique_individuals.add(key)
        return len(unique_individuals) / self.pop_size

    def copy(self) -> GPUPopulation:
        """Create a deep copy of the population."""
        if self.is_gpu:
            backend = self.get_backend()
            return GPUPopulation(
                feature_genes=backend.copy(self.feature_genes),
                discrete_genes=backend.copy(self.discrete_genes),
                fitness=backend.copy(self.fitness),
                needs_evaluation=backend.copy(self.needs_evaluation),
                generation=backend.copy(self.generation),
                device_info=self.device_info,
                individual_ids=list(self.individual_ids),
                metadata=[dict(m) for m in self.metadata],
            )

        return GPUPopulation(
            feature_genes=np.copy(self.feature_genes),
            discrete_genes=np.copy(self.discrete_genes),
            fitness=np.copy(self.fitness),
            needs_evaluation=np.copy(self.needs_evaluation),
            generation=np.copy(self.generation),
            device_info=self.device_info,
            individual_ids=list(self.individual_ids),
            metadata=[dict(m) for m in self.metadata],
        )

    def get_individual(self, index: int) -> dict[str, Any]:
        """Get a single individual as a dictionary.

        Args:
            index: Index of the individual.

        Returns:
            Dictionary with individual's genes, fitness, etc.
        """
        if self.is_gpu:
            backend = self.get_backend()
            return {
                "feature_genes": to_numpy(self.feature_genes[index]),
                "discrete_genes": to_numpy(self.discrete_genes[index]),
                "fitness": float(backend.to_scalar(self.fitness[index])),
                "needs_evaluation": bool(backend.to_scalar(self.needs_evaluation[index])),
                "generation": int(backend.to_scalar(self.generation[index])),
                "id": self.individual_ids[index],
                "metadata": self.metadata[index],
            }

        return {
            "feature_genes": np.asarray(self.feature_genes[index]),
            "discrete_genes": np.asarray(self.discrete_genes[index]),
            "fitness": float(self.fitness[index]),
            "needs_evaluation": bool(self.needs_evaluation[index]),
            "generation": int(self.generation[index]),
            "id": self.individual_ids[index],
            "metadata": self.metadata[index],
        }

    def set_fitness(self, index: int | ArrayLike, fitness: float | ArrayLike) -> None:
        """Set fitness for individual(s) and mark as evaluated.

        Args:
            index: Single index or array of indices.
            fitness: Single fitness or array of fitness values.
        """
        if isinstance(index, int):
            self.fitness[index] = fitness
            self.needs_evaluation[index] = False
        else:
            # Batch update
            if self.is_gpu:
                backend = self.get_backend()
                index = backend.asarray(index)
                fitness_arr = backend.asarray(fitness)
            else:
                index = np.asarray(index)
                fitness_arr = np.asarray(fitness)

            self.fitness[index] = fitness_arr
            self.needs_evaluation[index] = False

    def mark_for_evaluation(self, indices: ArrayLike | None = None) -> None:
        """Mark individuals for evaluation.

        Args:
            indices: Indices to mark. If None, marks all.
        """
        if indices is None:
            if self.is_gpu:
                backend = self.get_backend()
                self.needs_evaluation = backend.ones(self.pop_size, dtype="bool")
            else:
                self.needs_evaluation = np.ones(self.pop_size, dtype=bool)
        else:
            if self.is_gpu:
                backend = self.get_backend()
                indices = backend.asarray(indices)
            self.needs_evaluation[indices] = True

    def __len__(self) -> int:
        """Return population size."""
        return self.pop_size

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_fitness_statistics()
        return (
            f"GPUPopulation(pop_size={self.pop_size}, "
            f"n_features={self.n_features}, "
            f"n_discrete={self.n_discrete}, "
            f"n_evaluated={stats['n_evaluated']}, "
            f"device={self.device_info.backend_name})"
        )


class PopulationInitializer(ABC):
    """Abstract base class for population initialization strategies."""

    @abstractmethod
    def initialize(
        self,
        pop_size: int,
        n_features: int,
        n_discrete: int,
        discrete_bounds: Sequence[tuple[int, int]],
        use_gpu: bool = False,
        dtype: str = "float64",
    ) -> GPUPopulation:
        """Initialize a population.

        Args:
            pop_size: Number of individuals.
            n_features: Number of features.
            n_discrete: Number of discrete parameters.
            discrete_bounds: List of (min, max) bounds for each discrete param.
            use_gpu: Whether to create on GPU.
            dtype: Data type for fitness array.

        Returns:
            Initialized GPUPopulation.
        """


class RandomInitializer(PopulationInitializer):
    """Random population initialization.

    Initializes feature genes with uniform probability and discrete genes
    uniformly within their bounds.
    """

    def __init__(
        self,
        feature_init_prob: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """Initialize random initializer.

        Args:
            feature_init_prob: Probability of selecting each feature.
            seed: Random seed for reproducibility.
        """
        if not 0.0 <= feature_init_prob <= 1.0:
            msg = f"feature_init_prob must be in [0, 1], got {feature_init_prob}"
            raise ValueError(msg)

        self.feature_init_prob = feature_init_prob
        self.seed = seed

    def initialize(
        self,
        pop_size: int,
        n_features: int,
        n_discrete: int,
        discrete_bounds: Sequence[tuple[int, int]],
        use_gpu: bool = False,
        dtype: str = "float64",
    ) -> GPUPopulation:
        """Initialize population with random genes.

        Args:
            pop_size: Number of individuals.
            n_features: Number of features.
            n_discrete: Number of discrete parameters.
            discrete_bounds: List of (min, max) bounds for each discrete param.
            use_gpu: Whether to create on GPU.
            dtype: Data type for fitness array.

        Returns:
            Randomly initialized GPUPopulation.
        """
        if len(discrete_bounds) != n_discrete:
            msg = f"discrete_bounds length ({len(discrete_bounds)}) must match n_discrete ({n_discrete})"
            raise ValueError(msg)

        backend = get_backend()

        if self.seed is not None:
            backend.set_seed(self.seed)

        if use_gpu and is_gpu_available():
            feature_genes = backend.random_uniform(0, 1, (pop_size, n_features)) < self.feature_init_prob

            if n_discrete > 0:
                discrete_genes = backend.zeros((pop_size, n_discrete), dtype="int64")
                for i, (low, high) in enumerate(discrete_bounds):
                    discrete_genes[:, i] = backend.random_integers(low, high, (pop_size,))
            else:
                discrete_genes = backend.zeros((pop_size, 0), dtype="int64")

            fitness = backend.full(pop_size, float("-inf"), dtype=dtype)  # pyright: ignore[reportArgumentType]
            needs_evaluation = backend.ones(pop_size, dtype="bool")
            generation = backend.zeros(pop_size, dtype="int32")

            device_info = DeviceInfo.gpu()
        else:
            rng = np.random.default_rng(self.seed)

            feature_genes = rng.random((pop_size, n_features)) < self.feature_init_prob

            if n_discrete > 0:
                discrete_genes = np.zeros((pop_size, n_discrete), dtype=np.int64)
                for i, (low, high) in enumerate(discrete_bounds):
                    discrete_genes[:, i] = rng.integers(low, high + 1, size=pop_size)
            else:
                discrete_genes = np.zeros((pop_size, 0), dtype=np.int64)

            fitness = np.full(pop_size, -np.inf, dtype=dtype)
            needs_evaluation = np.ones(pop_size, dtype=bool)
            generation = np.zeros(pop_size, dtype=np.int32)

            device_info = DeviceInfo.cpu()

        return GPUPopulation(
            feature_genes=feature_genes,
            discrete_genes=discrete_genes,
            fitness=fitness,
            needs_evaluation=needs_evaluation,
            generation=generation,
            device_info=device_info,
        )


class BiasedInitializer(PopulationInitializer):
    """Biased population initialization using feature importance.

    Features with higher importance have higher probability of being selected.
    """

    def __init__(
        self,
        feature_importance: ArrayLike,
        base_prob: float = 0.3,
        importance_scale: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """Initialize biased initializer.

        Args:
            feature_importance: Importance scores for each feature.
            base_prob: Base probability for feature selection.
            importance_scale: Scale factor for importance effect.
            seed: Random seed for reproducibility.
        """
        self.feature_importance = np.asarray(feature_importance, dtype=np.float64)
        self.base_prob = base_prob
        self.importance_scale = importance_scale
        self.seed = seed

    def initialize(
        self,
        pop_size: int,
        n_features: int,
        n_discrete: int,
        discrete_bounds: Sequence[tuple[int, int]],
        use_gpu: bool = False,
        dtype: str = "float64",
    ) -> GPUPopulation:
        """Initialize population with biased feature selection.

        Args:
            pop_size: Number of individuals.
            n_features: Number of features.
            n_discrete: Number of discrete parameters.
            discrete_bounds: List of (min, max) bounds for each discrete param.
            use_gpu: Whether to create on GPU.
            dtype: Data type for fitness array.

        Returns:
            Bias-initialized GPUPopulation.
        """
        if len(discrete_bounds) != n_discrete:
            msg = f"discrete_bounds length ({len(discrete_bounds)}) must match n_discrete ({n_discrete})"
            raise ValueError(msg)

        if len(self.feature_importance) != n_features:
            msg = f"feature_importance length ({len(self.feature_importance)}) must match n_features ({n_features})"
            raise ValueError(msg)

        normalized_importance = self.feature_importance / (np.max(self.feature_importance) + 1e-10)
        selection_probs = self.base_prob + self.importance_scale * normalized_importance * (1 - self.base_prob)
        selection_probs = np.clip(selection_probs, 0, 1)

        rng = np.random.default_rng(self.seed)

        feature_genes = rng.random((pop_size, n_features)) < selection_probs

        if n_discrete > 0:
            discrete_genes = np.zeros((pop_size, n_discrete), dtype=np.int64)
            for i, (low, high) in enumerate(discrete_bounds):
                discrete_genes[:, i] = rng.integers(low, high + 1, size=pop_size)
        else:
            discrete_genes = np.zeros((pop_size, 0), dtype=np.int64)

        fitness = np.full(pop_size, -np.inf, dtype=dtype)
        needs_evaluation = np.ones(pop_size, dtype=bool)
        generation = np.zeros(pop_size, dtype=np.int32)

        population = GPUPopulation(
            feature_genes=feature_genes,
            discrete_genes=discrete_genes,
            fitness=fitness,
            needs_evaluation=needs_evaluation,
            generation=generation,
            device_info=DeviceInfo.cpu(),
        )

        if use_gpu and is_gpu_available():
            return population.to_gpu()

        return population


class LatinHypercubeInitializer(PopulationInitializer):
    """Latin Hypercube Sampling initialization for better coverage.

    Uses scipy's Latin Hypercube sampler on CPU and transfers to GPU.
    """

    def __init__(
        self,
        feature_init_prob: float = 0.5,
        seed: int | None = None,
        strength: int = 1,
    ) -> None:
        """Initialize LHS initializer.

        Args:
            feature_init_prob: Base probability for feature selection.
            seed: Random seed for reproducibility.
            strength: Number of iterations to improve LHS sample.
        """
        if not 0.0 <= feature_init_prob <= 1.0:
            msg = f"feature_init_prob must be in [0, 1], got {feature_init_prob}"
            raise ValueError(msg)

        self.feature_init_prob = feature_init_prob
        self.seed = seed
        self.strength = strength

    def initialize(
        self,
        pop_size: int,
        n_features: int,
        n_discrete: int,
        discrete_bounds: Sequence[tuple[int, int]],
        use_gpu: bool = False,
        dtype: str = "float64",
    ) -> GPUPopulation:
        """Initialize population using Latin Hypercube Sampling.

        Args:
            pop_size: Number of individuals.
            n_features: Number of features.
            n_discrete: Number of discrete parameters.
            discrete_bounds: List of (min, max) bounds for each discrete param.
            use_gpu: Whether to create on GPU.
            dtype: Data type for fitness array.

        Returns:
            LHS-initialized GPUPopulation.
        """
        if len(discrete_bounds) != n_discrete:
            msg = f"discrete_bounds length ({len(discrete_bounds)}) must match n_discrete ({n_discrete})"
            raise ValueError(msg)

        # Feature genes: Use LHS to generate samples in [0, 1], threshold to get binary
        if n_features > 0:
            sampler_features = qmc.LatinHypercube(d=n_features, rng=self.seed, strength=self.strength)
            lhs_features = sampler_features.random(n=pop_size)
            feature_genes = lhs_features < self.feature_init_prob
        else:
            feature_genes = np.zeros((pop_size, 0), dtype=bool)

        # Discrete genes: Use LHS and scale to bounds
        if n_discrete > 0:
            sampler_discrete = qmc.LatinHypercube(d=n_discrete, rng=self.seed, strength=self.strength)
            lhs_discrete = sampler_discrete.random(n=pop_size)

            # Scale to integer bounds
            discrete_genes = np.zeros((pop_size, n_discrete), dtype=np.int64)
            for i, (low, high) in enumerate(discrete_bounds):
                scaled = low + lhs_discrete[:, i] * (high - low + 1)
                discrete_genes[:, i] = np.clip(np.floor(scaled).astype(np.int64), low, high)
        else:
            discrete_genes = np.zeros((pop_size, 0), dtype=np.int64)

        fitness = np.full(pop_size, -np.inf, dtype=dtype)
        needs_evaluation = np.ones(pop_size, dtype=bool)
        generation = np.zeros(pop_size, dtype=np.int32)

        population = GPUPopulation(
            feature_genes=feature_genes,
            discrete_genes=discrete_genes,
            fitness=fitness,
            needs_evaluation=needs_evaluation,
            generation=generation,
            device_info=DeviceInfo.cpu(),
        )

        if use_gpu and is_gpu_available():
            return population.to_gpu()

        return population


class SobolInitializer(PopulationInitializer):
    """Sobol sequence initialization for quasi-random coverage.

    Provides excellent space-filling properties. Best when pop_size is a power of 2.
    """

    def __init__(
        self,
        feature_init_prob: float = 0.5,
        seed: int | None = None,
        scramble: bool = True,
    ) -> None:
        """Initialize Sobol initializer.

        Args:
            feature_init_prob: Base probability for feature selection.
            seed: Random seed for reproducibility.
            scramble: Whether to scramble the Sobol sequence.
        """
        if not 0.0 <= feature_init_prob <= 1.0:
            msg = f"feature_init_prob must be in [0, 1], got {feature_init_prob}"
            raise ValueError(msg)

        self.feature_init_prob = feature_init_prob
        self.seed = seed
        self.scramble = scramble

    def initialize(
        self,
        pop_size: int,
        n_features: int,
        n_discrete: int,
        discrete_bounds: Sequence[tuple[int, int]],
        use_gpu: bool = False,
        dtype: str = "float64",
    ) -> GPUPopulation:
        """Initialize population using Sobol sequence.

        Args:
            pop_size: Number of individuals (ideally power of 2).
            n_features: Number of features.
            n_discrete: Number of discrete parameters.
            discrete_bounds: List of (min, max) bounds for each discrete param.
            use_gpu: Whether to create on GPU.
            dtype: Data type for fitness array.

        Returns:
            Sobol-initialized GPUPopulation.
        """
        if len(discrete_bounds) != n_discrete:
            msg = f"discrete_bounds length ({len(discrete_bounds)}) must match n_discrete ({n_discrete})"
            raise ValueError(msg)

        # Feature genes
        if n_features > 0:
            sampler_features = qmc.Sobol(d=n_features, scramble=self.scramble, rng=self.seed)
            sobol_features = sampler_features.random(n=pop_size)
            feature_genes = sobol_features < self.feature_init_prob
        else:
            feature_genes = np.zeros((pop_size, 0), dtype=bool)

        # Discrete genes
        if n_discrete > 0:
            sampler_discrete = qmc.Sobol(d=n_discrete, scramble=self.scramble, rng=self.seed)
            sobol_discrete = sampler_discrete.random(n=pop_size)

            discrete_genes = np.zeros((pop_size, n_discrete), dtype=np.int64)
            for i, (low, high) in enumerate(discrete_bounds):
                scaled = low + sobol_discrete[:, i] * (high - low + 1)
                discrete_genes[:, i] = np.clip(np.floor(scaled).astype(np.int64), low, high)
        else:
            discrete_genes = np.zeros((pop_size, 0), dtype=np.int64)

        fitness = np.full(pop_size, -np.inf, dtype=dtype)
        needs_evaluation = np.ones(pop_size, dtype=bool)
        generation = np.zeros(pop_size, dtype=np.int32)

        population = GPUPopulation(
            feature_genes=feature_genes,
            discrete_genes=discrete_genes,
            fitness=fitness,
            needs_evaluation=needs_evaluation,
            generation=generation,
            device_info=DeviceInfo.cpu(),
        )

        if use_gpu and is_gpu_available():
            return population.to_gpu()

        return population


class GPUSelectionStrategy(ABC):
    """Abstract base class for GPU-accelerated selection strategies.

    All selection operations are vectorized for efficient GPU execution.
    """

    @abstractmethod
    def select(
        self,
        population: GPUPopulation,
        n_select: int,
    ) -> GPUPopulation:
        """Select individuals from the population.

        Args:
            population: Source population.
            n_select: Number of individuals to select.

        Returns:
            New GPUPopulation with selected individuals.
        """

    def _validate_inputs(self, population: GPUPopulation, n_select: int) -> None:
        """Validate selection inputs."""
        if population.pop_size == 0:
            msg = "Cannot select from empty population"
            raise ValueError(msg)
        if n_select <= 0:
            msg = f"n_select must be positive, got {n_select}"
            raise ValueError(msg)

    def _create_selected_population(
        self,
        population: GPUPopulation,
        selected_indices: ArrayLike,
        new_generation: int | None = None,
    ) -> GPUPopulation:
        """Create a new population from selected indices.

        Args:
            population: Source population.
            selected_indices: Indices of selected individuals.
            new_generation: Generation number for selected individuals.

        Returns:
            New GPUPopulation with selected individuals.
        """
        backend = get_backend()

        feature_genes = population.feature_genes[selected_indices]
        discrete_genes = population.discrete_genes[selected_indices]
        fitness = population.fitness[selected_indices]
        needs_evaluation = population.needs_evaluation[selected_indices]

        if new_generation is not None:
            if population.is_gpu:
                generation = backend.full((len(selected_indices),), new_generation, dtype="int32")
            else:
                generation = np.full(len(selected_indices), new_generation, dtype=np.int32)
        else:
            generation = population.generation[selected_indices]

        indices_cpu = to_numpy(selected_indices)
        selected_ids = [population.individual_ids[i] for i in indices_cpu]
        selected_metadata = [dict(population.metadata[i]) for i in indices_cpu]

        return GPUPopulation(
            feature_genes=feature_genes,
            discrete_genes=discrete_genes,
            fitness=fitness,
            needs_evaluation=needs_evaluation,
            generation=generation,
            device_info=population.device_info,
            individual_ids=selected_ids,
            metadata=selected_metadata,
        )


class GPUTournamentSelection(GPUSelectionStrategy):
    """GPU-accelerated tournament selection.

    Implements vectorized tournament selection where all tournaments
    are run simultaneously for GPU efficiency. Uses GPUKernels for
    optimized CUDA kernel execution when available.
    """

    def __init__(self, tournament_size: int = 3, use_kernels: bool = True) -> None:
        """Initialize tournament selection.

        Args:
            tournament_size: Number of individuals in each tournament.
            use_kernels: Whether to use GPUKernels for optimized execution.
        """
        if tournament_size < 1:
            msg = f"tournament_size must be >= 1, got {tournament_size}"
            raise ValueError(msg)
        self.tournament_size = tournament_size
        self.use_kernels = use_kernels
        self._kernels: GPUKernels | None = None

    def _get_kernels(self) -> GPUKernels:
        """Get GPUKernels instance (lazy initialization)."""
        if self._kernels is None:
            self._kernels = get_gpu_kernels()
        return self._kernels

    def select(
        self,
        population: GPUPopulation,
        n_select: int,
    ) -> GPUPopulation:
        """Select individuals using tournament selection.

        All tournaments are run in parallel using vectorized operations.
        Uses GPUKernels for optimized CUDA kernel execution when available.

        Args:
            population: Source population.
            n_select: Number of individuals to select.

        Returns:
            New GPUPopulation with tournament winners.
        """
        self._validate_inputs(population, n_select)

        pop_size = population.pop_size
        actual_k = min(self.tournament_size, pop_size)

        if self.use_kernels and population.is_gpu:
            try:
                kernels = self._get_kernels()
                if kernels.use_gpu:
                    winner_indices = kernels.tournament_selection(
                        population.fitness,
                        n_select,
                        tournament_size=actual_k,
                    )
                    return self._create_selected_population(population, winner_indices)
            except Exception as e:
                logger.debug("GPUKernels tournament_selection failed, using fallback: %s", e)

        backend = get_backend()

        if population.is_gpu:
            tournament_indices = backend.random_integers(0, pop_size - 1, (n_select, actual_k))
            tournament_fitness = population.fitness[tournament_indices]
            winner_positions = backend.argmax(tournament_fitness, axis=1)
            row_indices = backend.arange(0, n_select, 1)
            winner_indices = tournament_indices[row_indices, winner_positions]
        else:
            rng = np.random.default_rng()
            tournament_indices = rng.integers(0, pop_size, size=(n_select, actual_k))
            tournament_fitness = population.fitness[tournament_indices]
            winner_positions = np.argmax(tournament_fitness, axis=1)
            winner_indices = tournament_indices[np.arange(n_select), winner_positions]

        return self._create_selected_population(population, winner_indices)

    def __repr__(self) -> str:
        return f"GPUTournamentSelection(tournament_size={self.tournament_size})"


class GPURankSelection(GPUSelectionStrategy):
    """GPU-accelerated rank-based selection.

    Selection probability is based on fitness rank rather than raw fitness values.
    Uses linear ranking formula with configurable selection pressure.
    """

    def __init__(self, selection_pressure: float = 2.0) -> None:
        """Initialize rank selection.

        Args:
            selection_pressure: Selection pressure in [1.0, 2.0].
                               Higher values favor fitter individuals more.
        """
        if not 1.0 <= selection_pressure <= 2.0:
            msg = f"selection_pressure must be in [1.0, 2.0], got {selection_pressure}"
            raise ValueError(msg)
        self.selection_pressure = selection_pressure

    def select(
        self,
        population: GPUPopulation,
        n_select: int,
    ) -> GPUPopulation:
        """Select individuals using rank-based selection.

        Args:
            population: Source population.
            n_select: Number of individuals to select.

        Returns:
            New GPUPopulation with selected individuals.
        """
        self._validate_inputs(population, n_select)

        backend = get_backend()
        pop_size = population.pop_size

        if pop_size == 1:
            if population.is_gpu:
                indices = backend.zeros((n_select,), dtype="int64")
            else:
                indices = np.zeros(n_select, dtype=np.int64)
            return self._create_selected_population(population, indices)

        if population.is_gpu:
            fitness = population.fitness

            sorted_indices = backend.argsort(fitness)
            ranks = backend.zeros(pop_size, dtype="float64")
            ranks[sorted_indices] = backend.arange(0, pop_size, 1, dtype="float64")
            ranks = ranks + 1

            # Linear ranking formula
            s = self.selection_pressure
            n = float(pop_size)
            probs = (2 - s + 2 * (s - 1) * (ranks - 1) / (n - 1)) / n

            probs = backend.clip(probs, 0, None)
            probs = probs / backend.sum(probs)

            selected_indices = backend.random_choice(pop_size, n_select, p=probs, replace=True)

        else:
            from scipy.stats import rankdata

            fitness = population.fitness
            ranks = rankdata(fitness, method="average")

            s = self.selection_pressure
            n = float(pop_size)
            probs = (2 - s + 2 * (s - 1) * (ranks - 1) / (n - 1)) / n

            probs = np.maximum(probs, 0)
            probs = probs / np.sum(probs)

            rng = np.random.default_rng()
            selected_indices = rng.choice(pop_size, size=n_select, p=probs, replace=True)

        return self._create_selected_population(population, selected_indices)

    def __repr__(self) -> str:
        return f"GPURankSelection(selection_pressure={self.selection_pressure})"


class GPURouletteWheelSelection(GPUSelectionStrategy):
    """GPU-accelerated roulette wheel (fitness-proportionate) selection."""

    def __init__(self, min_fitness_offset: float = 1e-6) -> None:
        """Initialize roulette wheel selection.

        Args:
            min_fitness_offset: Small offset to ensure positive probabilities.
        """
        self.min_fitness_offset = min_fitness_offset

    def select(
        self,
        population: GPUPopulation,
        n_select: int,
    ) -> GPUPopulation:
        """Select individuals using roulette wheel selection.

        Args:
            population: Source population.
            n_select: Number of individuals to select.

        Returns:
            New GPUPopulation with selected individuals.
        """
        self._validate_inputs(population, n_select)

        backend = get_backend()
        pop_size = population.pop_size
        fitness = population.fitness

        if population.is_gpu:
            valid_mask = backend.logical_not(population.needs_evaluation)

            if not backend.to_scalar(backend.any(valid_mask)):
                indices = backend.random_integers(0, pop_size - 1, (n_select,))
                return self._create_selected_population(population, indices)

            valid_fitness = backend.where(
                valid_mask, fitness, backend.full(fitness.shape, float("-inf"))
            )
            min_fit = backend.min(valid_fitness)

            offset_needed = backend.where(
                min_fit <= 0,
                backend.abs(min_fit) + self.min_fitness_offset,
                self.min_fitness_offset,
            )

            adjusted = backend.where(
                valid_mask,
                fitness + backend.to_scalar(offset_needed),
                backend.zeros(fitness.shape, dtype="float64"),
            )

            total = backend.sum(adjusted)
            probs = adjusted / total

            selected_indices = backend.random_choice(pop_size, n_select, p=probs, replace=True)

        else:
            valid_mask = ~population.needs_evaluation

            if not np.any(valid_mask):
                rng = np.random.default_rng()
                indices = rng.integers(0, pop_size, size=n_select)
                return self._create_selected_population(population, indices)

            valid_fitness = np.where(valid_mask, fitness, -np.inf)
            min_fit = np.min(valid_fitness[valid_mask])

            offset = abs(min_fit) + self.min_fitness_offset if min_fit <= 0 else self.min_fitness_offset

            adjusted = np.where(valid_mask, fitness + offset, 0)
            total = np.sum(adjusted)

            if total == 0:
                valid_indices = np.where(valid_mask)[0]
                rng = np.random.default_rng()
                selected_indices = rng.choice(valid_indices, size=n_select, replace=True)
            else:
                probs = adjusted / total
                rng = np.random.default_rng()
                selected_indices = rng.choice(pop_size, size=n_select, p=probs, replace=True)

        return self._create_selected_population(population, selected_indices)

    def __repr__(self) -> str:
        return f"GPURouletteWheelSelection(offset={self.min_fitness_offset})"


class GPUTruncationSelection(GPUSelectionStrategy):
    """GPU-accelerated truncation selection."""

    def __init__(self, truncation_fraction: float = 0.5) -> None:
        """Initialize truncation selection.

        Args:
            truncation_fraction: Fraction of population to select from.
        """
        if not 0.0 < truncation_fraction <= 1.0:
            msg = f"truncation_fraction must be in (0, 1], got {truncation_fraction}"
            raise ValueError(msg)
        self.truncation_fraction = truncation_fraction

    def select(
        self,
        population: GPUPopulation,
        n_select: int,
    ) -> GPUPopulation:
        """Select individuals using truncation selection.

        Args:
            population: Source population.
            n_select: Number of individuals to select.

        Returns:
            New GPUPopulation with selected individuals.
        """
        self._validate_inputs(population, n_select)

        backend = get_backend()
        pop_size = population.pop_size
        fitness = population.fitness

        cutoff = max(1, int(pop_size * self.truncation_fraction))

        if population.is_gpu:
            sorted_indices = backend.argsort(-fitness)
            eligible_indices = sorted_indices[:cutoff]
            selection_positions = backend.random_integers(0, cutoff - 1, (n_select,))
            selected_indices = eligible_indices[selection_positions]
        else:
            sorted_indices = np.argsort(-fitness)
            eligible_indices = sorted_indices[:cutoff]
            rng = np.random.default_rng()
            selection_positions = rng.integers(0, cutoff, size=n_select)
            selected_indices = eligible_indices[selection_positions]

        return self._create_selected_population(population, selected_indices)

    def __repr__(self) -> str:
        return f"GPUTruncationSelection(fraction={self.truncation_fraction})"


class GPUElitistSelection(GPUSelectionStrategy):
    """GPU-accelerated elitist selection.

    Preserves the top n_elite individuals, fills rest with base strategy.
    """

    def __init__(
        self,
        n_elite: int = 1,
        base_strategy: GPUSelectionStrategy | None = None,
    ) -> None:
        """Initialize elitist selection.

        Args:
            n_elite: Number of elite individuals to preserve.
            base_strategy: Strategy for selecting remaining individuals.
        """
        if n_elite < 0:
            msg = f"n_elite must be >= 0, got {n_elite}"
            raise ValueError(msg)
        self.n_elite = n_elite
        self.base_strategy = base_strategy or GPUTournamentSelection(tournament_size=3)

    def select(
        self,
        population: GPUPopulation,
        n_select: int,
    ) -> GPUPopulation:
        """Select individuals with elitism.

        Args:
            population: Source population.
            n_select: Number of individuals to select.

        Returns:
            New GPUPopulation with elitism applied.
        """
        self._validate_inputs(population, n_select)

        backend = get_backend()
        pop_size = population.pop_size
        fitness = population.fitness

        actual_elite = min(self.n_elite, n_select, pop_size)

        if n_select <= actual_elite:
            if population.is_gpu:
                sorted_indices = backend.argsort(-fitness)
                elite_indices = sorted_indices[:n_select]
            else:
                sorted_indices = np.argsort(-fitness)
                elite_indices = sorted_indices[:n_select]

            return self._create_selected_population(population, elite_indices)

        # Get elite indices
        if population.is_gpu:
            sorted_indices = backend.argsort(-fitness)
            elite_indices = sorted_indices[:actual_elite]
        else:
            sorted_indices = np.argsort(-fitness)
            elite_indices = sorted_indices[:actual_elite]

        # Get remaining through base strategy
        n_remaining = n_select - actual_elite
        remaining_pop = self.base_strategy.select(population, n_remaining)

        # Combine elite and remaining
        elite_pop = self._create_selected_population(population, elite_indices)

        # Concatenate populations
        if population.is_gpu:
            combined_features = backend.concatenate(
                [elite_pop.feature_genes, remaining_pop.feature_genes], axis=0
            )
            combined_discrete = backend.concatenate(
                [elite_pop.discrete_genes, remaining_pop.discrete_genes], axis=0
            )
            combined_fitness = backend.concatenate(
                [elite_pop.fitness, remaining_pop.fitness], axis=0
            )
            combined_needs_eval = backend.concatenate(
                [elite_pop.needs_evaluation, remaining_pop.needs_evaluation], axis=0
            )
            combined_generation = backend.concatenate(
                [elite_pop.generation, remaining_pop.generation], axis=0
            )
        else:
            combined_features = np.concatenate(
                [elite_pop.feature_genes, remaining_pop.feature_genes], axis=0
            )
            combined_discrete = np.concatenate(
                [elite_pop.discrete_genes, remaining_pop.discrete_genes], axis=0
            )
            combined_fitness = np.concatenate(
                [elite_pop.fitness, remaining_pop.fitness], axis=0
            )
            combined_needs_eval = np.concatenate(
                [elite_pop.needs_evaluation, remaining_pop.needs_evaluation], axis=0
            )
            combined_generation = np.concatenate(
                [elite_pop.generation, remaining_pop.generation], axis=0
            )

        combined_ids = elite_pop.individual_ids + remaining_pop.individual_ids
        combined_metadata = elite_pop.metadata + remaining_pop.metadata

        return GPUPopulation(
            feature_genes=combined_features,
            discrete_genes=combined_discrete,
            fitness=combined_fitness,
            needs_evaluation=combined_needs_eval,
            generation=combined_generation,
            device_info=population.device_info,
            individual_ids=combined_ids,
            metadata=combined_metadata,
        )

    def __repr__(self) -> str:
        return f"GPUElitistSelection(n_elite={self.n_elite}, base={self.base_strategy})"


def create_initializer(
    method: str = "random",
    feature_init_prob: float = 0.5,
    feature_importance: ArrayLike | None = None,
    importance_scale: float = 1.0,
    seed: int | None = None,
) -> PopulationInitializer:
    """Create a population initializer.

    Args:
        method: Initialization method ('random', 'biased', 'lhs', 'sobol').
        feature_init_prob: Probability for feature selection.
        feature_importance: Feature importance for biased initialization.
        importance_scale: Scale factor for importance effect (biased only).
        seed: Random seed.

    Returns:
        PopulationInitializer instance.
    """
    method = method.lower()

    if method == "random":
        return RandomInitializer(feature_init_prob=feature_init_prob, seed=seed)

    if method == "biased":
        if feature_importance is None:
            msg = "feature_importance required for biased initialization"
            raise ValueError(msg)
        return BiasedInitializer(
            feature_importance=feature_importance,
            base_prob=feature_init_prob,
            importance_scale=importance_scale,
            seed=seed,
        )

    if method == "lhs":
        return LatinHypercubeInitializer(feature_init_prob=feature_init_prob, seed=seed)

    if method == "sobol":
        return SobolInitializer(feature_init_prob=feature_init_prob, seed=seed)

    msg = f"Unknown initialization method: {method}"
    raise ValueError(msg)


def create_selection_strategy(
    method: str = "tournament",
    tournament_size: int = 3,
    selection_pressure: float = 2.0,
    truncation_fraction: float = 0.5,
    n_elite: int = 0,
) -> GPUSelectionStrategy:
    """Create a selection strategy.

    Args:
        method: Selection method ('tournament', 'rank', 'roulette', 'truncation', 'elitist').
        tournament_size: Size for tournament selection.
        selection_pressure: Pressure for rank selection.
        truncation_fraction: Fraction for truncation selection.
        n_elite: Number of elite for elitist selection.

    Returns:
        GPUSelectionStrategy instance.
    """
    method = method.lower()

    if method == "tournament":
        return GPUTournamentSelection(tournament_size=tournament_size)

    if method == "rank":
        return GPURankSelection(selection_pressure=selection_pressure)

    if method == "roulette":
        return GPURouletteWheelSelection()

    if method == "truncation":
        return GPUTruncationSelection(truncation_fraction=truncation_fraction)

    if method == "elitist":
        base = GPUTournamentSelection(tournament_size=tournament_size)
        return GPUElitistSelection(n_elite=n_elite, base_strategy=base)

    msg = f"Unknown selection method: {method}"
    raise ValueError(msg)