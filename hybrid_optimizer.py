"""GPU-Accelerated Hybrid GA-PSO Optimizer.

This module implements the integration of Genetic Algorithm and Particle Swarm
Optimization for the hybrid optimization framework.

Phase 4 Implementation:
- Step 4.1: Integration Strategy (GPU-Optimized)
- Step 4.2: Warm-Starting PSO (GPU)
- Step 4.3: Two-Level Caching (GPU-Aware)
- Step 4.4: Hybrid Optimizer

The hybrid approach uses GA for discrete/combinatorial optimization (feature
selection, categorical parameters) and PSO for continuous parameter optimization.

Example:
    >>> from hybrid_optimizer import HybridGAPSOOptimizer, HybridConfig
    >>> from .backend import get_backend
    >>>
    >>> # Create optimizer
    >>> config = HybridConfig()
    >>> optimizer = HybridGAPSOOptimizer(
    ...     n_features=50,
    ...     discrete_bounds=[(0, 10), (0, 5)],
    ...     continuous_bounds=[(0.0, 1.0), (0.001, 0.1)],
    ...     config=config,
    ... )
    >>>
    >>> # Run optimization
    >>> result = optimizer.optimize(fitness_fn)
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from .utils.backend import (
    ArrayLike,
    get_backend,
    is_gpu_available,
    to_numpy,
)
from .optimization_config import (
    GASettings,
    PSOSettings,
)
from .utils.data import DeviceInfo
from .genetic_algorithm import (
    GPUPopulation,

    create_initializer as create_ga_initializer,
    create_selection_strategy,
)
from .particle_swarm import (

    RandomSwarmInitializer,
    PSOOptimizer,

)
from .results import GenerationStats
from .utils.performance import (
    GPUKernels,
    GPUMemoryManager,
    TransferOptimizer,
    create_gpu_kernels,
    create_memory_manager,
    create_transfer_optimizer,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)




class HybridFitnessFunction(Protocol):
    """Protocol for hybrid fitness evaluation functions.

    The function receives feature mask, discrete params, and continuous params
    separately for maximum flexibility.
    """

    def __call__(
        self,
        feature_mask: NDArray[np.bool_],
        discrete_params: NDArray[np.int64],
        continuous_params: NDArray[np.float64],
    ) -> float:
        """Evaluate fitness for a single configuration.

        Args:
            feature_mask: Boolean array indicating selected features.
            discrete_params: Array of discrete parameter values.
            continuous_params: Array of continuous parameter values.

        Returns:
            Fitness value (higher is better).
        """
        ...


class BatchHybridFitnessFunction(Protocol):
    """Protocol for batch hybrid fitness evaluation."""

    def __call__(
        self,
        feature_masks: NDArray[np.bool_],
        discrete_params: NDArray[np.int64],
        continuous_params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate fitness for batch of configurations.

        Args:
            feature_masks: Boolean array of shape (batch_size, n_features).
            discrete_params: Array of shape (batch_size, n_discrete).
            continuous_params: Array of shape (batch_size, n_continuous).

        Returns:
            Array of fitness values of shape (batch_size,).
        """
        ...




@dataclass
class CacheEntry:
    """Entry in the chromosome cache."""

    best_continuous_params: NDArray[np.float64]
    best_fitness: float
    pso_iterations: int
    converged: bool
    timestamp: float = field(default_factory=time.time)


class ChromosomeCache:
    """Level 1 cache for chromosome (feature + discrete params) results.

    Stores the best continuous parameters found by PSO for each unique
    chromosome configuration.
    """

    def __init__(
        self,
        max_size: int = 10000,
        eviction_policy: str = "lru",
    ) -> None:
        """Initialize chromosome cache.

        Args:
            max_size: Maximum number of entries to store.
            eviction_policy: Cache eviction policy ('lru', 'lfu', 'fifo').
        """
        self.max_size = max_size
        self.eviction_policy = eviction_policy.lower()

        # Use OrderedDict for LRU/FIFO
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._access_counts: dict[str, int] = {}  # For LFU

        # Statistics
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _compute_key(
        feature_mask: NDArray[np.bool_],
        discrete_params: NDArray[np.int64],
    ) -> str:
        """Compute cache key from chromosome data.

        Args:
            feature_mask: Boolean feature mask.
            discrete_params: Discrete parameter values.

        Returns:
            Hash string key.
        """
        # Convert to bytes for hashing
        feature_bytes = feature_mask.tobytes()
        discrete_bytes = discrete_params.tobytes()

        # Combine and hash
        combined = feature_bytes + discrete_bytes
        return hashlib.sha256(combined).hexdigest()[:16]

    def get(
        self,
        feature_mask: NDArray[np.bool_],
        discrete_params: NDArray[np.int64],
    ) -> CacheEntry | None:
        """Retrieve cached entry.

        Args:
            feature_mask: Boolean feature mask.
            discrete_params: Discrete parameter values.

        Returns:
            CacheEntry if found, None otherwise.
        """
        key = self._compute_key(feature_mask, discrete_params)

        if key in self._cache:
            self.hits += 1

            if self.eviction_policy == "lru":
                self._cache.move_to_end(key)

            if self.eviction_policy == "lfu":
                self._access_counts[key] = self._access_counts.get(key, 0) + 1

            return self._cache[key]

        self.misses += 1
        return None

    def put(
        self,
        feature_mask: NDArray[np.bool_],
        discrete_params: NDArray[np.int64],
        entry: CacheEntry,
    ) -> None:
        """Store entry in cache.

        Args:
            feature_mask: Boolean feature mask.
            discrete_params: Discrete parameter values.
            entry: Cache entry to store.
        """
        key = self._compute_key(feature_mask, discrete_params)

        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict()

        self._cache[key] = entry
        self._access_counts[key] = 1

    def _evict(self) -> None:
        """Evict one entry based on eviction policy."""
        if not self._cache:
            return

        if self.eviction_policy == "fifo":
            self._cache.popitem(last=False)

        elif self.eviction_policy == "lru":
            key, _ = self._cache.popitem(last=False)
            self._access_counts.pop(key, None)

        elif self.eviction_policy == "lfu":
            if self._access_counts:
                min_key = min(self._access_counts, key=self._access_counts.get)  # type: ignore[arg-type]
                del self._cache[min_key]
                del self._access_counts[min_key]
            else:
                self._cache.popitem(last=False)

    def contains(
        self,
        feature_mask: NDArray[np.bool_],
        discrete_params: NDArray[np.int64],
    ) -> bool:
        """Check if chromosome is in cache."""
        key = self._compute_key(feature_mask, discrete_params)
        return key in self._cache

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_counts.clear()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "eviction_policy": self.eviction_policy,
        }


class FullConfigCache:
    """Level 2 cache for full configuration (chromosome + continuous params).

    Stores fitness values for complete configurations.
    """

    def __init__(
        self,
        max_size: int = 100000,
        precision: int = 6,
    ) -> None:
        """Initialize full config cache.

        Args:
            max_size: Maximum number of entries.
            precision: Decimal precision for rounding continuous params.
        """
        self.max_size = max_size
        self.precision = precision
        self._cache: OrderedDict[str, float] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _compute_key(
        self,
        feature_mask: NDArray[np.bool_],
        discrete_params: NDArray[np.int64],
        continuous_params: NDArray[np.float64],
    ) -> str:
        """Compute cache key from full configuration."""
        rounded = np.round(continuous_params, self.precision)

        feature_bytes = feature_mask.tobytes()
        discrete_bytes = discrete_params.tobytes()
        continuous_bytes = rounded.tobytes()

        combined = feature_bytes + discrete_bytes + continuous_bytes
        return hashlib.sha256(combined).hexdigest()[:20]

    def get(
        self,
        feature_mask: NDArray[np.bool_],
        discrete_params: NDArray[np.int64],
        continuous_params: NDArray[np.float64],
    ) -> float | None:
        """Retrieve cached fitness."""
        key = self._compute_key(feature_mask, discrete_params, continuous_params)

        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        self.misses += 1
        return None

    def put(
        self,
        feature_mask: NDArray[np.bool_],
        discrete_params: NDArray[np.int64],
        continuous_params: NDArray[np.float64],
        fitness: float,
    ) -> None:
        """Store fitness in cache."""
        key = self._compute_key(feature_mask, discrete_params, continuous_params)

        if len(self._cache) >= self.max_size and key not in self._cache:
            self._cache.popitem(last=False)

        self._cache[key] = fitness

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0



class WarmStartManager:
    """Manages warm-starting of PSO from similar chromosomes.

    Uses Hamming distance to find similar chromosomes and initializes
    PSO swarm from cached results of similar chromosomes.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.1,
        noise_scale: float = 0.05,
        max_cached_positions: int = 1000,
    ) -> None:
        """Initialize warm start manager.

        Args:
            similarity_threshold: Maximum normalized Hamming distance for warm-start.
            noise_scale: Scale of noise to add to warm-started positions.
            max_cached_positions: Maximum positions to cache.
        """
        self.similarity_threshold = similarity_threshold
        self.noise_scale = noise_scale
        self.max_cached_positions = max_cached_positions

        # Cache of (chromosome_key, best_positions)
        self._position_cache: OrderedDict[str, NDArray[np.float64]] = OrderedDict()

    @staticmethod
    def compute_hamming_distance(
        chrom1: NDArray[np.bool_],
        chrom2: NDArray[np.bool_],
    ) -> float:
        """Compute normalized Hamming distance between two chromosomes."""
        return float(np.mean(chrom1 != chrom2))

    def compute_batch_hamming_distances(
        self,
        new_chromosomes: NDArray[np.bool_],
        cached_chromosomes: NDArray[np.bool_],
        use_gpu: bool = False,
    ) -> NDArray[np.float64]:
        """Compute Hamming distances between new and cached chromosomes.

        Args:
            new_chromosomes: Array of shape (n_new, n_features).
            cached_chromosomes: Array of shape (n_cached, n_features).
            use_gpu: Whether to use GPU for computation.

        Returns:
            Distance matrix of shape (n_new, n_cached).
        """
        if use_gpu and is_gpu_available():
            backend = get_backend()
            new_gpu = backend.asarray(new_chromosomes.astype(np.float64))
            cached_gpu = backend.asarray(cached_chromosomes.astype(np.float64))

            # Compute: |new[:, None] - cached[None, :]| averaged over features
            # Broadcasting: (n_new, 1, n_features) - (1, n_cached, n_features)
            diff = backend.abs(new_gpu[:, None, :] - cached_gpu[None, :, :])
            distances = backend.mean(diff, axis=2)

            return to_numpy(distances)

        return cdist(  # pyright: ignore[reportReturnType]
            new_chromosomes.astype(np.float64),
            cached_chromosomes.astype(np.float64),
            metric="hamming",
        )

    def find_similar(
        self,
        feature_mask: NDArray[np.bool_],
        discrete_params: NDArray[np.int64],
    ) -> tuple[str | None, float]:
        """Find most similar cached chromosome.

        Args:
            feature_mask: Feature mask to find similar for.
            discrete_params: Discrete params (included in key).

        Returns:
            Tuple of (cache_key, distance) or (None, inf) if none found.
        """
        if not self._position_cache:
            return None, float("inf")

        cached_keys = list(self._position_cache.keys())

        best_key = None
        best_distance = float("inf")

        for key in cached_keys:
            #Placeholder
            #linear scan with stored data
            pass

        return best_key, best_distance

    def get_warm_start_positions(
        self,
        feature_mask: NDArray[np.bool_],
        discrete_params: NDArray[np.int64],
        n_positions: int,
        bounds: Sequence[tuple[float, float]],
    ) -> NDArray[np.float64] | None:
        """Get warm-started positions for PSO.

        Args:
            feature_mask: Current chromosome's feature mask.
            discrete_params: Current chromosome's discrete params.
            n_positions: Number of positions needed (swarm size).
            bounds: Parameter bounds [(low, high), ...].

        Returns:
            Warm-started positions or None if no similar found.
        """
        key = ChromosomeCache._compute_key(feature_mask, discrete_params)

        if key in self._position_cache:
            # Direct hit - add noise and return
            cached_positions = self._position_cache[key]
            return self._add_noise(cached_positions, n_positions, bounds)

        return None

    def _add_noise(
        self,
        positions: NDArray[np.float64],
        n_positions: int,
        bounds: Sequence[tuple[float, float]],
    ) -> NDArray[np.float64]:
        """Add noise to cached positions for warm-starting.

        Args:
            positions: Cached positions of shape (n_cached, n_dims).
            n_positions: Number of positions needed.
            bounds: Parameter bounds.

        Returns:
            Noisy positions of shape (n_positions, n_dims).
        """
        n_dims = len(bounds)
        bounds_arr = np.array(bounds)
        ranges = bounds_arr[:, 1] - bounds_arr[:, 0]

        if len(positions) < n_positions:
            repeats = (n_positions + len(positions) - 1) // len(positions)
            positions = np.tile(positions, (repeats, 1))

        result = positions[:n_positions].copy()

        noise = np.random.randn(n_positions, n_dims) * self.noise_scale * ranges
        result = result + noise

        result = np.clip(result, bounds_arr[:, 0], bounds_arr[:, 1])

        return result

    def store_positions(
        self,
        feature_mask: NDArray[np.bool_],
        discrete_params: NDArray[np.int64],
        positions: NDArray[np.float64],
    ) -> None:
        """Store positions for future warm-starting.

        Args:
            feature_mask: Chromosome's feature mask.
            discrete_params: Chromosome's discrete params.
            positions: Best positions found (can be single or multiple).
        """
        key = ChromosomeCache._compute_key(feature_mask, discrete_params)

        # Evict if necessary
        if len(self._position_cache) >= self.max_cached_positions:
            self._position_cache.popitem(last=False)

        # Ensure 2D
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)

        self._position_cache[key] = positions.copy()

    def clear(self) -> None:
        """Clear all cached positions."""
        self._position_cache.clear()



class PSOChromosomeEvaluator:
    """Evaluates chromosomes using PSO for continuous parameter optimization.

    For each chromosome (feature mask + discrete params), runs PSO to find
    optimal continuous parameters.
    """

    def __init__(
        self,
        continuous_bounds: Sequence[tuple[float, float]],
        pso_config: PSOSettings | None = None,
        chromosome_cache: ChromosomeCache | None = None,
        full_config_cache: FullConfigCache | None = None,
        warm_start_manager: WarmStartManager | None = None,
        use_gpu: bool = False,
    ) -> None:
        """Initialize PSO chromosome evaluator.

        Args:
            continuous_bounds: Bounds for continuous parameters.
            pso_config: PSO configuration.
            chromosome_cache: Cache for chromosome results.
            full_config_cache: Cache for full configurations.
            warm_start_manager: Manager for PSO warm-starting.
            use_gpu: Whether to use GPU acceleration.
        """
        self.continuous_bounds = list(continuous_bounds)
        self.n_continuous = len(continuous_bounds)
        self.pso_config = pso_config or PSOSettings()
        self.chromosome_cache = chromosome_cache
        self.full_config_cache = full_config_cache
        self.warm_start_manager = warm_start_manager
        self.use_gpu = use_gpu and is_gpu_available()

        self._pso_optimizer: PSOOptimizer | None = None

        self.pso_evaluations = 0
        self.cache_hits = 0

    def _get_pso_optimizer(self) -> PSOOptimizer:
        """Get or create PSO optimizer."""
        if self._pso_optimizer is None:
            self._pso_optimizer = PSOOptimizer.from_config(
                self.pso_config,
                use_gpu=self.use_gpu,
            )
        return self._pso_optimizer

    def evaluate_chromosome(
        self,
        feature_mask: NDArray[np.bool_],
        discrete_params: NDArray[np.int64],
        fitness_fn: HybridFitnessFunction,
    ) -> tuple[float, NDArray[np.float64], dict[str, Any]]:
        """Evaluate a single chromosome using PSO.

        Args:
            feature_mask: Boolean feature mask.
            discrete_params: Discrete parameter values.
            fitness_fn: Fitness evaluation function.

        Returns:
            Tuple of (best_fitness, best_continuous_params, metadata).
        """
        if self.chromosome_cache is not None:
            cached = self.chromosome_cache.get(feature_mask, discrete_params)
            if cached is not None:
                self.cache_hits += 1
                return (
                    cached.best_fitness,
                    cached.best_continuous_params,
                    {
                        "cached": True,
                        "pso_iterations": cached.pso_iterations,
                        "converged": cached.converged,
                    },
                )

        eval_count = [0]
        
        def pso_fitness(positions: ArrayLike) -> ArrayLike:
            """PSO fitness function wrapper."""
            positions_np = to_numpy(positions)
            n_particles = positions_np.shape[0]
            fitness = np.empty(n_particles, dtype=np.float64)

            for i in range(n_particles):
                if self.full_config_cache is not None:
                    cached_fit = self.full_config_cache.get(
                        feature_mask, discrete_params, positions_np[i]
                    )
                    if cached_fit is not None:
                        fitness[i] = cached_fit
                        continue

                fitness[i] = fitness_fn(feature_mask, discrete_params, positions_np[i])
                eval_count[0] += 1

                if self.full_config_cache is not None:
                    self.full_config_cache.put(
                        feature_mask, discrete_params, positions_np[i], fitness[i]
                    )

            return fitness

        initial_swarm = None
        if self.warm_start_manager is not None:
            warm_positions = self.warm_start_manager.get_warm_start_positions(
                feature_mask,
                discrete_params,
                self.pso_config.swarm_size,
                self.continuous_bounds,
            )
            if warm_positions is not None:
                initializer = RandomSwarmInitializer()
                initial_swarm = initializer.initialize(
                    self.pso_config.swarm_size,
                    self.continuous_bounds,
                    use_gpu=self.use_gpu,
                )
                if self.use_gpu:
                    backend = get_backend()
                    initial_swarm.positions = backend.asarray(warm_positions)
                else:
                    initial_swarm.positions = warm_positions

        logger.debug("Starting PSO optimization for chromosome (swarm=%d, iter=%d)",
                    self.pso_config.swarm_size, self.pso_config.max_iterations)
        optimizer = self._get_pso_optimizer()
        result = optimizer.optimize(
            pso_fitness,
            self.continuous_bounds,
            initial_swarm=initial_swarm,
        )
        logger.debug("PSO completed: best_fitness=%.4f, evaluations=%d",
                    result.best_fitness, eval_count[0])

        self.pso_evaluations += 1

        if self.chromosome_cache is not None:
            entry = CacheEntry(
                best_continuous_params=result.best_position,
                best_fitness=result.best_fitness,
                pso_iterations=result.n_iterations,
                converged=result.converged,
            )
            self.chromosome_cache.put(feature_mask, discrete_params, entry)

        if self.warm_start_manager is not None:
            self.warm_start_manager.store_positions(
                feature_mask,
                discrete_params,
                result.best_position,
            )

        return (
            result.best_fitness,
            result.best_position,
            {
                "cached": False,
                "pso_iterations": result.n_iterations,
                "converged": result.converged,
            },
        )

    def evaluate_population(
        self,
        population: GPUPopulation,
        fitness_fn: HybridFitnessFunction,
    ) -> None:
        """Evaluate all chromosomes needing evaluation in population.

        Args:
            population: GPU population to evaluate.
            fitness_fn: Fitness evaluation function.
        """
        eval_indices = population.get_unevaluated_indices()
        eval_indices_np = to_numpy(eval_indices)

        if len(eval_indices_np) == 0:
            return

        n_to_eval = len(eval_indices_np)
        logger.info("Evaluating %d chromosomes...", n_to_eval)

        feature_genes_np = to_numpy(population.feature_genes)
        discrete_genes_np = to_numpy(population.discrete_genes)

        fitness_values = []
        continuous_params_list = []

        for i, idx in enumerate(eval_indices_np):
            if i > 0 and i % 5 == 0:
                logger.info("  Progress: %d/%d chromosomes evaluated", i, n_to_eval)
            
            feature_mask = feature_genes_np[idx].astype(bool)
            discrete_params = discrete_genes_np[idx]

            fitness, continuous_params, _ = self.evaluate_chromosome(
                feature_mask,
                discrete_params,
                fitness_fn,
            )

            fitness_values.append(fitness)
            continuous_params_list.append(continuous_params)

        logger.info("  Completed: %d/%d chromosomes evaluated", n_to_eval, n_to_eval)

        fitness_arr = np.array(fitness_values, dtype=np.float64)
        population.set_fitness(eval_indices, fitness_arr)

    def evaluate_population_batch(
        self,
        population: GPUPopulation,
        fitness_fn: BatchHybridFitnessFunction,
        batch_size: int = 32,
    ) -> None:
        """Evaluate population with batch fitness function for GPU efficiency.

        This method groups chromosomes and evaluates them in batches,
        which can be more efficient for GPU-accelerated fitness functions.

        Args:
            population: GPU population to evaluate.
            fitness_fn: Batch fitness evaluation function.
            batch_size: Number of configurations to evaluate at once.
        """
        eval_indices = population.get_unevaluated_indices()
        eval_indices_np = to_numpy(eval_indices)

        if len(eval_indices_np) == 0:
            return

        feature_genes_np = to_numpy(population.feature_genes)
        discrete_genes_np = to_numpy(population.discrete_genes)

        unique_chromosomes: dict[str, tuple[NDArray, NDArray, int]] = {}
        chromosome_to_continuous: dict[str, NDArray[np.float64]] = {}

        for idx in eval_indices_np:
            feature_mask = feature_genes_np[idx].astype(bool)
            discrete_params = discrete_genes_np[idx]
            key = ChromosomeCache._compute_key(feature_mask, discrete_params)

            if key not in unique_chromosomes:
                unique_chromosomes[key] = (feature_mask, discrete_params, idx)


        def single_fitness_wrapper(
            fm: NDArray[np.bool_],
            dp: NDArray[np.int64],
            cp: NDArray[np.float64],
        ) -> float:
            """Wrapper for batch function as single evaluator."""
            result = fitness_fn(
                fm.reshape(1, -1),
                dp.reshape(1, -1),
                cp.reshape(1, -1),
            )
            return float(result[0])

        for key, (feature_mask, discrete_params, _) in unique_chromosomes.items():
            _, continuous_params, _ = self.evaluate_chromosome(
                feature_mask,
                discrete_params,
                single_fitness_wrapper,  # pyright: ignore[reportArgumentType]
            )
            chromosome_to_continuous[key] = continuous_params

        n_eval = len(eval_indices_np)
        fitness_values = np.empty(n_eval, dtype=np.float64)

        for batch_start in range(0, n_eval, batch_size):
            batch_end = min(batch_start + batch_size, n_eval)
            batch_indices = eval_indices_np[batch_start:batch_end]
            batch_n = len(batch_indices)

            batch_features = np.empty((batch_n, population.n_features), dtype=bool)
            batch_discrete = np.empty((batch_n, population.n_discrete), dtype=np.int64)
            batch_continuous = np.empty(
                (batch_n, self.n_continuous), dtype=np.float64
            )

            for i, idx in enumerate(batch_indices):
                feature_mask = feature_genes_np[idx].astype(bool)
                discrete_params = discrete_genes_np[idx]
                key = ChromosomeCache._compute_key(feature_mask, discrete_params)

                batch_features[i] = feature_mask
                batch_discrete[i] = discrete_params
                batch_continuous[i] = chromosome_to_continuous[key]

            batch_fitness = fitness_fn(batch_features, batch_discrete, batch_continuous)
            fitness_values[batch_start:batch_end] = batch_fitness

        population.set_fitness(eval_indices, fitness_values)

    def get_stats(self) -> dict[str, Any]:
        """Get evaluator statistics."""
        stats = {
            "pso_evaluations": self.pso_evaluations,
            "cache_hits": self.cache_hits,
        }

        if self.chromosome_cache is not None:
            stats["chromosome_cache"] = self.chromosome_cache.get_stats()  # pyright: ignore[reportArgumentType]

        if self.full_config_cache is not None:
            stats["full_config_cache"] = {  # pyright: ignore[reportArgumentType]
                "size": self.full_config_cache.size,
                "hit_rate": self.full_config_cache.hit_rate,
            }

        return stats



@dataclass
class HybridConfig:
    """Configuration for hybrid GA-PSO optimizer.

    Attributes:
        ga_config: GA configuration.
        pso_config: PSO configuration.
        use_chromosome_cache: Whether to use chromosome caching.
        chromosome_cache_size: Maximum chromosome cache entries.
        use_full_config_cache: Whether to use full config caching.
        full_config_cache_size: Maximum full config cache entries.
        use_warm_start: Whether to use PSO warm-starting.
        warm_start_threshold: Similarity threshold for warm-starting.
        use_gpu: Whether to use GPU acceleration.
        verbose: Whether to print progress.
        random_seed: Random seed for reproducibility.
    """

    ga_config: GASettings = field(default_factory=GASettings)
    pso_config: PSOSettings = field(default_factory=PSOSettings)
    use_chromosome_cache: bool = True
    chromosome_cache_size: int = 10000
    use_full_config_cache: bool = True
    full_config_cache_size: int = 100000
    use_warm_start: bool = True
    warm_start_threshold: float = 0.1
    use_gpu: bool = False
    verbose: bool = True
    random_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if isinstance(self.ga_config, dict):
            self.ga_config = GASettings(**self.ga_config)
        if isinstance(self.pso_config, dict):
            self.pso_config = PSOSettings(**self.pso_config)


@dataclass
class HybridResult:
    """Result of hybrid GA-PSO optimization.

    Attributes:
        best_feature_mask: Best feature selection mask.
        best_discrete_params: Best discrete parameter values.
        best_continuous_params: Best continuous parameter values.
        best_fitness: Best fitness achieved.
        n_generations: Number of GA generations.
        n_pso_evaluations: Total PSO optimizations run.
        converged: Whether optimization converged.
        elapsed_time: Total optimization time in seconds.
        generation_history: History of best fitness per generation.
        cache_stats: Cache statistics.
        final_population: Final GA population.
        generation_stats: Structured generation statistics (from results.py).
    """

    best_feature_mask: NDArray[np.bool_]
    best_discrete_params: NDArray[np.int64]
    best_continuous_params: NDArray[np.float64]
    best_fitness: float
    n_generations: int
    n_pso_evaluations: int
    converged: bool
    elapsed_time: float
    generation_history: list[dict[str, float]]
    cache_stats: dict[str, Any]
    final_population: GPUPopulation | None = None
    generation_stats: list[GenerationStats] = field(default_factory=list)


class HybridGAPSOOptimizer:
    """GPU-accelerated Hybrid GA-PSO Optimizer.

    Uses Genetic Algorithm for discrete optimization (feature selection,
    categorical/integer parameters) and Particle Swarm Optimization for
    continuous parameter optimization.

    Example:
        >>> optimizer = HybridGAPSOOptimizer(
        ...     n_features=50,
        ...     discrete_bounds=[(0, 10), (0, 5)],
        ...     continuous_bounds=[(0.0, 1.0), (0.001, 0.1)],
        ... )
        >>> result = optimizer.optimize(fitness_fn)
    """

    def __init__(
        self,
        n_features: int,
        discrete_bounds: Sequence[tuple[int, int]],
        continuous_bounds: Sequence[tuple[float, float]],
        config: HybridConfig | None = None,
    ) -> None:
        """Initialize hybrid optimizer.

        Args:
            n_features: Number of features for selection.
            discrete_bounds: Bounds for discrete parameters [(min, max), ...].
            continuous_bounds: Bounds for continuous parameters [(min, max), ...].
            config: Hybrid configuration.
        """
        self.n_features = n_features
        self.discrete_bounds = list(discrete_bounds)
        self.continuous_bounds = list(continuous_bounds)
        self.n_discrete = len(discrete_bounds)
        self.n_continuous = len(continuous_bounds)
        self.config = config or HybridConfig()

        self._init_components()

        self.population: GPUPopulation | None = None
        self.best_feature_mask: NDArray[np.bool_] | None = None
        self.best_discrete_params: NDArray[np.int64] | None = None
        self.best_continuous_params: NDArray[np.float64] | None = None
        self.best_fitness: float = float("-inf")
        self.generation_history: list[dict[str, float]] = []
        self.generation_stats_list: list[GenerationStats] = []
        
        self._gpu_kernels: GPUKernels | None = None
        self._memory_manager: GPUMemoryManager | None = None
        self._transfer_optimizer: TransferOptimizer | None = None
        
        if self.config.use_gpu and is_gpu_available():
            try:
                self._gpu_kernels = create_gpu_kernels()
                self._memory_manager = create_memory_manager(memory_limit_fraction=0.8)
                self._transfer_optimizer = create_transfer_optimizer()
                if self.config.verbose:
                    logger.info("GPU components initialized successfully")
            except Exception as e:
                logger.warning("Failed to initialize GPU components: %s", e)
                self._gpu_kernels = None
                self._memory_manager = None
                self._transfer_optimizer = None

    def _init_components(self) -> None:
        """Initialize optimizer components."""
        use_gpu = self.config.use_gpu and is_gpu_available()

        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        chromosome_cache = None
        if self.config.use_chromosome_cache:
            chromosome_cache = ChromosomeCache(
                max_size=self.config.chromosome_cache_size,
            )

        full_config_cache = None
        if self.config.use_full_config_cache:
            full_config_cache = FullConfigCache(
                max_size=self.config.full_config_cache_size,
            )

        warm_start_manager = None
        if self.config.use_warm_start:
            warm_start_manager = WarmStartManager(
                similarity_threshold=self.config.warm_start_threshold,
            )

        self.pso_evaluator = PSOChromosomeEvaluator(
            continuous_bounds=self.continuous_bounds,
            pso_config=self.config.pso_config,
            chromosome_cache=chromosome_cache,
            full_config_cache=full_config_cache,
            warm_start_manager=warm_start_manager,
            use_gpu=use_gpu,
        )

        init_method_map = {
            "random": "random",
            "latin_hypercube": "lhs",
            "sobol": "sobol",
            "halton": "random",  # fallback
            "grid": "random",  # fallback
        }
        init_method = init_method_map.get(
            self.config.ga_config.initialization_method.name.lower(),
            "random",
        )

        self.population_initializer = create_ga_initializer(
            method=init_method,
            feature_init_prob=0.5,
            seed=self.config.random_seed,
        )

        selection_method_map = {
            "tournament": "tournament",
            "roulette": "roulette",
            "rank": "rank",
            "truncation": "truncation",
            "elitist": "elitist",
        }
        selection_method = selection_method_map.get(
            self.config.ga_config.selection_type.name.lower(),
            "tournament",
        )

        self.selection_strategy = create_selection_strategy(
            method=selection_method,
            tournament_size=self.config.ga_config.tournament_size,
            n_elite=self.config.ga_config.elitism_count,
        )

    def _initialize_population(self) -> GPUPopulation:
        """Initialize GA population."""
        use_gpu = self.config.use_gpu and is_gpu_available()

        return self.population_initializer.initialize(
            pop_size=self.config.ga_config.population_size,
            n_features=self.n_features,
            n_discrete=self.n_discrete,
            discrete_bounds=self.discrete_bounds,
            use_gpu=use_gpu,
        )

    def _crossover(
        self,
        parent1_features: ArrayLike,
        parent2_features: ArrayLike,
        parent1_discrete: ArrayLike,
        parent2_discrete: ArrayLike,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Perform crossover on two parents.

        Args:
            parent1_features: First parent's feature genes.
            parent2_features: Second parent's feature genes.
            parent1_discrete: First parent's discrete genes.
            parent2_discrete: Second parent's discrete genes.

        Returns:
            Tuple of (child1_features, child2_features, child1_discrete, child2_discrete).
        """
        mask = np.random.random(len(parent1_features)) < 0.5
        child1_features = np.where(mask, parent1_features, parent2_features)
        child2_features = np.where(mask, parent2_features, parent1_features)

        mask_discrete = np.random.random(len(parent1_discrete)) < 0.5
        child1_discrete = np.where(mask_discrete, parent1_discrete, parent2_discrete)
        child2_discrete = np.where(mask_discrete, parent2_discrete, parent1_discrete)

        return child1_features, child2_features, child1_discrete, child2_discrete

    def _mutate(
        self,
        feature_genes: ArrayLike,
        discrete_genes: ArrayLike,
        feature_mutation_rate: float,
        discrete_mutation_rate: float,
    ) -> tuple[ArrayLike, ArrayLike]:
        """Mutate genes.

        Args:
            feature_genes: Feature genes to mutate.
            discrete_genes: Discrete genes to mutate.
            feature_mutation_rate: Probability of mutating each feature.
            discrete_mutation_rate: Probability of mutating each discrete param.

        Returns:
            Tuple of (mutated_features, mutated_discrete).
        """
        feature_mask = np.random.random(len(feature_genes)) < feature_mutation_rate
        mutated_features = np.where(feature_mask, ~feature_genes, feature_genes)

        mutated_discrete = discrete_genes.copy()
        for i, (low, high) in enumerate(self.discrete_bounds):
            if np.random.random() < discrete_mutation_rate:
                mutated_discrete[i] = np.random.randint(low, high + 1)

        return mutated_features, mutated_discrete

    def _create_offspring(
        self,
        parents: GPUPopulation,
        generation: int,
    ) -> GPUPopulation:
        """Create offspring population through crossover and mutation.

        Args:
            parents: Parent population.
            generation: Current generation number.

        Returns:
            Offspring population.
        """
        pop_size = self.config.ga_config.population_size
        crossover_rate = self.config.ga_config.crossover_rate
        mutation_rate = self.config.ga_config.mutation_rate

        feature_mutation_rate = mutation_rate * 0.5

        parent_features = to_numpy(parents.feature_genes)
        parent_discrete = to_numpy(parents.discrete_genes)
        n_parents = len(parent_features)

        offspring_features = np.empty((pop_size, self.n_features), dtype=bool)
        offspring_discrete = np.empty((pop_size, self.n_discrete), dtype=np.int64)

        elite_count = self.config.ga_config.elitism_count
        if elite_count > 0:
            fitness_np = to_numpy(parents.fitness)
            elite_indices = np.argsort(-fitness_np)[:elite_count]

            for i, idx in enumerate(elite_indices):
                offspring_features[i] = parent_features[idx]
                offspring_discrete[i] = parent_discrete[idx]

        for i in range(elite_count, pop_size, 2):
            p1_idx = np.random.randint(n_parents)
            p2_idx = np.random.randint(n_parents)

            if np.random.random() < crossover_rate:
                # Crossover
                c1_f, c2_f, c1_d, c2_d = self._crossover(
                    parent_features[p1_idx],
                    parent_features[p2_idx],
                    parent_discrete[p1_idx],
                    parent_discrete[p2_idx],
                )
            else:
                # Copy parents
                c1_f, c1_d = parent_features[p1_idx].copy(), parent_discrete[p1_idx].copy()
                c2_f, c2_d = parent_features[p2_idx].copy(), parent_discrete[p2_idx].copy()

            # Mutation
            c1_f, c1_d = self._mutate(c1_f, c1_d, feature_mutation_rate, mutation_rate)
            c2_f, c2_d = self._mutate(c2_f, c2_d, feature_mutation_rate, mutation_rate)

            offspring_features[i] = c1_f
            offspring_discrete[i] = c1_d

            if i + 1 < pop_size:
                offspring_features[i + 1] = c2_f
                offspring_discrete[i + 1] = c2_d

        # Create offspring population
        use_gpu = self.config.use_gpu and is_gpu_available()

        if use_gpu:
            backend = get_backend()
            feature_genes = backend.asarray(offspring_features)
            discrete_genes = backend.asarray(offspring_discrete)
            fitness = backend.full((pop_size,), float("-inf"), dtype="float64")
            needs_evaluation = backend.ones((pop_size,), dtype="bool")
            generation_arr = backend.full((pop_size,), generation, dtype="int32")
            device_info = DeviceInfo.gpu()
        else:
            feature_genes = offspring_features
            discrete_genes = offspring_discrete
            fitness = np.full(pop_size, -np.inf, dtype=np.float64)
            needs_evaluation = np.ones(pop_size, dtype=bool)
            generation_arr = np.full(pop_size, generation, dtype=np.int32)
            device_info = DeviceInfo.cpu()

        if elite_count > 0:
            parent_fitness = to_numpy(parents.fitness)
            for i, idx in enumerate(elite_indices):
                if use_gpu:
                    backend = get_backend()
                    fitness[i] = backend.asarray([parent_fitness[idx]])[0]
                else:
                    fitness[i] = parent_fitness[idx]
                needs_evaluation[i] = False

        return GPUPopulation(
            feature_genes=feature_genes,
            discrete_genes=discrete_genes,
            fitness=fitness,
            needs_evaluation=needs_evaluation,
            generation=generation_arr,
            device_info=device_info,
        )

    def _update_best(
        self,
        feature_mask: NDArray[np.bool_],
        discrete_params: NDArray[np.int64],
        continuous_params: NDArray[np.float64],
        fitness: float,
    ) -> bool:
        """Update best solution if improved.

        Returns:
            True if best was updated.
        """
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_feature_mask = feature_mask.copy()
            self.best_discrete_params = discrete_params.copy()
            self.best_continuous_params = continuous_params.copy()
            return True
        return False

    def optimize(
        self,
        fitness_fn: HybridFitnessFunction,
    ) -> HybridResult:
        """Run hybrid GA-PSO optimization.

        Args:
            fitness_fn: Fitness function taking (feature_mask, discrete_params, continuous_params).

        Returns:
            HybridResult with optimization results.
        """
        start_time = time.time()

        if self.config.verbose:
            logger.info("Starting Hybrid GA-PSO optimization")
            logger.info(
                "Features: %d, Discrete: %d, Continuous: %d",
                self.n_features,
                self.n_discrete,
                self.n_continuous,
            )

        memory_context = None
        if self._memory_manager is not None:
            try:
                # Estimate memory needed for population
                pop_size = self.config.ga_config.population_size
                estimated_bytes = self._memory_manager.estimate_array_size(
                    shape=(pop_size, self.n_features),
                    dtype=np.float64,
                )
                if self._memory_manager.check_memory_available(estimated_bytes * 3):
                    memory_context = self._memory_manager.memory_scope("optimization")
                    memory_context.__enter__()
            except Exception as e:
                logger.debug("Memory manager setup failed: %s", e)
                memory_context = None

        try:
            self.population = self._initialize_population()

            self.pso_evaluator.evaluate_population(self.population, fitness_fn)

            self._find_and_update_best(fitness_fn)

            if self.config.verbose:
                logger.info("Initial best fitness: %.6f", self.best_fitness)

            generations_without_improvement = 0
            converged = False

            for generation in range(self.config.ga_config.max_generations):
                # Selection
                parents = self.selection_strategy.select(
                    self.population,
                    self.config.ga_config.population_size,
                )

                # Create offspring
                self.population = self._create_offspring(parents, generation + 1)

                # Evaluate offspring
                self.pso_evaluator.evaluate_population(self.population, fitness_fn)

                # Update best
                improved = self._find_and_update_best(fitness_fn)

                # Track statistics using GenerationStats from results.py
                stats = self.population.get_fitness_statistics()
                stats["generation"] = float(generation)
                stats["best_ever"] = self.best_fitness
                self.generation_history.append(stats)
                
                # Create structured GenerationStats
                gen_stats = GenerationStats(
                    generation=generation,
                    best_fitness=stats.get("best", float("-inf")),
                    mean_fitness=stats.get("mean", 0.0),
                    std_fitness=stats.get("std", 0.0),
                    median_fitness=stats.get("median", 0.0),
                    min_fitness=stats.get("min", float("-inf")),
                    diversity=stats.get("diversity", 0.0) if "diversity" in stats else 0.0,
                    n_evaluated=int(stats.get("n_valid", 0)),
                    elapsed_time=time.time() - start_time,
                )
                self.generation_stats_list.append(gen_stats)

                if improved:
                    generations_without_improvement = 0
                    if self.config.verbose:
                        logger.info(
                            "Gen %d: New best fitness: %.6f",
                            generation,
                            self.best_fitness,
                        )
                else:
                    generations_without_improvement += 1

                if (
                    generations_without_improvement
                    >= self.config.ga_config.early_stopping_generations
                ):
                    if self.config.verbose:
                        logger.info("Early stopping at generation %d", generation)
                    converged = True
                    break

            elapsed_time = time.time() - start_time

            if self.config.verbose:
                logger.info("Optimization complete in %.2fs", elapsed_time)
                logger.info("Best fitness: %.6f", self.best_fitness)

            memory_stats: dict[Any, Any] = {}
            if self._memory_manager is not None:
                try:
                    mem_info = self._memory_manager.get_stats()
                    memory_stats = {
                        "peak_memory_mb": mem_info.peak_allocated / (1024 * 1024),  # pyright: ignore[reportAttributeAccessIssue]
                        "current_memory_mb": mem_info.current_allocated / (1024 * 1024),  # pyright: ignore[reportAttributeAccessIssue]
                    }
                except Exception:
                    pass

            result = HybridResult(
                best_feature_mask=self.best_feature_mask,  # type: ignore[arg-type]
                best_discrete_params=self.best_discrete_params,  # type: ignore[arg-type]
                best_continuous_params=self.best_continuous_params,  # type: ignore[arg-type]
                best_fitness=self.best_fitness,
                n_generations=len(self.generation_history),
                n_pso_evaluations=self.pso_evaluator.pso_evaluations,
                converged=converged,
                elapsed_time=elapsed_time,
                generation_history=self.generation_history,
                cache_stats={**self.pso_evaluator.get_stats(), **memory_stats},
                final_population=self.population,
                generation_stats=self.generation_stats_list,
            )

        finally:
            if memory_context is not None:
                try:
                    memory_context.__exit__(None, None, None)
                except Exception:
                    pass
            if self._memory_manager is not None:
                try:
                    self._memory_manager.free_unused()
                except Exception:
                    pass

        return result

    def _find_and_update_best(self, fitness_fn: HybridFitnessFunction) -> bool:
        """Find and update best individual in population.

        Returns:
            True if best was updated.
        """
        if self.population is None:
            return False

        best_idx = self.population.get_best_index()
        individual = self.population.get_individual(best_idx)

        feature_mask = individual["feature_genes"].astype(bool)
        discrete_params = individual["discrete_genes"]
        fitness = individual["fitness"]

        if self.pso_evaluator.chromosome_cache is not None:
            cached = self.pso_evaluator.chromosome_cache.get(feature_mask, discrete_params)
            if cached is not None:
                continuous_params = cached.best_continuous_params
            else:
                _, continuous_params, _ = self.pso_evaluator.evaluate_chromosome(
                    feature_mask, discrete_params, fitness_fn
                )
        else:
            _, continuous_params, _ = self.pso_evaluator.evaluate_chromosome(
                feature_mask, discrete_params, fitness_fn
            )

        return self._update_best(feature_mask, discrete_params, continuous_params, fitness)


def create_hybrid_optimizer(
    n_features: int,
    discrete_bounds: Sequence[tuple[int, int]],
    continuous_bounds: Sequence[tuple[float, float]],
    population_size: int = 50,
    max_generations: int = 100,
    swarm_size: int = 30,
    pso_iterations: int = 50,
    use_gpu: bool = False,
    seed: int | None = None,
) -> HybridGAPSOOptimizer:
    """Create a hybrid GA-PSO optimizer with common settings.

    Args:
        n_features: Number of features.
        discrete_bounds: Bounds for discrete parameters.
        continuous_bounds: Bounds for continuous parameters.
        population_size: GA population size.
        max_generations: Maximum GA generations.
        swarm_size: PSO swarm size.
        pso_iterations: Maximum PSO iterations.
        use_gpu: Whether to use GPU.
        seed: Random seed.

    Returns:
        HybridGAPSOOptimizer instance.
    """
    ga_config = GASettings(
        population_size=population_size,
        max_generations=max_generations,
    )

    pso_config = PSOSettings(
        swarm_size=swarm_size,
        max_iterations=pso_iterations,
    )

    config = HybridConfig(
        ga_config=ga_config,
        pso_config=pso_config,
        use_gpu=use_gpu,
        random_seed=seed,
    )

    return HybridGAPSOOptimizer(
        n_features=n_features,
        discrete_bounds=discrete_bounds,
        continuous_bounds=continuous_bounds,
        config=config,
    )
