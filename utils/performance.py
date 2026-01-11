from __future__ import annotations

import contextlib
import functools
import logging
import time
import weakref
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from numpy.typing import NDArray


from .common import (
    check_gpu_available,
    ensure_numpy,
    get_array_module,
    is_gpu_array,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

logger = logging.getLogger(__name__)

ArrayType = TypeVar("ArrayType", bound=np.ndarray)


class MemoryUnit(Enum):
    """Memory unit for display."""

    BYTES = auto()
    KB = auto()
    MB = auto()
    GB = auto()


MEMORY_FACTORS: dict[MemoryUnit, float] = {
    MemoryUnit.BYTES: 1.0,
    MemoryUnit.KB: 1024.0,
    MemoryUnit.MB: 1024.0**2,
    MemoryUnit.GB: 1024.0**3,
}

DEFAULT_MEMORY_LIMIT_FRACTION = 0.8
DEFAULT_POOL_SIZE_MB = 512
MIN_ARRAY_SIZE_FOR_GPU = 1000
DEFAULT_TRANSFER_CHUNK_SIZE = 64 * 1024 * 1024


class GPUStreamManager:
    """Manages CUDA streams for concurrent kernel execution and async transfers.

    Provides a pool of reusable streams to overlap:
    - Data transfers (CPU↔GPU)
    - Kernel execution
    - Multiple independent operations

    Example:
        >>> manager = GPUStreamManager(n_streams=4)
        >>> with manager.get_stream() as stream:
        ...     # Operations on this stream
        ...     pass
    """

    def __init__(self, n_streams: int = 4) -> None:
        """Initialize stream manager.

        Args:
            n_streams: Number of streams to pre-allocate.
        """
        self._gpu_available = check_gpu_available()
        self._n_streams = n_streams
        self._streams: list[Any] = []
        self._stream_in_use: list[bool] = []
        self._events: dict[int, Any] = {}

        if self._gpu_available:
            self._initialize_streams()

    def _initialize_streams(self) -> None:
        """Initialize CUDA streams."""
        try:
            import cupy as cp

            for _ in range(self._n_streams):
                stream = cp.cuda.Stream(non_blocking=True)
                self._streams.append(stream)
                self._stream_in_use.append(False)

            logger.debug("Initialized %d CUDA streams", self._n_streams)
        except Exception as e:
            logger.warning("Failed to initialize CUDA streams: %s", e)
            self._gpu_available = False

    def get_stream(self, index: int | None = None) -> Any:
        """Get an available stream.

        Args:
            index: Specific stream index, or None to get first available.

        Returns:
            CUDA stream or None if GPU unavailable.
        """
        if not self._gpu_available or not self._streams:
            return None

        if index is not None:
            if 0 <= index < len(self._streams):
                self._stream_in_use[index] = True
                return self._streams[index]
            return None

        for i, in_use in enumerate(self._stream_in_use):
            if not in_use:
                self._stream_in_use[i] = True
                return self._streams[i]

        return self._streams[0]

    def release_stream(self, stream: Any) -> None:
        """Release a stream back to the pool.

        Args:
            stream: Stream to release.
        """
        if stream is None:
            return

        for i, s in enumerate(self._streams):
            if s is stream:
                self._stream_in_use[i] = False
                break

    @contextlib.contextmanager
    def stream_context(self, index: int | None = None) -> Generator[Any, None, None]:
        """Context manager for stream usage.

        Args:
            index: Specific stream index.

        Yields:
            CUDA stream.
        """
        stream = self.get_stream(index)
        try:
            if stream is not None:
                with stream:
                    yield stream
            else:
                yield None
        finally:
            self.release_stream(stream)

    def record_event(self, stream: Any, event_id: int) -> Any:
        """Record an event on a stream for synchronization.

        Args:
            stream: Stream to record event on.
            event_id: Unique identifier for the event.

        Returns:
            CUDA event or None.
        """
        if not self._gpu_available or stream is None:
            return None

        try:
            import cupy as cp

            event = cp.cuda.Event()
            event.record(stream)
            self._events[event_id] = event
            return event
        except Exception:
            return None

    def wait_event(self, stream: Any, event_id: int) -> None:
        """Make stream wait for an event.

        Args:
            stream: Stream that should wait.
            event_id: Event to wait for.
        """
        if stream is None or event_id not in self._events:
            return

        try:
            event = self._events[event_id]
            stream.wait_event(event)
        except Exception:
            pass

    def synchronize_all(self) -> None:
        """Synchronize all streams."""
        for stream in self._streams:
            try:
                stream.synchronize()
            except Exception:
                pass

    def synchronize_stream(self, stream: Any) -> None:
        """Synchronize a specific stream.

        Args:
            stream: Stream to synchronize.
        """
        if stream is not None:
            try:
                stream.synchronize()
            except Exception:
                pass


_stream_manager: GPUStreamManager | None = None


def get_stream_manager(n_streams: int = 4) -> GPUStreamManager:
    """Get or create global stream manager.

    Args:
        n_streams: Number of streams (only used on first call).

    Returns:
        GPUStreamManager instance.
    """
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = GPUStreamManager(n_streams=n_streams)
    return _stream_manager


class GPUKernels:
    """Collection of optimized CuPy kernels for performance-critical operations.

    Provides optimized GPU kernels with:
    - Fully vectorized operations (no Python loops)
    - Optimal block/grid sizing using occupancy API
    - Shared memory usage where beneficial
    - Proper NaN handling

    Attributes:
        use_gpu: Whether GPU is available and should be used.
        kernels_compiled: Set of kernel names that have been compiled.
    """

    def __init__(self, use_gpu: bool | None = None) -> None:
        """Initialize GPU kernels.

        Args:
            use_gpu: Whether to use GPU. If None, auto-detect.
        """
        self.use_gpu = use_gpu if use_gpu is not None else check_gpu_available()
        self.kernels_compiled: set[str] = set()
        self._kernels: dict[str, Any] = {}
        self._stream_manager: GPUStreamManager | None = None

        if self.use_gpu:
            self._compile_kernels()
            self._stream_manager = get_stream_manager()

    def _compile_kernels(self) -> None:
        """Compile all custom CUDA kernels."""
        try:
            import cupy as cp

            # Kernel 1: Vectorized feature constraint repair (min features)
            # Operates on entire population at once
            self._kernels["repair_min_features_vectorized"] = cp.RawKernel(
                r"""
                extern "C" __global__
                void repair_min_features_vectorized(
                    bool* features,
                    const int* n_selected,
                    const float* rand_vals,
                    const int pop_size,
                    const int n_features,
                    const int min_features
                ) {
                    int tid = blockDim.x * blockIdx.x + threadIdx.x;
                    int total_elements = pop_size * n_features;
                    if (tid >= total_elements) return;

                    int ind_idx = tid / n_features;
                    int feat_idx = tid % n_features;

                    int deficit = min_features - n_selected[ind_idx];
                    if (deficit <= 0) return;

                    // Only consider unselected features
                    if (!features[tid]) {
                        // Probability of selection proportional to deficit
                        float prob = (float)deficit / (float)(n_features - n_selected[ind_idx] + 1);
                        if (rand_vals[tid] < prob) {
                            features[tid] = true;
                        }
                    }
                }
                """,
                "repair_min_features_vectorized",
            )
            self.kernels_compiled.add("repair_min_features_vectorized")

            # Kernel 2: Vectorized feature constraint repair (max features)
            self._kernels["repair_max_features_vectorized"] = cp.RawKernel(
                r"""
                extern "C" __global__
                void repair_max_features_vectorized(
                    bool* features,
                    const int* n_selected,
                    const float* rand_vals,
                    const int pop_size,
                    const int n_features,
                    const int max_features
                ) {
                    int tid = blockDim.x * blockIdx.x + threadIdx.x;
                    int total_elements = pop_size * n_features;
                    if (tid >= total_elements) return;

                    int ind_idx = tid / n_features;
                    int feat_idx = tid % n_features;

                    int excess = n_selected[ind_idx] - max_features;
                    if (excess <= 0) return;

                    // Only consider selected features
                    if (features[tid]) {
                        // Probability of deselection proportional to excess
                        float prob = (float)excess / (float)n_selected[ind_idx];
                        if (rand_vals[tid] < prob) {
                            features[tid] = false;
                        }
                    }
                }
                """,
                "repair_max_features_vectorized",
            )
            self.kernels_compiled.add("repair_max_features_vectorized")

            # Kernel 3: Optimized tournament selection with shared memory
            self._kernels["tournament_selection_optimized"] = cp.RawKernel(
                r"""
                extern "C" __global__
                void tournament_selection_optimized(
                    const double* fitness,
                    const int* candidates,
                    int* winners,
                    const int n_tournaments,
                    const int tournament_size,
                    const int pop_size
                ) {
                    // Shared memory for fitness values within tournament
                    extern __shared__ double shared_fitness[];

                    int tid = blockDim.x * blockIdx.x + threadIdx.x;
                    if (tid >= n_tournaments) return;

                    int best_idx = candidates[tid * tournament_size];
                    double best_fit = fitness[best_idx];

                    // Unrolled loop for common tournament sizes
                    #pragma unroll 4
                    for (int i = 1; i < tournament_size; i++) {
                        int idx = candidates[tid * tournament_size + i];
                        double fit = fitness[idx];

                        // NaN-safe comparison: NaN is never better
                        bool is_better = !isnan(fit) && (isnan(best_fit) || fit > best_fit);
                        if (is_better) {
                            best_fit = fit;
                            best_idx = idx;
                        }
                    }

                    winners[tid] = best_idx;
                }
                """,
                "tournament_selection_optimized",
            )
            self.kernels_compiled.add("tournament_selection_optimized")

            # Kernel 4: Velocity clamping (elementwise)
            self._kernels["clamp_velocity"] = cp.ElementwiseKernel(
                "float64 velocity, float64 v_min, float64 v_max",
                "float64 clamped",
                """
                if (velocity < v_min) {
                    clamped = v_min;
                } else if (velocity > v_max) {
                    clamped = v_max;
                } else {
                    clamped = velocity;
                }
                """,
                "clamp_velocity_kernel",
            )
            self.kernels_compiled.add("clamp_velocity")

            # Kernel 5: Fitness comparison with NaN handling
            self._kernels["fitness_compare"] = cp.ElementwiseKernel(
                "float64 current, float64 personal_best",
                "bool is_better",
                """
                // Handle NaN: NaN is never better
                if (isnan(current)) {
                    is_better = false;
                } else if (isnan(personal_best)) {
                    is_better = true;
                } else {
                    is_better = current > personal_best;
                }
                """,
                "fitness_compare_kernel",
            )
            self.kernels_compiled.add("fitness_compare")

            # Kernel 6: Mutation mask generation
            self._kernels["mutation_mask"] = cp.ElementwiseKernel(
                "float32 rand, float32 mutation_rate",
                "bool should_mutate",
                "should_mutate = rand < mutation_rate;",
                "mutation_mask_kernel",
            )
            self.kernels_compiled.add("mutation_mask")

            # Kernel 7: Boundary reflection for PSO positions
            self._kernels["reflect_boundary"] = cp.ElementwiseKernel(
                "float64 position, float64 lower, float64 upper, float64 velocity",
                "float64 reflected",
                """
                float64 pos = position;

                // Reflect position back into bounds (max 10 iterations for safety)
                for (int iter = 0; iter < 10 && (pos < lower || pos > upper); iter++) {
                    if (pos < lower) {
                        pos = lower + (lower - pos);
                    }
                    if (pos > upper) {
                        pos = upper - (pos - upper);
                    }
                }
                // Final clamp for safety
                if (pos < lower) pos = lower;
                if (pos > upper) pos = upper;
                reflected = pos;
                """,
                "reflect_boundary_kernel",
            )
            self.kernels_compiled.add("reflect_boundary")

            # Kernel 8: Vectorized crossover mask generation
            self._kernels["crossover_mask"] = cp.RawKernel(
                r"""
                extern "C" __global__
                void crossover_mask(
                    bool* mask,
                    const float* rand_vals,
                    const int n_pairs,
                    const int n_genes,
                    const float crossover_rate
                ) {
                    int tid = blockDim.x * blockIdx.x + threadIdx.x;
                    int total = n_pairs * n_genes;
                    if (tid >= total) return;

                    // Each pair gets its own crossover decision
                    int pair_idx = tid / n_genes;
                    int gene_idx = tid % n_genes;

                    // Use first random value per pair for crossover decision
                    float pair_rand = rand_vals[pair_idx * n_genes];
                    if (pair_rand >= crossover_rate) {
                        mask[tid] = false;
                        return;
                    }

                    // Uniform crossover: each gene has 50% chance
                    mask[tid] = rand_vals[tid] < 0.5f;
                }
                """,
                "crossover_mask",
            )
            self.kernels_compiled.add("crossover_mask")

            # Kernel 9: GPU-side statistics computation
            self._kernels["compute_stats"] = cp.ReductionKernel(
                "float64 x, bool mask",
                "float64 y",
                "mask ? x : 0.0",
                "a + b",
                "y = a",
                "0.0",
                "masked_sum_kernel",
            )
            self.kernels_compiled.add("compute_stats")

            # Kernel 10: Parallel prefix sum for feature counting
            self._kernels["count_features"] = cp.ReductionKernel(
                "bool x",
                "int32 y",
                "x ? 1 : 0",
                "a + b",
                "y = a",
                "0",
                "count_true_kernel",
            )
            self.kernels_compiled.add("count_features")

            logger.info("Compiled %d GPU kernels", len(self.kernels_compiled))

        except ImportError:
            logger.warning("CuPy not available, GPU kernels disabled")
            self.use_gpu = False
        except Exception:
            logger.exception("Failed to compile GPU kernels")
            self.use_gpu = False

    def _get_optimal_block_size(
        self,
        kernel: Any,
        shared_mem_per_block: int = 0,
    ) -> int:
        """Get optimal block size using occupancy API.

        Args:
            kernel: CUDA kernel.
            shared_mem_per_block: Shared memory bytes per block.

        Returns:
            Optimal block size.
        """
        try:
            import cupy as cp

            # Use occupancy API if available (CuPy 9.0+)
            if hasattr(kernel, "max_dynamic_shared_size_bytes"):
                device = cp.cuda.Device()
                max_threads = device.attributes["MaxThreadsPerBlock"]
                return min(256, max_threads)
        except Exception:
            pass

        return 256

    def repair_feature_constraints(
        self,
        feature_genes: Any,
        min_features: int = 1,
        max_features: int | None = None,
        rng: Any | None = None,
    ) -> Any:
        """Repair feature selection constraints using fully vectorized GPU kernels.

        Ensures each individual has between min_features and max_features selected.
        All operations are fully vectorized - no Python loops over individuals.

        Args:
            feature_genes: Boolean array of shape (pop_size, n_features).
            min_features: Minimum features that must be selected.
            max_features: Maximum features allowed. None for no limit.
            rng: Random number generator (numpy or cupy).

        Returns:
            Repaired feature genes array.
        """
        xp = get_array_module(feature_genes)

        if not self.use_gpu or not is_gpu_array(feature_genes):
            return self._repair_features_cpu_vectorized(
                feature_genes, min_features, max_features, rng
            )

        import cupy as cp

        pop_size, n_features = feature_genes.shape
        repaired = feature_genes.astype(cp.bool_).copy()

        if rng is None:
            rng = cp.random.default_rng()

        n_selected = cp.sum(repaired, axis=1, dtype=cp.int32)

        if min_features > 0 and "repair_min_features_vectorized" in self.kernels_compiled:
            needs_repair = cp.any(n_selected < min_features)
            if needs_repair:
                for _ in range(min(min_features, 10)):
                    n_selected = cp.sum(repaired, axis=1, dtype=cp.int32)
                    if not cp.any(n_selected < min_features):
                        break

                    rand_vals = rng.random((pop_size, n_features), dtype=cp.float32)  # pyright: ignore[reportOptionalMemberAccess]
                    total_elements = pop_size * n_features
                    block_size = self._get_optimal_block_size(
                        self._kernels["repair_min_features_vectorized"]
                    )
                    grid_size = (total_elements + block_size - 1) // block_size

                    self._kernels["repair_min_features_vectorized"](
                        (grid_size,),
                        (block_size,),
                        (
                            repaired,
                            n_selected,
                            rand_vals,
                            np.int32(pop_size),
                            np.int32(n_features),
                            np.int32(min_features),
                        ),
                    )

        # Repair maximum constraints using vectorized kernel
        if (
            max_features is not None
            and "repair_max_features_vectorized" in self.kernels_compiled
        ):
            n_selected = cp.sum(repaired, axis=1, dtype=cp.int32)
            needs_repair = cp.any(n_selected > max_features)
            if needs_repair:
                for _ in range(min(n_features - max_features + 1, 10)):
                    n_selected = cp.sum(repaired, axis=1, dtype=cp.int32)
                    if not cp.any(n_selected > max_features):
                        break

                    rand_vals = rng.random((pop_size, n_features), dtype=cp.float32)  # pyright: ignore[reportOptionalMemberAccess]
                    total_elements = pop_size * n_features
                    block_size = self._get_optimal_block_size(
                        self._kernels["repair_max_features_vectorized"]
                    )
                    grid_size = (total_elements + block_size - 1) // block_size

                    self._kernels["repair_max_features_vectorized"](
                        (grid_size,),
                        (block_size,),
                        (
                            repaired,
                            n_selected,
                            rand_vals,
                            np.int32(pop_size),
                            np.int32(n_features),
                            np.int32(max_features),
                        ),
                    )

        return repaired

    def _repair_features_cpu_vectorized(
        self,
        feature_genes: NDArray[np.bool_],
        min_features: int,
        max_features: int | None,
        rng: np.random.Generator | None,
    ) -> NDArray[np.bool_]:
        """CPU fallback for feature constraint repair using vectorized operations.

        Uses NumPy vectorization to minimize Python loops.

        Args:
            feature_genes: Boolean array of shape (pop_size, n_features).
            min_features: Minimum features required.
            max_features: Maximum features allowed.
            rng: Random number generator.

        Returns:
            Repaired feature genes.
        """
        if rng is None:
            rng = np.random.default_rng()

        repaired = np.asarray(feature_genes, dtype=np.bool_).copy()
        pop_size, n_features = repaired.shape

        n_selected = np.sum(repaired, axis=1)

        below_min_mask = n_selected < min_features
        if np.any(below_min_mask):
            below_min_indices = np.where(below_min_mask)[0]

            for idx in below_min_indices:
                deficit = min_features - n_selected[idx]
                unselected = np.where(~repaired[idx])[0]
                if len(unselected) >= deficit:
                    to_select = rng.choice(unselected, size=deficit, replace=False)
                    repaired[idx, to_select] = True

        if max_features is not None:
            n_selected = np.sum(repaired, axis=1)
            above_max_mask = n_selected > max_features
            if np.any(above_max_mask):
                above_max_indices = np.where(above_max_mask)[0]

                for idx in above_max_indices:
                    excess = n_selected[idx] - max_features
                    selected = np.where(repaired[idx])[0]
                    if len(selected) >= excess:
                        to_deselect = rng.choice(selected, size=int(excess), replace=False)
                        repaired[idx, to_deselect] = False

        return repaired

    def clamp_velocities(
        self,
        velocities: Any,
        v_min: float | Any,
        v_max: float | Any,
    ) -> Any:
        """Clamp PSO velocities to bounds using GPU kernel.

        Args:
            velocities: Velocity array of shape (swarm_size, n_dims).
            v_min: Minimum velocity (scalar or per-dimension array).
            v_max: Maximum velocity (scalar or per-dimension array).

        Returns:
            Clamped velocities.
        """
        xp = get_array_module(velocities)

        if self.use_gpu and is_gpu_array(velocities):
            if "clamp_velocity" in self.kernels_compiled:
                # Broadcast bounds if scalar
                if np.isscalar(v_min):
                    v_min = xp.full_like(velocities, v_min)
                if np.isscalar(v_max):
                    v_max = xp.full_like(velocities, v_max)

                return self._kernels["clamp_velocity"](velocities, v_min, v_max)

        return xp.clip(velocities, v_min, v_max)

    def compare_fitness(
        self,
        current_fitness: Any,
        personal_best_fitness: Any,
    ) -> Any:
        """Compare fitness values with NaN handling using GPU kernel.

        Args:
            current_fitness: Current fitness values.
            personal_best_fitness: Personal best fitness values.

        Returns:
            Boolean array where True indicates current is better.
        """
        xp = get_array_module(current_fitness)

        if self.use_gpu and is_gpu_array(current_fitness):
            if "fitness_compare" in self.kernels_compiled:
                return self._kernels["fitness_compare"](
                    current_fitness, personal_best_fitness
                )

        # CPU fallback with NaN handling
        with np.errstate(invalid="ignore"):
            is_better = current_fitness > personal_best_fitness
            # NaN in current is never better
            is_better = is_better & ~np.isnan(current_fitness)
            # NaN in personal best means current is better (if not NaN)
            is_better = is_better | (
                np.isnan(personal_best_fitness) & ~np.isnan(current_fitness)
            )
        return is_better

    def generate_mutation_mask(
        self,
        shape: tuple[int, ...],
        mutation_rate: float,
        rng: Any | None = None,
    ) -> Any:
        """Generate mutation mask using GPU kernel.

        Args:
            shape: Shape of the mask array.
            mutation_rate: Probability of mutation at each position.
            rng: Random number generator.

        Returns:
            Boolean mask array.
        """
        if self.use_gpu:
            try:
                import cupy as cp

                if rng is None:
                    rng = cp.random.default_rng()

                rand = rng.random(shape, dtype=cp.float32)  # pyright: ignore[reportOptionalMemberAccess]

                if "mutation_mask" in self.kernels_compiled:
                    return self._kernels["mutation_mask"](rand, cp.float32(mutation_rate))
                return rand < mutation_rate

            except ImportError:
                pass

        if rng is None:
            rng = np.random.default_rng()
        return rng.random(shape) < mutation_rate

    def reflect_positions(
        self,
        positions: Any,
        lower_bounds: Any,
        upper_bounds: Any,
        velocities: Any | None = None,
    ) -> Any:
        """Reflect positions back into bounds using GPU kernel.

        Args:
            positions: Position array of shape (n_particles, n_dims).
            lower_bounds: Lower bounds per dimension.
            upper_bounds: Upper bounds per dimension.
            velocities: Optional velocities (for compatibility).

        Returns:
            Reflected positions.
        """
        xp = get_array_module(positions)

        if self.use_gpu and is_gpu_array(positions):
            if "reflect_boundary" in self.kernels_compiled:
                if velocities is None:
                    velocities = xp.zeros_like(positions)
                return self._kernels["reflect_boundary"](
                    positions, lower_bounds, upper_bounds, velocities
                )

        reflected = ensure_numpy(positions).copy()
        lower = ensure_numpy(lower_bounds)
        upper = ensure_numpy(upper_bounds)

        for _ in range(10):
            below = reflected < lower
            above = reflected > upper
            if not np.any(below | above):
                break
            reflected = np.where(below, 2 * lower - reflected, reflected)
            reflected = np.where(above, 2 * upper - reflected, reflected)

        reflected = np.clip(reflected, lower, upper)

        if is_gpu_array(positions):
            import cupy as cp

            return cp.asarray(reflected)
        return reflected

    def tournament_selection(
        self,
        fitness: Any,
        n_select: int,
        tournament_size: int = 3,
        rng: Any | None = None,
    ) -> Any:
        """Perform tournament selection using optimized GPU kernel.

        Uses shared memory and optimal block sizing for efficiency.

        Args:
            fitness: Fitness array of shape (pop_size,).
            n_select: Number of individuals to select.
            tournament_size: Number of candidates per tournament.
            rng: Random number generator.

        Returns:
            Indices of selected individuals.
        """
        xp = get_array_module(fitness)
        pop_size = len(fitness)

        if self.use_gpu and is_gpu_array(fitness):
            try:
                import cupy as cp

                if rng is None:
                    rng = cp.random.default_rng()

                if "tournament_selection_optimized" in self.kernels_compiled:
                    candidates = rng.integers(  # pyright: ignore[reportOptionalMemberAccess]
                        0, pop_size, size=(n_select, tournament_size), dtype=cp.int32
                    )
                    winners = cp.zeros(n_select, dtype=cp.int32)

                    block_size = self._get_optimal_block_size(
                        self._kernels["tournament_selection_optimized"]
                    )
                    grid_size = (n_select + block_size - 1) // block_size

                    shared_mem = tournament_size * 8  # sizeof(double)

                    self._kernels["tournament_selection_optimized"](
                        (grid_size,),
                        (block_size,),
                        (
                            fitness.astype(cp.float64),
                            candidates.ravel(),
                            winners,
                            np.int32(n_select),
                            np.int32(tournament_size),
                            np.int32(pop_size),
                        ),
                        shared_mem=shared_mem,
                    )
                    return winners

            except ImportError:
                pass

        fitness_np = ensure_numpy(fitness)
        if rng is None:
            rng = np.random.default_rng()

        candidates = rng.integers(0, pop_size, size=(n_select, tournament_size))
        candidate_fitness = fitness_np[candidates]
        winner_in_tournament = np.argmax(candidate_fitness, axis=1)
        winners = candidates[np.arange(n_select), winner_in_tournament]

        if is_gpu_array(fitness):
            import cupy as cp

            return cp.asarray(winners)
        return winners

    def compute_fitness_statistics_gpu(
        self,
        fitness: Any,
        mask: Any | None = None,
    ) -> dict[str, float]:
        """Compute fitness statistics entirely on GPU, transferring only scalars.

        Args:
            fitness: Fitness array.
            mask: Optional boolean mask for valid entries.

        Returns:
            Dictionary with min, max, mean, std, median statistics.
        """
        if not self.use_gpu or not is_gpu_array(fitness):
            fitness_np = ensure_numpy(fitness)
            if mask is not None:
                mask_np = ensure_numpy(mask)
                valid = fitness_np[mask_np]
            else:
                valid = fitness_np[np.isfinite(fitness_np)]

            if len(valid) == 0:
                return {
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                    "std": 0.0,
                    "median": 0.0,
                    "n_valid": 0,
                }

            return {
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid)),
                "median": float(np.median(valid)),
                "n_valid": len(valid),
            }

        import cupy as cp

        if mask is None:
            mask = cp.isfinite(fitness)

        n_valid = int(cp.sum(mask))
        if n_valid == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "n_valid": 0,
            }

        valid_fitness = fitness[mask]

        return {
            "min": float(cp.min(valid_fitness)),
            "max": float(cp.max(valid_fitness)),
            "mean": float(cp.mean(valid_fitness)),
            "std": float(cp.std(valid_fitness)),
            "median": float(cp.median(valid_fitness)),
            "n_valid": n_valid,
        }

    def generate_crossover_mask(
        self,
        n_pairs: int,
        n_genes: int,
        crossover_rate: float,
        rng: Any | None = None,
    ) -> Any:
        """Generate crossover mask using GPU kernel.

        Args:
            n_pairs: Number of parent pairs.
            n_genes: Number of genes per individual.
            crossover_rate: Probability of crossover.
            rng: Random number generator.

        Returns:
            Boolean mask of shape (n_pairs, n_genes).
        """
        if self.use_gpu and "crossover_mask" in self.kernels_compiled:
            try:
                import cupy as cp

                if rng is None:
                    rng = cp.random.default_rng()

                mask = cp.zeros((n_pairs, n_genes), dtype=cp.bool_)
                rand_vals = rng.random((n_pairs, n_genes), dtype=cp.float32)  # pyright: ignore[reportOptionalMemberAccess]

                total = n_pairs * n_genes
                block_size = 256
                grid_size = (total + block_size - 1) // block_size

                self._kernels["crossover_mask"](
                    (grid_size,),
                    (block_size,),
                    (
                        mask,
                        rand_vals,
                        np.int32(n_pairs),
                        np.int32(n_genes),
                        np.float32(crossover_rate),
                    ),
                )
                return mask

            except ImportError:
                pass

        if rng is None:
            rng = np.random.default_rng()

        do_crossover = rng.random(n_pairs) < crossover_rate
        gene_mask = rng.random((n_pairs, n_genes)) < 0.5
        mask = gene_mask & do_crossover[:, np.newaxis]
        return mask

    def get_compiled_kernels(self) -> list[str]:
        """Get list of compiled kernel names.

        Returns:
            List of compiled kernel names.
        """
        return list(self.kernels_compiled)


@dataclass
class MemoryStats:
    """GPU memory statistics.

    Attributes:
        used_bytes: Currently used memory in bytes.
        total_bytes: Total allocated memory in bytes.
        free_bytes: Available memory in bytes.
        peak_bytes: Peak memory usage in bytes.
        n_allocations: Number of active allocations.
    """

    used_bytes: int
    total_bytes: int
    free_bytes: int
    peak_bytes: int = 0
    n_allocations: int = 0

    def format(self, unit: MemoryUnit = MemoryUnit.MB) -> str:
        """Format memory stats as string.

        Args:
            unit: Memory unit for display.

        Returns:
            Formatted string.
        """
        factor = MEMORY_FACTORS[unit]
        unit_name = unit.name
        return (
            f"Memory: {self.used_bytes / factor:.2f} {unit_name} used, "
            f"{self.free_bytes / factor:.2f} {unit_name} free, "
            f"{self.total_bytes / factor:.2f} {unit_name} total"
        )


@dataclass
class ArrayBuffer:
    """Pre-allocated array buffer for reuse.

    Attributes:
        array: The allocated array.
        shape: Shape of the array.
        dtype: Data type of the array.
        in_use: Whether buffer is currently in use.
        last_used: Timestamp of last use.
    """

    array: Any
    shape: tuple[int, ...]
    dtype: np.dtype
    in_use: bool = False
    last_used: float = field(default_factory=time.time)


class GPUMemoryManager:
    """GPU memory management with pooling and monitoring.

    Provides utilities for:
    - Memory pool management
    - Pre-allocation and buffer reuse
    - Memory monitoring and statistics
    - Automatic cleanup

    Attributes:
        memory_limit_fraction: Fraction of GPU memory to use (0.0-1.0).
        pool_enabled: Whether memory pooling is enabled.
        track_allocations: Whether to track individual allocations.
    """

    def __init__(
        self,
        memory_limit_fraction: float = DEFAULT_MEMORY_LIMIT_FRACTION,
        pool_enabled: bool = True,
        track_allocations: bool = False,
    ) -> None:
        """Initialize GPU memory manager.

        Args:
            memory_limit_fraction: Maximum fraction of GPU memory to use.
            pool_enabled: Enable CuPy memory pooling.
            track_allocations: Track individual allocations (slower but useful for debugging).
        """
        self.memory_limit_fraction = memory_limit_fraction
        self.pool_enabled = pool_enabled
        self.track_allocations = track_allocations

        self._initialized = False
        self._gpu_available = check_gpu_available()
        self._peak_usage: int = 0
        self._allocation_history: deque[tuple[float, int]] = deque(maxlen=1000)
        self._buffers: dict[str, ArrayBuffer] = {}
        self._buffer_refs: weakref.WeakValueDictionary[str, Any] = (
            weakref.WeakValueDictionary()
        )

    def initialize(self) -> bool:
        """Initialize GPU memory management.

        Sets up memory pool and configures limits.

        Returns:
            True if initialization successful.
        """
        if not self._gpu_available:
            logger.warning("GPU not available, memory manager disabled")
            return False

        try:
            import cupy as cp

            if self.pool_enabled:
                mempool = cp.get_default_memory_pool()
                pinned_mempool = cp.get_default_pinned_memory_pool()  # noqa: F841

                device = cp.cuda.Device()
                total_memory = device.mem_info[1]
                memory_limit = int(total_memory * self.memory_limit_fraction)

                try:
                    mempool.set_limit(size=memory_limit)
                    logger.info(
                        "GPU memory pool limit set to %.2f GB",
                        memory_limit / 1e9,
                    )
                except AttributeError:
                    logger.warning("CuPy version does not support memory limits")

            self._initialized = True
            logger.info("GPU memory manager initialized")
            return True

        except ImportError:
            logger.warning("CuPy not available")
            return False
        except Exception:
            logger.exception("Failed to initialize GPU memory manager")
            return False

    def get_stats(self) -> MemoryStats:
        """Get current GPU memory statistics.

        Returns:
            MemoryStats object with current usage information.
        """
        if not self._gpu_available:
            return MemoryStats(used_bytes=0, total_bytes=0, free_bytes=0, peak_bytes=0)

        try:
            import cupy as cp

            mempool = cp.get_default_memory_pool()

            used = mempool.used_bytes()
            total = mempool.total_bytes()

            # Update peak tracking
            self._peak_usage = max(self._peak_usage, used)

            # Track allocation history
            if self.track_allocations:
                self._allocation_history.append((time.time(), used))

            return MemoryStats(
                used_bytes=used,
                total_bytes=total,
                free_bytes=total - used,
                peak_bytes=self._peak_usage,
                n_allocations=len(self._buffers),
            )

        except Exception:
            return MemoryStats(used_bytes=0, total_bytes=0, free_bytes=0)

    def free_unused(self) -> int:
        """Free unused GPU memory blocks.

        Returns:
            Number of bytes freed.
        """
        if not self._gpu_available:
            return 0

        try:
            import cupy as cp

            mempool = cp.get_default_memory_pool()
            used_before = mempool.used_bytes()
            mempool.free_all_blocks()
            used_after = mempool.used_bytes()

            freed = used_before - used_after
            if freed > 0:
                logger.debug("Freed %.2f MB of GPU memory", freed / 1e6)
            return freed

        except Exception:
            return 0

    def free_pinned(self) -> int:
        """Free pinned (page-locked) memory.

        Returns:
            Number of bytes freed.
        """
        if not self._gpu_available:
            return 0

        try:
            import cupy as cp

            pinned_mempool = cp.get_default_pinned_memory_pool()
            n_blocks = pinned_mempool.n_free_blocks()
            pinned_mempool.free_all_blocks()

            logger.debug("Freed %d pinned memory blocks", n_blocks)
            return n_blocks

        except Exception:
            return 0

    def allocate_buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype | type = np.float64,
        fill_value: float | None = None,
    ) -> Any:
        """Allocate or retrieve a named buffer.

        Buffers are reused across calls if shape and dtype match.

        Args:
            name: Unique name for the buffer.
            shape: Shape of the array.
            dtype: Data type.
            fill_value: Optional value to fill array with.

        Returns:
            Allocated array (GPU if available).
        """
        dtype = np.dtype(dtype)
        buffer_key = f"{name}_{shape}_{dtype}"

        if buffer_key in self._buffers:
            buffer = self._buffers[buffer_key]
            if buffer.shape == shape and buffer.dtype == dtype:
                buffer.in_use = True
                buffer.last_used = time.time()
                if fill_value is not None:
                    buffer.array.fill(fill_value)
                return buffer.array

        if self._gpu_available:
            try:
                import cupy as cp

                if fill_value is not None:
                    arr = cp.full(shape, fill_value, dtype=dtype)
                else:
                    arr = cp.empty(shape, dtype=dtype)

            except (ImportError, Exception):
                logger.warning("GPU allocation failed, falling back to CPU")
                if fill_value is not None:
                    arr = np.full(shape, fill_value, dtype=dtype)
                else:
                    arr = np.empty(shape, dtype=dtype)
        else:
            if fill_value is not None:
                arr = np.full(shape, fill_value, dtype=dtype)
            else:
                arr = np.empty(shape, dtype=dtype)

        self._buffers[buffer_key] = ArrayBuffer(
            array=arr,
            shape=shape,
            dtype=dtype,
            in_use=True,
            last_used=time.time(),
        )

        return arr

    def release_buffer(self, name: str) -> None:
        """Release a named buffer for reuse.

        Args:
            name: Name of the buffer to release.
        """
        for key, buffer in self._buffers.items():
            if key.startswith(f"{name}_"):
                buffer.in_use = False
                break

    def clear_buffers(self) -> None:
        """Clear all buffers and free memory."""
        self._buffers.clear()
        self._buffer_refs.clear()
        self.free_unused()
        self.free_pinned()

    def check_memory_available(self, required_bytes: int) -> bool:
        """Check if sufficient GPU memory is available.

        Args:
            required_bytes: Required memory in bytes.

        Returns:
            True if enough memory is available.
        """
        if not self._gpu_available:
            return True

        stats = self.get_stats()
        return stats.free_bytes >= required_bytes

    def estimate_array_size(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype | type = np.float64,
    ) -> int:
        """Estimate memory requirement for an array.

        Args:
            shape: Shape of the array.
            dtype: Data type.

        Returns:
            Estimated size in bytes.
        """
        dtype = np.dtype(dtype)
        n_elements = int(np.prod(shape))
        return n_elements * dtype.itemsize

    @contextlib.contextmanager
    def memory_scope(self, name: str = "unnamed") -> Generator[MemoryStats, None, None]:
        """Context manager for tracking memory usage in a scope.

        Args:
            name: Name for logging.

        Yields:
            Initial memory stats.
        """
        initial_stats = self.get_stats()
        logger.debug("Memory scope '%s' started: %s", name, initial_stats.format())

        try:
            yield initial_stats
        finally:
            final_stats = self.get_stats()
            delta = final_stats.used_bytes - initial_stats.used_bytes
            logger.debug(
                "Memory scope '%s' ended: delta=%.2f MB",
                name,
                delta / 1e6,
            )

    def get_allocation_history(self) -> list[tuple[float, int]]:
        """Get allocation history for analysis.

        Returns:
            List of (timestamp, used_bytes) tuples.
        """
        return list(self._allocation_history)


class TransferOptimizer:
    """Optimizes CPU-GPU data transfers.

    Provides utilities for:
    - Pinned memory allocation for faster transfers
    - Asynchronous transfers with stream management
    - Transfer batching
    - Overlapped computation and transfers

    Attributes:
        use_gpu: Whether GPU is available.
        use_pinned: Whether to use pinned memory.
        use_async: Whether to use async transfers.
    """

    def __init__(
        self,
        use_pinned: bool = True,
        use_async: bool = True,
        chunk_size: int = DEFAULT_TRANSFER_CHUNK_SIZE,
    ) -> None:
        """Initialize transfer optimizer.

        Args:
            use_pinned: Use pinned (page-locked) memory for transfers.
            use_async: Use asynchronous transfers when possible.
            chunk_size: Chunk size for large transfers.
        """
        self.use_pinned = use_pinned
        self.use_async = use_async
        self.chunk_size = chunk_size

        self._gpu_available = check_gpu_available()
        self._pinned_buffers: dict[str, Any] = {}
        self._stream_manager = get_stream_manager() if self._gpu_available else None
        self._transfer_stats = {
            "n_transfers": 0,
            "total_bytes": 0,
            "total_time": 0.0,
        }

    def allocate_pinned(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype | type = np.float64,
    ) -> NDArray:
        """Allocate pinned (page-locked) memory.

        Pinned memory enables faster CPU-GPU transfers.

        Args:
            shape: Shape of the array.
            dtype: Data type.

        Returns:
            NumPy array backed by pinned memory.
        """
        dtype = np.dtype(dtype)
        n_bytes = int(np.prod(shape)) * dtype.itemsize

        if self._gpu_available and self.use_pinned:
            try:
                import cupy as cp

                mem = cp.cuda.alloc_pinned_memory(n_bytes)
                arr = np.frombuffer(mem, dtype=dtype).reshape(shape)
                return arr

            except Exception:
                logger.debug("Pinned memory allocation failed, using regular memory")

        return np.empty(shape, dtype=dtype)

    def to_gpu_async(
        self,
        arr: NDArray,
        stream: Any | None = None,
    ) -> tuple[Any, Any]:
        """Transfer array to GPU asynchronously.

        Args:
            arr: NumPy array to transfer.
            stream: CUDA stream for async operation. If None, gets from pool.

        Returns:
            Tuple of (GPU array, stream used).
        """
        if not self._gpu_available:
            return arr, None

        try:
            import cupy as cp

            if stream is None and self._stream_manager is not None:
                stream = self._stream_manager.get_stream()

            start_time = time.perf_counter()

            if stream is not None:
                with stream:
                    if not arr.flags.c_contiguous:
                        arr = np.ascontiguousarray(arr)
                    gpu_arr = cp.asarray(arr)
            else:
                if not arr.flags.c_contiguous:
                    arr = np.ascontiguousarray(arr)
                gpu_arr = cp.asarray(arr)

            self._transfer_stats["n_transfers"] += 1
            self._transfer_stats["total_bytes"] += arr.nbytes
            self._transfer_stats["total_time"] += time.perf_counter() - start_time

            return gpu_arr, stream

        except Exception:
            return arr, None

    def to_cpu_async(
        self,
        arr: Any,
        out: NDArray | None = None,
        stream: Any | None = None,
    ) -> tuple[NDArray, Any]:
        """Transfer array from GPU to CPU asynchronously.

        Args:
            arr: GPU array to transfer.
            out: Optional pre-allocated output array (preferably pinned).
            stream: CUDA stream for async operation.

        Returns:
            Tuple of (CPU array, stream used).
        """
        if not is_gpu_array(arr):
            return ensure_numpy(arr), None

        try:
            import cupy as cp

            if stream is None and self._stream_manager is not None:
                stream = self._stream_manager.get_stream()

            start_time = time.perf_counter()

            if stream is not None:
                with stream:
                    if out is not None:
                        arr.get(out=out)
                        cpu_arr = out
                    else:
                        cpu_arr = arr.get()
            else:
                if out is not None:
                    arr.get(out=out)
                    cpu_arr = out
                else:
                    cpu_arr = arr.get()

            # Update stats
            self._transfer_stats["n_transfers"] += 1
            self._transfer_stats["total_bytes"] += arr.nbytes
            self._transfer_stats["total_time"] += time.perf_counter() - start_time

            return cpu_arr, stream

        except Exception:
            return ensure_numpy(arr), None

    def synchronize(self, stream: Any = None) -> None:
        """Synchronize CUDA stream or device.

        Args:
            stream: Stream to synchronize. If None, synchronizes device.
        """
        if not self._gpu_available:
            return

        try:
            import cupy as cp

            if stream is not None:
                stream.synchronize()
            else:
                cp.cuda.Device().synchronize()

        except Exception:
            pass

    def create_stream(self, non_blocking: bool = True) -> Any:
        """Create a new CUDA stream.

        Args:
            non_blocking: If True, stream doesn't synchronize with default stream.

        Returns:
            CUDA stream or None if GPU not available.
        """
        if self._stream_manager is not None:
            return self._stream_manager.get_stream()
        return None

    @contextlib.contextmanager
    def stream_context(self, stream: Any = None) -> Generator[Any, None, None]:
        """Context manager for CUDA stream operations.

        Args:
            stream: Stream to use. If None, gets from pool.

        Yields:
            CUDA stream.
        """
        if self._stream_manager is not None:
            with self._stream_manager.stream_context() as s:
                yield s
        else:
            yield None

    def batch_transfer_to_gpu(
        self,
        arrays: list[NDArray],
        concurrent: bool = True,
    ) -> list[Any]:
        """Transfer multiple arrays to GPU, optionally in parallel.

        Args:
            arrays: List of NumPy arrays.
            concurrent: If True, use multiple streams for concurrent transfer.

        Returns:
            List of GPU arrays.
        """
        if not self._gpu_available:
            return arrays

        n_arrays = len(arrays)
        gpu_arrays = []
        streams = []

        for i, arr in enumerate(arrays):
            # Use different streams for concurrent transfer
            stream_idx = i % 4 if concurrent and self._stream_manager else None
            stream = (
                self._stream_manager.get_stream(stream_idx)
                if self._stream_manager
                else None
            )
            gpu_arr, s = self.to_gpu_async(arr, stream)
            gpu_arrays.append(gpu_arr)
            streams.append(s)

        # Synchronize all streams
        if concurrent:
            for stream in streams:
                if stream is not None:
                    self.synchronize(stream)
                    if self._stream_manager:
                        self._stream_manager.release_stream(stream)

        return gpu_arrays

    def batch_transfer_to_cpu(
        self,
        arrays: list[Any],
        out_arrays: list[NDArray] | None = None,
        concurrent: bool = True,
    ) -> list[NDArray]:
        """Transfer multiple arrays from GPU to CPU.

        Args:
            arrays: List of GPU arrays.
            out_arrays: Optional pre-allocated output arrays.
            concurrent: If True, use multiple streams.

        Returns:
            List of CPU arrays.
        """
        if out_arrays is None:
            out_arrays = [None] * len(arrays)  # pyright: ignore[reportAssignmentType]

        cpu_arrays = []
        streams = []

        for i, (arr, out) in enumerate(zip(arrays, out_arrays)):  # pyright: ignore[reportGeneralTypeIssues, reportArgumentType]
            stream_idx = i % 4 if concurrent and self._stream_manager else None
            stream = (
                self._stream_manager.get_stream(stream_idx)
                if self._stream_manager
                else None
            )
            cpu_arr, s = self.to_cpu_async(arr, out, stream)
            cpu_arrays.append(cpu_arr)
            streams.append(s)

        # Synchronize all
        if concurrent:
            for stream in streams:
                if stream is not None:
                    self.synchronize(stream)
                    if self._stream_manager:
                        self._stream_manager.release_stream(stream)

        return cpu_arrays

    def overlap_transfer_compute(
        self,
        data_to_transfer: NDArray,
        compute_fn: Callable[[Any], Any],
        existing_gpu_data: Any,
    ) -> tuple[Any, Any]:
        """Overlap data transfer with computation.

        Transfers new data while computing on existing GPU data.

        Args:
            data_to_transfer: New data to transfer to GPU.
            compute_fn: Function to run on existing GPU data.
            existing_gpu_data: Data already on GPU.

        Returns:
            Tuple of (transferred data on GPU, computation result).
        """
        if not self._gpu_available or self._stream_manager is None:
            gpu_data, _ = self.to_gpu_async(data_to_transfer)
            result = compute_fn(existing_gpu_data)
            return gpu_data, result

        transfer_stream = self._stream_manager.get_stream(0)
        compute_stream = self._stream_manager.get_stream(1)

        try:
            import cupy as cp

            with transfer_stream:
                gpu_data = cp.asarray(data_to_transfer)

            with compute_stream:
                result = compute_fn(existing_gpu_data)

            self._stream_manager.synchronize_stream(transfer_stream)
            self._stream_manager.synchronize_stream(compute_stream)

            return gpu_data, result

        except Exception:
            gpu_data, _ = self.to_gpu_async(data_to_transfer)
            result = compute_fn(existing_gpu_data)
            return gpu_data, result

        finally:
            self._stream_manager.release_stream(transfer_stream)
            self._stream_manager.release_stream(compute_stream)

    def get_transfer_stats(self) -> dict[str, Any]:
        """Get transfer statistics.

        Returns:
            Dictionary with transfer statistics.
        """
        stats = self._transfer_stats.copy()
        if stats["n_transfers"] > 0:
            stats["avg_bytes"] = stats["total_bytes"] / stats["n_transfers"]
            stats["avg_time"] = stats["total_time"] / stats["n_transfers"]
            if stats["total_time"] > 0:
                stats["bandwidth_MBps"] = stats["total_bytes"] / stats["total_time"] / 1e6
        return stats

    def reset_stats(self) -> None:
        """Reset transfer statistics."""
        self._transfer_stats = {
            "n_transfers": 0,
            "total_bytes": 0,
            "total_time": 0.0,
        }

    def cleanup(self) -> None:
        """Clean up streams and pinned buffers."""
        if self._stream_manager is not None:
            self._stream_manager.synchronize_all()
        self._pinned_buffers.clear()


@dataclass
class TimingRecord:
    """Record for a single timing measurement.

    Attributes:
        name: Name of the operation.
        start_time: Start timestamp.
        end_time: End timestamp.
        duration: Duration in seconds.
        metadata: Additional metadata.
    """

    name: str
    start_time: float
    end_time: float
    duration: float
    metadata: dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """Performance profiler for tracking operation timing.

    Provides utilities for:
    - Timing individual operations
    - Tracking cumulative statistics
    - GPU synchronization for accurate timing
    - Hierarchical timing scopes

    Example:
        >>> profiler = PerformanceProfiler()
        >>> with profiler.scope("training"):
        ...     model.fit(X, y)
        >>> print(profiler.get_stats())
    """

    def __init__(
        self,
        enable_gpu_sync: bool = True,
        track_memory: bool = False,
    ) -> None:
        """Initialize profiler.

        Args:
            enable_gpu_sync: Synchronize GPU before timing for accuracy.
            track_memory: Also track memory usage.
        """
        self.enable_gpu_sync = enable_gpu_sync
        self.track_memory = track_memory

        self._records: list[TimingRecord] = []
        self._stats: dict[str, dict[str, float]] = {}
        self._active_scopes: list[tuple[str, float]] = []
        self._gpu_available = check_gpu_available()
        self._memory_manager = GPUMemoryManager() if track_memory else None

    def _sync_gpu(self) -> None:
        """Synchronize GPU for accurate timing."""
        if self.enable_gpu_sync and self._gpu_available:
            try:
                import cupy as cp

                cp.cuda.Device().synchronize()
            except Exception:
                pass

    def start(self, name: str) -> float:
        """Start timing an operation.

        Args:
            name: Name of the operation.

        Returns:
            Start timestamp.
        """
        self._sync_gpu()
        start_time = time.perf_counter()
        self._active_scopes.append((name, start_time))
        return start_time

    def stop(self, name: str | None = None) -> float:
        """Stop timing an operation.

        Args:
            name: Name of the operation (must match start). If None, stops most recent.

        Returns:
            Duration in seconds.
        """
        self._sync_gpu()
        end_time = time.perf_counter()

        if not self._active_scopes:
            logger.warning("No active timing scope to stop")
            return 0.0

        if name is None:
            scope_name, start_time = self._active_scopes.pop()
        else:
            for i in range(len(self._active_scopes) - 1, -1, -1):
                if self._active_scopes[i][0] == name:
                    scope_name, start_time = self._active_scopes.pop(i)
                    break
            else:
                logger.warning("No matching scope found for '%s'", name)
                return 0.0

        duration = end_time - start_time

        record = TimingRecord(
            name=scope_name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
        )
        self._records.append(record)

        if scope_name not in self._stats:
            self._stats[scope_name] = {
                "count": 0,
                "total": 0.0,
                "min": float("inf"),
                "max": 0.0,
            }
        stats = self._stats[scope_name]
        stats["count"] += 1
        stats["total"] += duration
        stats["min"] = min(stats["min"], duration)
        stats["max"] = max(stats["max"], duration)

        return duration

    @contextlib.contextmanager
    def scope(self, name: str) -> Generator[None, None, None]:
        """Context manager for timing a scope.

        Args:
            name: Name of the scope.

        Yields:
            None
        """
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def time_function(self, name: str | None = None) -> Callable:
        """Decorator to time a function.

        Args:
            name: Custom name for timing. If None, uses function name.

        Returns:
            Decorator function.
        """

        def decorator(func: Callable) -> Callable:
            timing_name = name or func.__name__

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.scope(timing_name):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Get timing statistics.

        Returns:
            Dictionary mapping operation names to statistics.
        """
        result = {}
        for name, stats in self._stats.items():
            result[name] = {
                "count": stats["count"],
                "total": stats["total"],
                "mean": stats["total"] / stats["count"] if stats["count"] > 0 else 0.0,
                "min": stats["min"] if stats["min"] != float("inf") else 0.0,
                "max": stats["max"],
            }
        return result

    def get_records(self) -> list[TimingRecord]:
        """Get all timing records.

        Returns:
            List of TimingRecord objects.
        """
        return list(self._records)

    def summary(self, top_n: int = 10) -> str:
        """Generate summary of timing statistics.

        Args:
            top_n: Number of top operations to show.

        Returns:
            Formatted summary string.
        """
        stats = self.get_stats()
        if not stats:
            return "No timing data recorded"

        sorted_stats = sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True)

        lines = ["Performance Summary:", "-" * 60]
        for name, data in sorted_stats[:top_n]:
            lines.append(
                f"{name:30s} | count={data['count']:5d} | "
                f"total={data['total']:.4f}s | mean={data['mean']:.4f}s"
            )

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all timing data."""
        self._records.clear()
        self._stats.clear()
        self._active_scopes.clear()

    def __enter__(self) -> PerformanceProfiler:
        """Enter context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager."""
        pass


def create_memory_manager(
    memory_limit_fraction: float = DEFAULT_MEMORY_LIMIT_FRACTION,
    **kwargs: Any,
) -> GPUMemoryManager:
    """Create and initialize a GPU memory manager.

    Args:
        memory_limit_fraction: Fraction of GPU memory to use.
        **kwargs: Additional arguments for GPUMemoryManager.

    Returns:
        Initialized GPUMemoryManager.
    """
    manager = GPUMemoryManager(memory_limit_fraction=memory_limit_fraction, **kwargs)
    manager.initialize()
    return manager


def create_transfer_optimizer(**kwargs: Any) -> TransferOptimizer:
    """Create a transfer optimizer.

    Args:
        **kwargs: Arguments for TransferOptimizer.

    Returns:
        TransferOptimizer instance.
    """
    return TransferOptimizer(**kwargs)


def create_gpu_kernels(use_gpu: bool | None = None) -> GPUKernels:
    """Create GPU kernels instance.

    Args:
        use_gpu: Whether to use GPU. If None, auto-detect.

    Returns:
        GPUKernels instance.
    """
    return GPUKernels(use_gpu=use_gpu)


@contextlib.contextmanager
def performance_context(
    name: str = "operation",
    enable_gpu_sync: bool = True,
) -> Generator[PerformanceProfiler, None, None]:
    """Context manager for simple performance profiling.

    Args:
        name: Name of the operation.
        enable_gpu_sync: Synchronize GPU for accurate timing.

    Yields:
        PerformanceProfiler instance.

    Example:
        >>> with performance_context("training") as profiler:
        ...     model.fit(X, y)
        >>> print(profiler.summary())
    """
    profiler = PerformanceProfiler(enable_gpu_sync=enable_gpu_sync)
    profiler.start(name)
    try:
        yield profiler
    finally:
        profiler.stop(name)
        logger.debug(profiler.summary())