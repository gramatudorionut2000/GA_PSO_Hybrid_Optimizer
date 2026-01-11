from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

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
from .optimization_config import (
    BoundaryHandling,
    InertiaConfig,
    InertiaStrategy,
    InitializationMethod,
    PSOSettings,
)
from .utils.data import DeviceInfo

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)



class FitnessFunction(Protocol):
    """Protocol for fitness evaluation functions."""

    def __call__(self, positions: ArrayLike) -> ArrayLike:
        """Evaluate fitness for batch of positions.

        Args:
            positions: Array of shape (n_particles, n_dims) with particle positions.

        Returns:
            Array of shape (n_particles,) with fitness values.
        """
        ...



@dataclass
class GPUSwarm:
    """GPU-accelerated swarm representation for Particle Swarm Optimization.

    Stores all swarm data as contiguous arrays for efficient GPU operations.
    All PSO operations (velocity update, position update, best updates)
    operate on these arrays in a vectorized manner.

    Attributes:
        positions: Float array of shape (swarm_size, n_dims).
                  Current particle positions.
        velocities: Float array of shape (swarm_size, n_dims).
                   Current particle velocities.
        personal_best_positions: Float array of shape (swarm_size, n_dims).
                                Best positions found by each particle.
        personal_best_fitness: Float array of shape (swarm_size,).
                              Fitness at personal best positions.
        fitness: Float array of shape (swarm_size,).
                Current fitness values.
        global_best_position: Float array of shape (n_dims,).
                             Best position found by any particle.
        global_best_fitness: Float scalar.
                            Fitness at global best position.
        lower_bounds: Float array of shape (n_dims,).
                     Lower bounds for each dimension.
        upper_bounds: Float array of shape (n_dims,).
                     Upper bounds for each dimension.
        device_info: Information about storage device (CPU/GPU).
        iteration: Current iteration number.
        particle_ids: List of unique IDs for each particle (CPU).
    """

    positions: ArrayLike
    velocities: ArrayLike
    personal_best_positions: ArrayLike
    personal_best_fitness: ArrayLike
    fitness: ArrayLike
    global_best_position: ArrayLike
    global_best_fitness: float
    lower_bounds: ArrayLike
    upper_bounds: ArrayLike
    device_info: DeviceInfo = field(default_factory=DeviceInfo.cpu)
    iteration: int = 0
    particle_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate swarm data and initialize IDs if needed."""
        # Validate dimensions
        if self.positions.ndim != 2:
            msg = f"positions must be 2D, got {self.positions.ndim}D"
            raise ValueError(msg)

        if self.velocities.ndim != 2:
            msg = f"velocities must be 2D, got {self.velocities.ndim}D"
            raise ValueError(msg)

        swarm_size, n_dims = self.positions.shape

        if self.velocities.shape != (swarm_size, n_dims):
            msg = f"velocities shape {self.velocities.shape} must match positions {self.positions.shape}"
            raise ValueError(msg)

        if self.personal_best_positions.shape != (swarm_size, n_dims):
            msg = "personal_best_positions shape must match positions"
            raise ValueError(msg)

        if len(self.personal_best_fitness) != swarm_size:
            msg = "personal_best_fitness length must match swarm_size"
            raise ValueError(msg)

        if len(self.fitness) != swarm_size:
            msg = "fitness length must match swarm_size"
            raise ValueError(msg)

        if len(self.global_best_position) != n_dims:
            msg = "global_best_position length must match n_dims"
            raise ValueError(msg)

        if len(self.lower_bounds) != n_dims:
            msg = "lower_bounds length must match n_dims"
            raise ValueError(msg)

        if len(self.upper_bounds) != n_dims:
            msg = "upper_bounds length must match n_dims"
            raise ValueError(msg)

        if not self.particle_ids:
            self.particle_ids = [self._generate_id() for _ in range(swarm_size)]

    @staticmethod
    def _generate_id() -> str:
        """Generate a unique particle ID."""
        return f"p_{uuid.uuid4().hex[:8]}"

    @property
    def swarm_size(self) -> int:
        """Number of particles in swarm."""
        return self.positions.shape[0]

    @property
    def n_dims(self) -> int:
        """Number of dimensions."""
        return self.positions.shape[1]

    @property
    def is_gpu(self) -> bool:
        """Check if swarm is on GPU."""
        return self.device_info.is_gpu

    @property
    def is_cpu(self) -> bool:
        """Check if swarm is on CPU."""
        return self.device_info.is_cpu

    @property
    def range_size(self) -> ArrayLike:
        """Get the range size for each dimension."""
        return self.upper_bounds - self.lower_bounds

    def get_backend(self) -> ArrayBackend:
        """Get the appropriate backend for this swarm.

        Returns the backend that matches the swarm's device (CPU/GPU).
        This ensures operations use the correct array library.
        """
        from .utils.backend import get_backend_manager

        manager = get_backend_manager()
        if self.is_gpu:
            try:
                return manager.get_backend("cupy")
            except ValueError as e:
                msg = "Swarm is on GPU but CuPy backend not available"
                raise RuntimeError(msg) from e
        return manager.get_backend("numpy")

    def to_cpu(self) -> GPUSwarm:
        """Transfer all array data to CPU.

        Returns:
            New GPUSwarm with all arrays on CPU.
        """
        if self.is_cpu:
            return self

        return GPUSwarm(
            positions=to_numpy(self.positions),
            velocities=to_numpy(self.velocities),
            personal_best_positions=to_numpy(self.personal_best_positions),
            personal_best_fitness=to_numpy(self.personal_best_fitness),
            fitness=to_numpy(self.fitness),
            global_best_position=to_numpy(self.global_best_position),
            global_best_fitness=self.global_best_fitness,
            lower_bounds=to_numpy(self.lower_bounds),
            upper_bounds=to_numpy(self.upper_bounds),
            device_info=DeviceInfo.cpu(),
            iteration=self.iteration,
            particle_ids=list(self.particle_ids),
        )

    def to_gpu(self, device_id: int = 0) -> GPUSwarm:
        """Transfer all array data to GPU.

        Args:
            device_id: CUDA device ID.

        Returns:
            New GPUSwarm with all arrays on GPU.

        Raises:
            RuntimeError: If GPU is not available.
        """
        if not is_gpu_available():
            msg = "GPU not available"
            raise RuntimeError(msg)

        if self.is_gpu:
            return self

        backend = get_backend()
        return GPUSwarm(
            positions=backend.asarray(self.positions),
            velocities=backend.asarray(self.velocities),
            personal_best_positions=backend.asarray(self.personal_best_positions),
            personal_best_fitness=backend.asarray(self.personal_best_fitness),
            fitness=backend.asarray(self.fitness),
            global_best_position=backend.asarray(self.global_best_position),
            global_best_fitness=self.global_best_fitness,
            lower_bounds=backend.asarray(self.lower_bounds),
            upper_bounds=backend.asarray(self.upper_bounds),
            device_info=DeviceInfo.gpu(device_id),
            iteration=self.iteration,
            particle_ids=list(self.particle_ids),
        )

    def to_device(self, device_info: DeviceInfo) -> GPUSwarm:
        """Transfer to specified device.

        Args:
            device_info: Target device information.

        Returns:
            GPUSwarm on the target device.
        """
        if device_info.is_gpu:
            return self.to_gpu(device_info.device_id)
        return self.to_cpu()

    def get_best_particle_index(self) -> int:
        """Get index of the particle with best current fitness.

        Returns:
            Index of particle with highest fitness.
        """
        backend = self.get_backend()
        best_idx = backend.argmax(self.fitness)
        return int(backend.to_scalar(best_idx))

    def get_best_personal_index(self) -> int:
        """Get index of the particle with best personal best fitness.

        Returns:
            Index of particle with highest personal best fitness.
        """
        backend = self.get_backend()
        best_idx = backend.argmax(self.personal_best_fitness)
        return int(backend.to_scalar(best_idx))

    def get_statistics(self) -> dict[str, float]:
        """Compute swarm statistics.

        Returns:
            Dictionary with fitness statistics and swarm metrics.
        """
        if self.is_gpu:
            fitness_cpu = to_numpy(self.fitness)
            positions_cpu = to_numpy(self.positions)
            pb_fitness_cpu = to_numpy(self.personal_best_fitness)
        else:
            fitness_cpu = np.asarray(self.fitness)
            positions_cpu = np.asarray(self.positions)
            pb_fitness_cpu = np.asarray(self.personal_best_fitness)

        position_std = float(np.mean(np.std(positions_cpu, axis=0)))

        valid_fitness = fitness_cpu[np.isfinite(fitness_cpu)]
        valid_pb_fitness = pb_fitness_cpu[np.isfinite(pb_fitness_cpu)]

        if len(valid_fitness) > 0:
            current_stats = {
                "current_min": float(np.min(valid_fitness)),
                "current_max": float(np.max(valid_fitness)),
                "current_mean": float(np.mean(valid_fitness)),
                "current_std": float(np.std(valid_fitness)),
            }
        else:
            current_stats = {
                "current_min": float("-inf"),
                "current_max": float("-inf"),
                "current_mean": float("-inf"),
                "current_std": 0.0,
            }

        if len(valid_pb_fitness) > 0:
            pb_stats = {
                "personal_best_min": float(np.min(valid_pb_fitness)),
                "personal_best_max": float(np.max(valid_pb_fitness)),
                "personal_best_mean": float(np.mean(valid_pb_fitness)),
            }
        else:
            pb_stats = {
                "personal_best_min": float("-inf"),
                "personal_best_max": float("-inf"),
                "personal_best_mean": float("-inf"),
            }

        return {
            **current_stats,
            **pb_stats,
            "global_best": self.global_best_fitness,
            "position_spread": position_std,
            "iteration": self.iteration,
        }

    def get_convergence_metric(self) -> float:
        """Compute convergence metric based on position spread.

        Returns:
            Average normalized standard deviation across dimensions.
            Lower values indicate convergence.
        """
        backend = self.get_backend()

        pos_std = backend.std(self.positions, axis=0)
        range_size = self.range_size

        normalized_std = pos_std / (range_size + 1e-10)
        convergence = float(backend.to_scalar(backend.mean(normalized_std)))

        return convergence

    def has_converged(self, threshold: float = 1e-6) -> bool:
        """Check if swarm has converged.

        Args:
            threshold: Convergence threshold.

        Returns:
            True if convergence metric is below threshold.
        """
        return self.get_convergence_metric() < threshold

    def copy(self) -> GPUSwarm:
        """Create a deep copy of the swarm."""
        backend = self.get_backend()

        if self.is_gpu:
            return GPUSwarm(
                positions=backend.copy(self.positions),
                velocities=backend.copy(self.velocities),
                personal_best_positions=backend.copy(self.personal_best_positions),
                personal_best_fitness=backend.copy(self.personal_best_fitness),
                fitness=backend.copy(self.fitness),
                global_best_position=backend.copy(self.global_best_position),
                global_best_fitness=self.global_best_fitness,
                lower_bounds=backend.copy(self.lower_bounds),
                upper_bounds=backend.copy(self.upper_bounds),
                device_info=self.device_info,
                iteration=self.iteration,
                particle_ids=list(self.particle_ids),
            )

        return GPUSwarm(
            positions=np.copy(self.positions),
            velocities=np.copy(self.velocities),
            personal_best_positions=np.copy(self.personal_best_positions),
            personal_best_fitness=np.copy(self.personal_best_fitness),
            fitness=np.copy(self.fitness),
            global_best_position=np.copy(self.global_best_position),
            global_best_fitness=self.global_best_fitness,
            lower_bounds=np.copy(self.lower_bounds),
            upper_bounds=np.copy(self.upper_bounds),
            device_info=self.device_info,
            iteration=self.iteration,
            particle_ids=list(self.particle_ids),
        )

    def get_particle(self, index: int) -> dict[str, Any]:
        """Get a single particle as a dictionary.

        Args:
            index: Index of the particle.

        Returns:
            Dictionary with particle's state.
        """
        backend = self.get_backend()

        return {
            "position": to_numpy(self.positions[index]),
            "velocity": to_numpy(self.velocities[index]),
            "fitness": float(backend.to_scalar(self.fitness[index])),
            "personal_best_position": to_numpy(self.personal_best_positions[index]),
            "personal_best_fitness": float(
                backend.to_scalar(self.personal_best_fitness[index])
            ),
            "id": self.particle_ids[index],
        }

    def __len__(self) -> int:
        """Return swarm size."""
        return self.swarm_size

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"GPUSwarm(swarm_size={self.swarm_size}, "
            f"n_dims={self.n_dims}, "
            f"iteration={self.iteration}, "
            f"global_best={stats['global_best']:.6f}, "
            f"device={self.device_info.backend_name})"
        )


class SwarmInitializer(ABC):
    """Abstract base class for swarm initialization strategies."""

    @abstractmethod
    def initialize(
        self,
        swarm_size: int,
        bounds: Sequence[tuple[float, float]],
        *,
        use_gpu: bool = False,
        dtype: str = "float64",
    ) -> GPUSwarm:
        """Initialize a swarm.

        Args:
            swarm_size: Number of particles.
            bounds: List of (lower, upper) bounds for each dimension.
            use_gpu: Whether to create on GPU.
            dtype: Data type for arrays.

        Returns:
            Initialized GPUSwarm.
        """


class RandomSwarmInitializer(SwarmInitializer):
    """Random swarm initialization.

    Initializes positions uniformly within bounds and velocities as small
    random values.
    """

    def __init__(
        self,
        velocity_scale: float = 0.1,
        seed: int | None = None,
    ) -> None:
        """Initialize random swarm initializer.

        Args:
            velocity_scale: Scale factor for initial velocities (fraction of range).
            seed: Random seed for reproducibility.
        """
        if not 0.0 < velocity_scale <= 1.0:
            msg = f"velocity_scale must be in (0, 1], got {velocity_scale}"
            raise ValueError(msg)

        self.velocity_scale = velocity_scale
        self.seed = seed

    def initialize(
        self,
        swarm_size: int,
        bounds: Sequence[tuple[float, float]],
        *,
        use_gpu: bool = False,
        dtype: str = "float64",
    ) -> GPUSwarm:
        """Initialize swarm with random positions and velocities.

        Args:
            swarm_size: Number of particles.
            bounds: List of (lower, upper) bounds for each dimension.
            use_gpu: Whether to create on GPU.
            dtype: Data type for arrays.

        Returns:
            Randomly initialized GPUSwarm.
        """
        n_dims = len(bounds)
        lower_bounds = np.array([b[0] for b in bounds], dtype=np.float64)
        upper_bounds = np.array([b[1] for b in bounds], dtype=np.float64)
        range_size = upper_bounds - lower_bounds

        backend = get_backend()

        if self.seed is not None:
            backend.set_seed(self.seed)

        if use_gpu and is_gpu_available():
            lower_gpu = backend.asarray(lower_bounds)
            upper_gpu = backend.asarray(upper_bounds)
            range_gpu = backend.asarray(range_size)

            positions = lower_gpu + backend.random_uniform(
                0, 1, (swarm_size, n_dims)
            ) * range_gpu

            velocities = backend.random_uniform(
                -self.velocity_scale, self.velocity_scale, (swarm_size, n_dims)
            ) * range_gpu

            personal_best_positions = backend.copy(positions)
            personal_best_fitness = backend.full(swarm_size, float("-inf"), dtype=dtype)  # pyright: ignore[reportArgumentType]
            fitness = backend.full(swarm_size, float("-inf"), dtype=dtype)  # pyright: ignore[reportArgumentType]

            global_best_position = backend.copy(positions[0])
            global_best_fitness = float("-inf")

            device_info = DeviceInfo.gpu()
        else:
            rng = np.random.default_rng(self.seed)

            positions = lower_bounds + rng.random((swarm_size, n_dims)) * range_size
            velocities = rng.uniform(
                -self.velocity_scale, self.velocity_scale, (swarm_size, n_dims)
            ) * range_size

            personal_best_positions = np.copy(positions)
            personal_best_fitness = np.full(swarm_size, -np.inf, dtype=dtype)
            fitness = np.full(swarm_size, -np.inf, dtype=dtype)
            global_best_position = np.copy(positions[0])
            global_best_fitness = float("-inf")

            device_info = DeviceInfo.cpu()
            lower_gpu = lower_bounds  # type: ignore[assignment]
            upper_gpu = upper_bounds  # type: ignore[assignment]

        return GPUSwarm(
            positions=positions,
            velocities=velocities,
            personal_best_positions=personal_best_positions,
            personal_best_fitness=personal_best_fitness,
            fitness=fitness,
            global_best_position=global_best_position,
            global_best_fitness=global_best_fitness,
            lower_bounds=lower_gpu,
            upper_bounds=upper_gpu,
            device_info=device_info,
            iteration=0,
        )


class LatinHypercubeSwarmInitializer(SwarmInitializer):
    """Latin Hypercube Sampling initialization for better coverage.

    Uses scipy's Latin Hypercube sampler on CPU and transfers to GPU.
    """

    def __init__(
        self,
        velocity_scale: float = 0.1,
        seed: int | None = None,
        strength: int = 1,
    ) -> None:
        """Initialize LHS swarm initializer.

        Args:
            velocity_scale: Scale factor for initial velocities.
            seed: Random seed for reproducibility.
            strength: Number of iterations to improve LHS sample.
        """
        self.velocity_scale = velocity_scale
        self.seed = seed
        self.strength = strength

    def initialize(
        self,
        swarm_size: int,
        bounds: Sequence[tuple[float, float]],
        *,
        use_gpu: bool = False,
        dtype: str = "float64",
    ) -> GPUSwarm:
        """Initialize swarm using Latin Hypercube Sampling.

        Args:
            swarm_size: Number of particles.
            bounds: List of (lower, upper) bounds for each dimension.
            use_gpu: Whether to create on GPU.
            dtype: Data type for arrays.

        Returns:
            LHS-initialized GPUSwarm.
        """
        n_dims = len(bounds)
        lower_bounds = np.array([b[0] for b in bounds], dtype=np.float64)
        upper_bounds = np.array([b[1] for b in bounds], dtype=np.float64)
        range_size = upper_bounds - lower_bounds

        sampler = qmc.LatinHypercube(
            d=n_dims, rng=self.seed, strength=self.strength
        )
        samples = sampler.random(n=swarm_size)

        positions_cpu = lower_bounds + samples * range_size

        rng = np.random.default_rng(self.seed)
        velocities_cpu = rng.uniform(
            -self.velocity_scale, self.velocity_scale, (swarm_size, n_dims)
        ) * range_size

        personal_best_positions_cpu = np.copy(positions_cpu)
        personal_best_fitness_cpu = np.full(swarm_size, -np.inf, dtype=dtype)
        fitness_cpu = np.full(swarm_size, -np.inf, dtype=dtype)
        global_best_position_cpu = np.copy(positions_cpu[0])

        if use_gpu and is_gpu_available():
            backend = get_backend()
            return GPUSwarm(
                positions=backend.asarray(positions_cpu),
                velocities=backend.asarray(velocities_cpu),
                personal_best_positions=backend.asarray(personal_best_positions_cpu),
                personal_best_fitness=backend.asarray(personal_best_fitness_cpu),
                fitness=backend.asarray(fitness_cpu),
                global_best_position=backend.asarray(global_best_position_cpu),
                global_best_fitness=float("-inf"),
                lower_bounds=backend.asarray(lower_bounds),
                upper_bounds=backend.asarray(upper_bounds),
                device_info=DeviceInfo.gpu(),
                iteration=0,
            )

        return GPUSwarm(
            positions=positions_cpu,
            velocities=velocities_cpu,
            personal_best_positions=personal_best_positions_cpu,
            personal_best_fitness=personal_best_fitness_cpu,
            fitness=fitness_cpu,
            global_best_position=global_best_position_cpu,
            global_best_fitness=float("-inf"),
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            device_info=DeviceInfo.cpu(),
            iteration=0,
        )


class SobolSwarmInitializer(SwarmInitializer):
    """Sobol sequence initialization for quasi-random coverage."""

    def __init__(
        self,
        velocity_scale: float = 0.1,
        seed: int | None = None,
        scramble: bool = True,
    ) -> None:
        """Initialize Sobol swarm initializer.

        Args:
            velocity_scale: Scale factor for initial velocities.
            seed: Random seed for reproducibility.
            scramble: Whether to scramble the Sobol sequence.
        """
        self.velocity_scale = velocity_scale
        self.seed = seed
        self.scramble = scramble

    def initialize(
        self,
        swarm_size: int,
        bounds: Sequence[tuple[float, float]],
        *,
        use_gpu: bool = False,
        dtype: str = "float64",
    ) -> GPUSwarm:
        """Initialize swarm using Sobol sequence.

        Args:
            swarm_size: Number of particles.
            bounds: List of (lower, upper) bounds for each dimension.
            use_gpu: Whether to create on GPU.
            dtype: Data type for arrays.

        Returns:
            Sobol-initialized GPUSwarm.
        """
        n_dims = len(bounds)
        lower_bounds = np.array([b[0] for b in bounds], dtype=np.float64)
        upper_bounds = np.array([b[1] for b in bounds], dtype=np.float64)
        range_size = upper_bounds - lower_bounds

        sampler = qmc.Sobol(d=n_dims, rng=self.seed, scramble=self.scramble)
        samples = sampler.random(n=swarm_size)

        positions_cpu = lower_bounds + samples * range_size

        rng = np.random.default_rng(self.seed)
        velocities_cpu = rng.uniform(
            -self.velocity_scale, self.velocity_scale, (swarm_size, n_dims)
        ) * range_size

        personal_best_positions_cpu = np.copy(positions_cpu)
        personal_best_fitness_cpu = np.full(swarm_size, -np.inf, dtype=dtype)
        fitness_cpu = np.full(swarm_size, -np.inf, dtype=dtype)
        global_best_position_cpu = np.copy(positions_cpu[0])

        if use_gpu and is_gpu_available():
            backend = get_backend()
            return GPUSwarm(
                positions=backend.asarray(positions_cpu),
                velocities=backend.asarray(velocities_cpu),
                personal_best_positions=backend.asarray(personal_best_positions_cpu),
                personal_best_fitness=backend.asarray(personal_best_fitness_cpu),
                fitness=backend.asarray(fitness_cpu),
                global_best_position=backend.asarray(global_best_position_cpu),
                global_best_fitness=float("-inf"),
                lower_bounds=backend.asarray(lower_bounds),
                upper_bounds=backend.asarray(upper_bounds),
                device_info=DeviceInfo.gpu(),
                iteration=0,
            )

        return GPUSwarm(
            positions=positions_cpu,
            velocities=velocities_cpu,
            personal_best_positions=personal_best_positions_cpu,
            personal_best_fitness=personal_best_fitness_cpu,
            fitness=fitness_cpu,
            global_best_position=global_best_position_cpu,
            global_best_fitness=float("-inf"),
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            device_info=DeviceInfo.cpu(),
            iteration=0,
        )




def update_velocities(
    swarm: GPUSwarm,
    inertia: float,
    cognitive_coef: float,
    social_coef: float,
    velocity_clamp: float = 0.5,
) -> None:
    """Update particle velocities using standard PSO formula (in-place).

    Standard PSO velocity update:
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)

    Args:
        swarm: GPUSwarm to update.
        inertia: Inertia weight (w).
        cognitive_coef: Cognitive coefficient (c1).
        social_coef: Social coefficient (c2).
        velocity_clamp: Velocity clamping factor (fraction of range).
    """
    backend = swarm.get_backend()

    r1 = backend.random_uniform(0, 1, swarm.positions.shape)
    r2 = backend.random_uniform(0, 1, swarm.positions.shape)

    cognitive = cognitive_coef * r1 * (
        swarm.personal_best_positions - swarm.positions
    )


    social = social_coef * r2 * (swarm.global_best_position - swarm.positions)

    swarm.velocities = inertia * swarm.velocities + cognitive + social

    max_velocity = velocity_clamp * swarm.range_size
    swarm.velocities = backend.clip(swarm.velocities, -max_velocity, max_velocity)


def update_velocities_constriction(
    swarm: GPUSwarm,
    cognitive_coef: float = 2.05,
    social_coef: float = 2.05,
    velocity_clamp: float = 0.5,
) -> None:
    """Update velocities using constriction factor method (in-place).

    Uses Clerc's constriction factor for improved convergence.

    Args:
        swarm: GPUSwarm to update.
        cognitive_coef: Cognitive coefficient (c1).
        social_coef: Social coefficient (c2).
        velocity_clamp: Velocity clamping factor.
    """
    backend = swarm.get_backend()

    phi = cognitive_coef + social_coef
    if phi <= 4.0:
        chi = 1.0
    else:
        chi = 2.0 / abs(2.0 - phi - np.sqrt(phi * phi - 4.0 * phi))

    r1 = backend.random_uniform(0, 1, swarm.positions.shape)
    r2 = backend.random_uniform(0, 1, swarm.positions.shape)

    cognitive = cognitive_coef * r1 * (
        swarm.personal_best_positions - swarm.positions
    )
    social = social_coef * r2 * (swarm.global_best_position - swarm.positions)

    swarm.velocities = chi * (swarm.velocities + cognitive + social)

    max_velocity = velocity_clamp * swarm.range_size
    swarm.velocities = backend.clip(swarm.velocities, -max_velocity, max_velocity)




def update_positions(
    swarm: GPUSwarm,
    boundary_handling: BoundaryHandling = BoundaryHandling.CLAMP,
) -> None:
    """Update particle positions (in-place).

    Args:
        swarm: GPUSwarm to update.
        boundary_handling: How to handle particles at boundaries.
    """
    backend = swarm.get_backend()

    swarm.positions = swarm.positions + swarm.velocities

    if boundary_handling == BoundaryHandling.CLAMP:
        _handle_boundary_clamp(swarm, backend)
    elif boundary_handling == BoundaryHandling.REFLECT:
        _handle_boundary_reflect(swarm, backend)
    elif boundary_handling == BoundaryHandling.WRAP:
        _handle_boundary_wrap(swarm, backend)
    elif boundary_handling == BoundaryHandling.RANDOM:
        _handle_boundary_random(swarm, backend)
    else:
        _handle_boundary_clamp(swarm, backend)


def _handle_boundary_clamp(swarm: GPUSwarm, backend: ArrayBackend) -> None:
    """Clamp positions to bounds (absorbing boundaries)."""
    swarm.positions = backend.clip(
        swarm.positions, swarm.lower_bounds, swarm.upper_bounds
    )

    at_lower = swarm.positions <= swarm.lower_bounds
    at_upper = swarm.positions >= swarm.upper_bounds
    at_boundary = backend.logical_or(at_lower, at_upper)
    swarm.velocities = backend.where(at_boundary, 0.0, swarm.velocities)


def _handle_boundary_reflect(swarm: GPUSwarm, backend: ArrayBackend) -> None:
    """Reflect positions at boundaries."""
    lower = swarm.lower_bounds
    upper = swarm.upper_bounds

    out_low = swarm.positions < lower
    out_high = swarm.positions > upper

    swarm.positions = backend.where(
        out_low, 2.0 * lower - swarm.positions, swarm.positions
    )
    swarm.positions = backend.where(
        out_high, 2.0 * upper - swarm.positions, swarm.positions
    )

    swarm.positions = backend.clip(swarm.positions, lower, upper)

    at_boundary = backend.logical_or(out_low, out_high)
    swarm.velocities = backend.where(at_boundary, -swarm.velocities, swarm.velocities)


def _handle_boundary_wrap(swarm: GPUSwarm, backend: ArrayBackend) -> None:
    """Wrap positions around boundaries (periodic boundaries)."""
    lower = swarm.lower_bounds
    upper = swarm.upper_bounds  # noqa: F841
    range_size = swarm.range_size

    normalized = (swarm.positions - lower) / range_size

    normalized = normalized - backend.floor(normalized)

    swarm.positions = lower + normalized * range_size


def _handle_boundary_random(swarm: GPUSwarm, backend: ArrayBackend) -> None:
    """Reinitialize positions randomly if out of bounds."""
    lower = swarm.lower_bounds
    upper = swarm.upper_bounds
    range_size = swarm.range_size

    out_low = swarm.positions < lower
    out_high = swarm.positions > upper
    out_of_bounds = backend.logical_or(out_low, out_high)

    random_positions = lower + backend.random_uniform(
        0, 1, swarm.positions.shape
    ) * range_size

    swarm.positions = backend.where(out_of_bounds, random_positions, swarm.positions)

    random_velocities = backend.random_uniform(
        -0.1, 0.1, swarm.velocities.shape
    ) * range_size
    swarm.velocities = backend.where(out_of_bounds, random_velocities, swarm.velocities)



def update_personal_bests(swarm: GPUSwarm) -> int:
    """Update personal best positions and fitness (in-place).

    Args:
        swarm: GPUSwarm to update.

    Returns:
        Number of particles that improved their personal best.
    """
    backend = swarm.get_backend()
    xp = backend.xp

    fitness = backend.asarray(swarm.fitness)
    personal_best_fitness = backend.asarray(swarm.personal_best_fitness)

    improved = fitness > personal_best_fitness


    improved_2d = xp.reshape(improved, (-1, 1))
    improved_broadcast = xp.broadcast_to(
        improved_2d, swarm.positions.shape
    ).copy()

    swarm.personal_best_positions = backend.where(
        improved_broadcast,
        backend.asarray(swarm.positions),
        backend.asarray(swarm.personal_best_positions),
    )

    swarm.personal_best_fitness = backend.where(
        improved, fitness, personal_best_fitness
    )

    n_improved = int(backend.to_scalar(xp.sum(improved.astype(xp.int32))))

    return n_improved


def update_global_best(swarm: GPUSwarm) -> bool:
    """Update global best position and fitness (in-place).

    Args:
        swarm: GPUSwarm to update.

    Returns:
        True if global best was improved.
    """
    backend = swarm.get_backend()

    best_idx = backend.argmax(swarm.personal_best_fitness)
    best_fitness = backend.to_scalar(swarm.personal_best_fitness[best_idx])

    if best_fitness > swarm.global_best_fitness:
        swarm.global_best_fitness = float(best_fitness)
        swarm.global_best_position = backend.copy(
            swarm.personal_best_positions[best_idx]
        )
        return True

    return False


def update_all_bests(swarm: GPUSwarm) -> tuple[int, bool]:
    """Update both personal and global bests.

    Args:
        swarm: GPUSwarm to update.

    Returns:
        Tuple of (n_personal_improvements, global_improved).
    """
    n_improved = update_personal_bests(swarm)
    global_improved = update_global_best(swarm)
    return n_improved, global_improved


def evaluate_swarm(
    swarm: GPUSwarm,
    fitness_fn: FitnessFunction,
) -> None:
    """Evaluate fitness for all particles (in-place).

    Args:
        swarm: GPUSwarm to evaluate.
        fitness_fn: Function that takes positions array and returns fitness array.
    """
    swarm.fitness = fitness_fn(swarm.positions)


def evaluate_swarm_batch(
    swarm: GPUSwarm,
    fitness_fn: Callable[[NDArray], NDArray],
    batch_size: int | None = None,
) -> None:
    """Evaluate fitness in batches (useful for memory constraints).

    This method transfers to CPU for evaluation if needed.

    Args:
        swarm: GPUSwarm to evaluate.
        fitness_fn: Function that takes NumPy positions and returns NumPy fitness.
        batch_size: Batch size for evaluation. None for all at once.
    """
    backend = swarm.get_backend()
    positions_cpu = to_numpy(swarm.positions)

    if batch_size is None or batch_size >= swarm.swarm_size:
        fitness_cpu = fitness_fn(positions_cpu)
    else:
        fitness_cpu = np.empty(swarm.swarm_size, dtype=np.float64)
        for start in range(0, swarm.swarm_size, batch_size):
            end = min(start + batch_size, swarm.swarm_size)
            fitness_cpu[start:end] = fitness_fn(positions_cpu[start:end])

    if swarm.is_gpu:
        swarm.fitness = backend.asarray(fitness_cpu)
    else:
        swarm.fitness = fitness_cpu



@dataclass
class PSOIterationResult:
    """Result of a single PSO iteration."""

    iteration: int
    global_best_fitness: float
    global_best_position: NDArray
    n_personal_improvements: int
    global_improved: bool
    convergence_metric: float
    statistics: dict[str, float]


def pso_iteration(
    swarm: GPUSwarm,
    fitness_fn: FitnessFunction,
    inertia: float,
    cognitive_coef: float,
    social_coef: float,
    velocity_clamp: float = 0.5,
    boundary_handling: BoundaryHandling = BoundaryHandling.CLAMP,
) -> PSOIterationResult:
    """Perform a single PSO iteration.

    Args:
        swarm: GPUSwarm to evolve.
        fitness_fn: Fitness evaluation function.
        inertia: Inertia weight.
        cognitive_coef: Cognitive coefficient.
        social_coef: Social coefficient.
        velocity_clamp: Velocity clamping factor.
        boundary_handling: Boundary handling method.

    Returns:
        PSOIterationResult with iteration statistics.
    """
    update_velocities(
        swarm,
        inertia=inertia,
        cognitive_coef=cognitive_coef,
        social_coef=social_coef,
        velocity_clamp=velocity_clamp,
    )

    update_positions(swarm, boundary_handling=boundary_handling)

    evaluate_swarm(swarm, fitness_fn)

    n_personal, global_improved = update_all_bests(swarm)

    swarm.iteration += 1

    stats = swarm.get_statistics()
    convergence = swarm.get_convergence_metric()

    return PSOIterationResult(
        iteration=swarm.iteration,
        global_best_fitness=swarm.global_best_fitness,
        global_best_position=to_numpy(swarm.global_best_position),
        n_personal_improvements=n_personal,
        global_improved=global_improved,
        convergence_metric=convergence,
        statistics=stats,
    )



@dataclass
class PSOResult:
    """Result of PSO optimization."""

    best_position: NDArray
    best_fitness: float
    n_iterations: int
    converged: bool
    convergence_metric: float
    history: list[PSOIterationResult]
    final_swarm: GPUSwarm | None = None


class PSOOptimizer:
    """GPU-accelerated Particle Swarm Optimizer.

    Example:
        >>> optimizer = PSOOptimizer(swarm_size=50, max_iterations=100)
        >>> bounds = [(0.0, 1.0), (0.001, 0.1), (0.5, 0.99)]
        >>> result = optimizer.optimize(objective_fn, bounds)
        >>> print(f"Best fitness: {result.best_fitness}")
    """

    def __init__(
        self,
        swarm_size: int = 30,
        max_iterations: int = 50,
        inertia_config: InertiaConfig | None = None,
        cognitive_coef: float = 2.0,
        social_coef: float = 2.0,
        velocity_clamp: float = 0.5,
        boundary_handling: BoundaryHandling = BoundaryHandling.CLAMP,
        convergence_threshold: float = 1e-8,
        initialization_method: InitializationMethod = InitializationMethod.RANDOM,
        use_gpu: bool = False,
        seed: int | None = None,
        keep_history: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize PSO optimizer.

        Args:
            swarm_size: Number of particles.
            max_iterations: Maximum iterations.
            inertia_config: Inertia weight configuration.
            cognitive_coef: Cognitive coefficient (c1).
            social_coef: Social coefficient (c2).
            velocity_clamp: Velocity clamping factor.
            boundary_handling: Boundary handling method.
            convergence_threshold: Convergence threshold.
            initialization_method: Swarm initialization method.
            use_gpu: Whether to use GPU.
            seed: Random seed.
            keep_history: Whether to keep iteration history.
            verbose: Whether to print progress.
        """
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.inertia_config = inertia_config or InertiaConfig()
        self.cognitive_coef = cognitive_coef
        self.social_coef = social_coef
        self.velocity_clamp = velocity_clamp
        self.boundary_handling = boundary_handling
        self.convergence_threshold = convergence_threshold
        self.initialization_method = initialization_method
        self.use_gpu = use_gpu and is_gpu_available()
        self.seed = seed
        self.keep_history = keep_history
        self.verbose = verbose

    @classmethod
    def from_config(cls, config: PSOSettings, use_gpu: bool = False) -> PSOOptimizer:
        """Create optimizer from PSOConfig.

        Args:
            config: PSO configuration.
            use_gpu: Whether to use GPU.

        Returns:
            PSOOptimizer instance.
        """
        inertia_config = InertiaConfig(
            strategy=config.inertia_strategy,
            initial=config.inertia_start,
            final=config.inertia_end,
        )
        
        return cls(
            swarm_size=config.swarm_size,
            max_iterations=config.max_iterations,
            inertia_config=inertia_config,
            cognitive_coef=config.cognitive_coef,
            social_coef=config.social_coef,
            velocity_clamp=config.velocity_clamp,
            boundary_handling=config.boundary_handling,
            convergence_threshold=config.early_stopping_tolerance,
            initialization_method=config.initialization_method,
            use_gpu=use_gpu,
        )

    def _create_initializer(self) -> SwarmInitializer:
        """Create swarm initializer based on configuration."""
        if self.initialization_method == InitializationMethod.LATIN_HYPERCUBE:
            return LatinHypercubeSwarmInitializer(seed=self.seed)
        if self.initialization_method == InitializationMethod.SOBOL:
            return SobolSwarmInitializer(seed=self.seed)
        return RandomSwarmInitializer(seed=self.seed)

    def optimize(
        self,
        fitness_fn: FitnessFunction,
        bounds: Sequence[tuple[float, float]],
        initial_swarm: GPUSwarm | None = None,
    ) -> PSOResult:
        """Run PSO optimization.

        Args:
            fitness_fn: Fitness function to maximize.
            bounds: List of (lower, upper) bounds for each dimension.
            initial_swarm: Optional initial swarm.

        Returns:
            PSOResult with optimization results.
        """
        if initial_swarm is not None:
            swarm = initial_swarm.to_device(
                DeviceInfo.gpu() if self.use_gpu else DeviceInfo.cpu()
            )
        else:
            initializer = self._create_initializer()
            swarm = initializer.initialize(
                self.swarm_size, bounds, use_gpu=self.use_gpu
            )

        evaluate_swarm(swarm, fitness_fn)
        update_all_bests(swarm)

        if self.verbose:
            logger.info(
                "Initial global best: %.6f", swarm.global_best_fitness
            )

        history: list[PSOIterationResult] = []
        converged = False

        for iteration in range(self.max_iterations):
            inertia = self.inertia_config.get_inertia(iteration, self.max_iterations)

            result = pso_iteration(
                swarm,
                fitness_fn,
                inertia=inertia,
                cognitive_coef=self.cognitive_coef,
                social_coef=self.social_coef,
                velocity_clamp=self.velocity_clamp,
                boundary_handling=self.boundary_handling,
            )

            if self.keep_history:
                history.append(result)

            if self.verbose and iteration % 10 == 0:
                logger.info(
                    "Iteration %d: best=%.6f, conv=%.6e",
                    iteration,
                    result.global_best_fitness,
                    result.convergence_metric,
                )

            if result.convergence_metric < self.convergence_threshold:
                converged = True
                if self.verbose:
                    logger.info("Converged at iteration %d", iteration)
                break

        return PSOResult(
            best_position=to_numpy(swarm.global_best_position),
            best_fitness=swarm.global_best_fitness,
            n_iterations=swarm.iteration,
            converged=converged,
            convergence_metric=swarm.get_convergence_metric(),
            history=history,
            final_swarm=swarm if self.keep_history else None,
        )




def create_swarm_initializer(
    method: str = "random",
    velocity_scale: float = 0.1,
    seed: int | None = None,
) -> SwarmInitializer:
    """Create a swarm initializer.

    Args:
        method: Initialization method ('random', 'lhs', 'sobol').
        velocity_scale: Scale factor for initial velocities.
        seed: Random seed.

    Returns:
        SwarmInitializer instance.
    """
    method = method.lower()

    if method == "random":
        return RandomSwarmInitializer(velocity_scale=velocity_scale, seed=seed)

    if method in ("lhs", "latin_hypercube"):
        return LatinHypercubeSwarmInitializer(velocity_scale=velocity_scale, seed=seed)

    if method == "sobol":
        return SobolSwarmInitializer(velocity_scale=velocity_scale, seed=seed)

    msg = f"Unknown initialization method: {method}"
    raise ValueError(msg)


def create_pso_optimizer(
    swarm_size: int = 30,
    max_iterations: int = 50,
    inertia_strategy: str = "linear_decay",
    inertia_initial: float = 0.9,
    inertia_final: float = 0.4,
    cognitive_coef: float = 2.0,
    social_coef: float = 2.0,
    velocity_clamp: float = 0.5,
    boundary_handling: str = "clamp",
    initialization: str = "random",
    use_gpu: bool = False,
    seed: int | None = None,
) -> PSOOptimizer:
    """Create a PSO optimizer with common settings.

    Args:
        swarm_size: Number of particles.
        max_iterations: Maximum iterations.
        inertia_strategy: Inertia strategy ('constant', 'linear_decay', 'nonlinear_decay').
        inertia_initial: Initial inertia weight.
        inertia_final: Final inertia weight.
        cognitive_coef: Cognitive coefficient.
        social_coef: Social coefficient.
        velocity_clamp: Velocity clamping factor.
        boundary_handling: Boundary handling ('clamp', 'reflect', 'wrap', 'random').
        initialization: Initialization method ('random', 'lhs', 'sobol').
        use_gpu: Whether to use GPU.
        seed: Random seed.

    Returns:
        PSOOptimizer instance.
    """
    strategy_map = {
        "constant": InertiaStrategy.CONSTANT,
        "linear_decay": InertiaStrategy.LINEAR_DECAY,
        "nonlinear_decay": InertiaStrategy.NONLINEAR_DECAY,
        "adaptive": InertiaStrategy.ADAPTIVE,
    }
    boundary_map = {
        "clamp": BoundaryHandling.CLAMP,
        "reflect": BoundaryHandling.REFLECT,
        "wrap": BoundaryHandling.WRAP,
        "random": BoundaryHandling.RANDOM,
    }
    init_map = {
        "random": InitializationMethod.RANDOM,
        "lhs": InitializationMethod.LATIN_HYPERCUBE,
        "latin_hypercube": InitializationMethod.LATIN_HYPERCUBE,
        "sobol": InitializationMethod.SOBOL,
    }

    inertia_config = InertiaConfig(
        strategy=strategy_map.get(inertia_strategy.lower(), InertiaStrategy.LINEAR_DECAY),
        initial=inertia_initial,
        final=inertia_final,
    )

    return PSOOptimizer(
        swarm_size=swarm_size,
        max_iterations=max_iterations,
        inertia_config=inertia_config,
        cognitive_coef=cognitive_coef,
        social_coef=social_coef,
        velocity_clamp=velocity_clamp,
        boundary_handling=boundary_map.get(
            boundary_handling.lower(), BoundaryHandling.CLAMP
        ),
        initialization_method=init_map.get(
            initialization.lower(), InitializationMethod.RANDOM
        ),
        use_gpu=use_gpu,
        seed=seed,
    )