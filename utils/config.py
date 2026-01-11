from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class ParameterType(Enum):
    """Types of parameters in the optimization space."""

    FEATURE = auto()  # Binary feature selection
    DISCRETE = auto()  # Categorical or integer hyperparameters
    CONTINUOUS = auto()  # Float hyperparameters


class OptimizationDirection(Enum):
    """Direction of optimization."""

    MAXIMIZE = auto()
    MINIMIZE = auto()


class SelectionMethod(Enum):
    """Selection methods for GA."""

    TOURNAMENT = auto()
    ROULETTE = auto()
    RANK = auto()
    TRUNCATION = auto()
    ELITIST = auto()


class CrossoverMethod(Enum):
    """Crossover methods for GA."""

    UNIFORM = auto()
    SINGLE_POINT = auto()
    TWO_POINT = auto()
    BLEND = auto()
    SBX = auto()  # Simulated Binary Crossover


class MutationMethod(Enum):
    """Mutation methods for GA."""

    UNIFORM = auto()
    GAUSSIAN = auto()
    POLYNOMIAL = auto()
    ADAPTIVE = auto()


class BoundaryHandling(Enum):
    """Boundary handling methods for PSO."""

    CLAMP = auto()
    REFLECT = auto()
    WRAP = auto()
    RANDOM = auto()


class InertiaStrategy(Enum):
    """Inertia weight strategies for PSO."""

    CONSTANT = auto()
    LINEAR_DECAY = auto()
    NONLINEAR_DECAY = auto()
    ADAPTIVE = auto()


class InitializationMethod(Enum):
    """Population/swarm initialization methods."""

    RANDOM = auto()
    LATIN_HYPERCUBE = auto()
    SOBOL = auto()
    HALTON = auto()
    GRID = auto()


# =============================================================================
# GPU Configuration
# =============================================================================


@dataclass
class GPUConfig:
    """Configuration for GPU acceleration.

    Attributes:
        use_gpu: Whether to attempt GPU acceleration.
        device_id: CUDA device ID to use.
        memory_limit: Fraction of GPU memory to use (0.0-1.0).
        min_array_size_for_gpu: Minimum array size to benefit from GPU.
        transfer_threshold: Minimum bytes to warrant CPU<->GPU transfer.
        fallback_to_cpu: Fall back to CPU on GPU errors.
        use_float32: Use float32 instead of float64 to save memory.
        memory_pool_size: GPU memory pool size in bytes (None = auto).
        sync_mode: Synchronize GPU operations for debugging.
    """

    use_gpu: bool = True
    device_id: int = 0
    memory_limit: float = 0.8
    min_array_size_for_gpu: int = 1000
    transfer_threshold: int = 10000
    fallback_to_cpu: bool = True
    use_float32: bool = False
    memory_pool_size: int | None = None
    sync_mode: bool = False

    def __post_init__(self) -> None:
        """Validate GPU configuration."""
        if not 0.0 < self.memory_limit <= 1.0:
            msg = f"memory_limit must be in (0, 1], got {self.memory_limit}"
            raise ValueError(msg)

        if self.min_array_size_for_gpu < 0:
            msg = "min_array_size_for_gpu must be non-negative"
            raise ValueError(msg)

        if self.transfer_threshold < 0:
            msg = "transfer_threshold must be non-negative"
            raise ValueError(msg)

        if self.memory_pool_size is not None and self.memory_pool_size <= 0:
            msg = "memory_pool_size must be positive if specified"
            raise ValueError(msg)

    @property
    def dtype(self) -> str:
        """Get the numpy dtype string based on configuration."""
        return "float32" if self.use_float32 else "float64"

    def estimate_memory(
        self,
        n_samples: int,
        n_features: int,
        pop_size: int,
        swarm_size: int,
        n_discrete: int,
        n_continuous: int,
    ) -> dict[str, int]:
        """Estimate GPU memory requirements.

        Args:
            n_samples: Number of data samples.
            n_features: Number of features.
            pop_size: GA population size.
            swarm_size: PSO swarm size.
            n_discrete: Number of discrete parameters.
            n_continuous: Number of continuous parameters.

        Returns:
            Dictionary with memory estimates in bytes.
        """
        dtype_size = 4 if self.use_float32 else 8

        data_memory = n_samples * n_features * dtype_size * 2
        ga_memory = pop_size * (n_features + n_discrete) * dtype_size
        ga_memory += pop_size * dtype_size  # fitness array
        pso_memory = swarm_size * n_continuous * dtype_size * 4
        pso_memory += n_continuous * dtype_size  # global best
        temp_memory = max(
            pop_size * pop_size * dtype_size,  # diversity matrix
            n_samples * dtype_size * 10,  # CV predictions
        )

        total = data_memory + ga_memory + pso_memory + temp_memory

        return {
            "data_memory": data_memory,
            "ga_memory": ga_memory,
            "pso_memory": pso_memory,
            "temp_memory": temp_memory,
            "total_memory": total,
            "dtype_size": dtype_size,
        }


# =============================================================================
# Feature Configuration
# =============================================================================


@dataclass
class FeatureConfig:
    """Configuration for feature selection.

    Attributes:
        n_features: Number of features (auto-detect if None).
        min_features: Minimum number of features to select.
        max_features: Maximum number of features (None = all).
        feature_mutation_rate: Mutation rate for feature genes.
        feature_importance_bias: Use importance for initialization.
        feature_groups: Groups of related features (for constraints).
    """

    n_features: int | None = None
    min_features: int = 1
    max_features: int | None = None
    feature_mutation_rate: float = 0.05
    feature_importance_bias: bool = False
    feature_groups: list[list[int]] | None = None

    def __post_init__(self) -> None:
        """Validate feature configuration."""
        if self.min_features < 1:
            msg = "min_features must be at least 1"
            raise ValueError(msg)

        if self.max_features is not None and self.max_features < self.min_features:
            msg = "max_features must be >= min_features"
            raise ValueError(msg)

        if not 0.0 <= self.feature_mutation_rate <= 1.0:
            msg = f"feature_mutation_rate must be in [0, 1], got {self.feature_mutation_rate}"
            raise ValueError(msg)


# =============================================================================
# GA Configuration
# =============================================================================


@dataclass
class GAConfig:
    """Configuration for Genetic Algorithm.

    Attributes:
        population_size: Number of individuals in population.
        max_generations: Maximum number of generations.
        crossover_rate: Probability of crossover.
        mutation_rate: Probability of mutation per gene.
        elitism_count: Number of elite individuals to preserve.
        selection_method: Selection strategy.
        crossover_method: Crossover operator.
        mutation_method: Mutation operator.
        tournament_size: Tournament size (for tournament selection).
        selection_pressure: Selection pressure (for rank selection).
        early_stopping_generations: Generations without improvement to stop.
        early_stopping_tolerance: Minimum improvement to reset counter.
        use_caching: Enable fitness caching.
        cache_size: Maximum cache entries (0 = unlimited).
        initialization_method: Population initialization method.
    """

    population_size: int = 50
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elitism_count: int = 2
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.BLEND
    mutation_method: MutationMethod = MutationMethod.ADAPTIVE
    tournament_size: int = 3
    selection_pressure: float = 1.8
    early_stopping_generations: int = 20
    early_stopping_tolerance: float = 1e-6
    use_caching: bool = True
    cache_size: int = 0
    initialization_method: InitializationMethod = InitializationMethod.LATIN_HYPERCUBE

    def __post_init__(self) -> None:
        """Validate GA configuration."""
        if self.population_size < 2:
            msg = f"population_size must be >= 2, got {self.population_size}"
            raise ValueError(msg)

        if self.max_generations < 1:
            msg = f"max_generations must be >= 1, got {self.max_generations}"
            raise ValueError(msg)

        if not 0.0 <= self.crossover_rate <= 1.0:
            msg = f"crossover_rate must be in [0, 1], got {self.crossover_rate}"
            raise ValueError(msg)

        if not 0.0 <= self.mutation_rate <= 1.0:
            msg = f"mutation_rate must be in [0, 1], got {self.mutation_rate}"
            raise ValueError(msg)

        if self.elitism_count < 0:
            msg = f"elitism_count must be >= 0, got {self.elitism_count}"
            raise ValueError(msg)

        if self.elitism_count >= self.population_size:
            msg = f"elitism_count ({self.elitism_count}) must be < population_size ({self.population_size})"
            raise ValueError(msg)

        if self.tournament_size < 1:
            msg = f"tournament_size must be >= 1, got {self.tournament_size}"
            raise ValueError(msg)

        if not 1.0 <= self.selection_pressure <= 2.0:
            msg = f"selection_pressure must be in [1, 2], got {self.selection_pressure}"
            raise ValueError(msg)

        if self.early_stopping_generations < 1:
            msg = f"early_stopping_generations must be >= 1, got {self.early_stopping_generations}"
            raise ValueError(msg)

        if self.cache_size < 0:
            msg = f"cache_size must be >= 0, got {self.cache_size}"
            raise ValueError(msg)


# =============================================================================
# PSO Configuration
# =============================================================================


@dataclass
class InertiaConfig:
    """Configuration for PSO inertia weight.

    Attributes:
        strategy: Inertia weight strategy.
        initial: Initial inertia weight.
        final: Final inertia weight (for decay strategies).
        decay_exponent: Exponent for nonlinear decay.
    """

    strategy: InertiaStrategy = InertiaStrategy.LINEAR_DECAY
    initial: float = 0.9
    final: float = 0.4
    decay_exponent: float = 2.0

    def __post_init__(self) -> None:
        """Validate inertia configuration."""
        if self.initial < 0:
            msg = f"initial inertia must be >= 0, got {self.initial}"
            raise ValueError(msg)

        if self.final < 0:
            msg = f"final inertia must be >= 0, got {self.final}"
            raise ValueError(msg)

        if self.decay_exponent <= 0:
            msg = f"decay_exponent must be > 0, got {self.decay_exponent}"
            raise ValueError(msg)

    def get_inertia(self, iteration: int, max_iterations: int) -> float:
        """Compute inertia weight for given iteration.

        Args:
            iteration: Current iteration (0-indexed).
            max_iterations: Maximum iterations.

        Returns:
            Inertia weight value.
        """
        if max_iterations <= 1:
            progress = 1.0
        else:
            progress = iteration / (max_iterations - 1)

        if self.strategy == InertiaStrategy.CONSTANT:
            return self.initial

        if self.strategy == InertiaStrategy.LINEAR_DECAY:
            return self.initial + progress * (self.final - self.initial)

        if self.strategy == InertiaStrategy.NONLINEAR_DECAY:
            decay = progress**self.decay_exponent
            return self.initial + decay * (self.final - self.initial)

        # Default to linear decay
        return self.initial + progress * (self.final - self.initial)


@dataclass
class PSOConfig:
    """Configuration for Particle Swarm Optimization.

    Attributes:
        swarm_size: Number of particles in swarm.
        max_iterations: Maximum PSO iterations.
        inertia: Inertia weight configuration.
        cognitive_coef: Cognitive coefficient (c1).
        social_coef: Social coefficient (c2).
        velocity_clamp: Velocity clamping factor (fraction of range).
        boundary_handling: How to handle particles at boundaries.
        local_search_enabled: Enable local search refinement.
        local_search_iterations: Local search iterations per particle.
        convergence_threshold: Stop if swarm converges within threshold.
        initialization_method: Swarm initialization method.
    """

    swarm_size: int = 30
    max_iterations: int = 50
    inertia: InertiaConfig = field(default_factory=InertiaConfig)
    cognitive_coef: float = 2.0
    social_coef: float = 2.0
    velocity_clamp: float = 0.5
    boundary_handling: BoundaryHandling = BoundaryHandling.CLAMP
    local_search_enabled: bool = False
    local_search_iterations: int = 5
    convergence_threshold: float = 1e-8
    initialization_method: InitializationMethod = InitializationMethod.RANDOM

    def __post_init__(self) -> None:
        """Validate PSO configuration."""
        if self.swarm_size < 2:
            msg = f"swarm_size must be >= 2, got {self.swarm_size}"
            raise ValueError(msg)

        if self.max_iterations < 1:
            msg = f"max_iterations must be >= 1, got {self.max_iterations}"
            raise ValueError(msg)

        if self.cognitive_coef < 0:
            msg = f"cognitive_coef must be >= 0, got {self.cognitive_coef}"
            raise ValueError(msg)

        if self.social_coef < 0:
            msg = f"social_coef must be >= 0, got {self.social_coef}"
            raise ValueError(msg)

        if not 0.0 < self.velocity_clamp <= 1.0:
            msg = f"velocity_clamp must be in (0, 1], got {self.velocity_clamp}"
            raise ValueError(msg)

        if self.local_search_iterations < 0:
            msg = f"local_search_iterations must be >= 0, got {self.local_search_iterations}"
            raise ValueError(msg)

        if self.convergence_threshold < 0:
            msg = f"convergence_threshold must be >= 0, got {self.convergence_threshold}"
            raise ValueError(msg)

        # Ensure inertia is proper type
        if isinstance(self.inertia, dict):
            self.inertia = InertiaConfig(**self.inertia)


# =============================================================================
# Cross-Validation Configuration
# =============================================================================


@dataclass
class CVConfig:
    """Configuration for cross-validation.

    Attributes:
        outer_cv: Number of outer CV folds.
        inner_cv: Number of inner CV folds.
        stratified: Use stratified splitting for classification.
        shuffle: Shuffle data before splitting.
        scoring: Scoring metric name.
        n_jobs: Number of parallel jobs (-1 = all CPUs).
        refit: Refit best model on full training data.
    """

    outer_cv: int = 5
    inner_cv: int = 3
    stratified: bool = True
    shuffle: bool = True
    scoring: str = "accuracy"
    n_jobs: int = -1
    refit: bool = True

    def __post_init__(self) -> None:
        """Validate CV configuration."""
        if self.outer_cv < 2:
            msg = f"outer_cv must be >= 2, got {self.outer_cv}"
            raise ValueError(msg)

        if self.inner_cv < 2:
            msg = f"inner_cv must be >= 2, got {self.inner_cv}"
            raise ValueError(msg)


# =============================================================================
# Main Optimization Configuration
# =============================================================================


@dataclass
class OptimizationConfig:
    """Main configuration for the optimization framework.

    Attributes:
        gpu_config: GPU acceleration settings.
        ga_config: Genetic Algorithm settings.
        pso_config: Particle Swarm Optimization settings.
        cv_config: Cross-validation settings.
        feature_config: Feature selection settings.
        random_seed: Random seed for reproducibility.
        n_jobs: Number of parallel jobs for CPU operations.
        verbosity: Verbosity level (0=silent, 1=progress, 2=detailed).
        checkpoint_dir: Directory for saving checkpoints.
        checkpoint_frequency: Save checkpoint every N generations.
        log_file: Log file path (None = no file logging).
        direction: Optimization direction.
    """

    gpu_config: GPUConfig = field(default_factory=GPUConfig)
    ga_config: GAConfig = field(default_factory=GAConfig)
    pso_config: PSOConfig = field(default_factory=PSOConfig)
    cv_config: CVConfig = field(default_factory=CVConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    random_seed: int | None = None
    n_jobs: int = -1
    verbosity: int = 1
    checkpoint_dir: str | None = None
    checkpoint_frequency: int = 10
    log_file: str | None = None
    direction: OptimizationDirection = OptimizationDirection.MAXIMIZE

    def __post_init__(self) -> None:
        """Validate and convert nested configs."""
        # Convert dicts to dataclass instances
        if isinstance(self.gpu_config, dict):
            self.gpu_config = GPUConfig(**self.gpu_config)
        if isinstance(self.ga_config, dict):
            self.ga_config = GAConfig(**self.ga_config)
        if isinstance(self.pso_config, dict):
            self.pso_config = PSOConfig(**self.pso_config)
        if isinstance(self.cv_config, dict):
            self.cv_config = CVConfig(**self.cv_config)
        if isinstance(self.feature_config, dict):
            self.feature_config = FeatureConfig(**self.feature_config)

        # Validate
        if self.verbosity < 0:
            msg = f"verbosity must be >= 0, got {self.verbosity}"
            raise ValueError(msg)

        if self.checkpoint_frequency < 1:
            msg = f"checkpoint_frequency must be >= 1, got {self.checkpoint_frequency}"
            raise ValueError(msg)

    @property
    def maximize(self) -> bool:
        """Whether optimizing for maximum."""
        return self.direction == OptimizationDirection.MAXIMIZE

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationConfig:
        """Create configuration from dictionary."""
        # Handle enum conversions
        if "direction" in data and isinstance(data["direction"], str):
            data["direction"] = OptimizationDirection[data["direction"].upper()]

        # Handle nested configs
        if "ga_config" in data:
            ga_data = data["ga_config"]
            for key in ["selection_method", "crossover_method", "mutation_method", "initialization_method"]:
                if key in ga_data and isinstance(ga_data[key], str):
                    enum_cls = {
                        "selection_method": SelectionMethod,
                        "crossover_method": CrossoverMethod,
                        "mutation_method": MutationMethod,
                        "initialization_method": InitializationMethod,
                    }[key]
                    ga_data[key] = enum_cls[ga_data[key].upper()]

        if "pso_config" in data:
            pso_data = data["pso_config"]
            if "boundary_handling" in pso_data and isinstance(pso_data["boundary_handling"], str):
                pso_data["boundary_handling"] = BoundaryHandling[pso_data["boundary_handling"].upper()]
            if "initialization_method" in pso_data and isinstance(pso_data["initialization_method"], str):
                pso_data["initialization_method"] = InitializationMethod[pso_data["initialization_method"].upper()]
            if "inertia" in pso_data and isinstance(pso_data["inertia"], dict):
                inertia_data = pso_data["inertia"]
                if "strategy" in inertia_data and isinstance(inertia_data["strategy"], str):
                    inertia_data["strategy"] = InertiaStrategy[inertia_data["strategy"].upper()]

        return cls(**data)

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize configuration to JSON.

        Args:
            path: Optional file path to save to.

        Returns:
            JSON string representation.
        """

        def _serialize(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.name
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            if hasattr(obj, "__dict__"):
                return {k: _serialize(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_serialize(item) for item in obj]
            return obj

        data = _serialize(self.to_dict())
        json_str = json.dumps(data, indent=2)

        if path is not None:
            Path(path).write_text(json_str)

        return json_str

    @classmethod
    def from_json(cls, json_str: str | None = None, path: str | Path | None = None) -> OptimizationConfig:
        """Load configuration from JSON.

        Args:
            json_str: JSON string.
            path: File path to load from.

        Returns:
            OptimizationConfig instance.
        """
        if path is not None:
            json_str = Path(path).read_text()

        if json_str is None:
            msg = "Either json_str or path must be provided"
            raise ValueError(msg)

        data = json.loads(json_str)
        return cls.from_dict(data)

    def copy(self) -> OptimizationConfig:
        """Create a deep copy of configuration."""
        return OptimizationConfig.from_dict(self.to_dict())

    def with_updates(self, **kwargs: Any) -> OptimizationConfig:
        """Create copy with updated values.

        Args:
            **kwargs: Values to update.

        Returns:
            New OptimizationConfig with updates.
        """
        data = self.to_dict()
        data.update(kwargs)
        return OptimizationConfig.from_dict(data)


# =============================================================================
# Configuration Presets
# =============================================================================


def get_quick_config() -> OptimizationConfig:
    """Get configuration preset for quick experiments.

    Small population, few generations, minimal overhead.
    """
    return OptimizationConfig(
        ga_config=GAConfig(
            population_size=20,
            max_generations=30,
            elitism_count=1,
            early_stopping_generations=10,
        ),
        pso_config=PSOConfig(
            swarm_size=15,
            max_iterations=20,
        ),
        cv_config=CVConfig(
            outer_cv=3,
            inner_cv=2,
        ),
        verbosity=1,
    )


def get_thorough_config() -> OptimizationConfig:
    """Get configuration preset for thorough optimization.

    Large population, many generations, extensive search.
    """
    return OptimizationConfig(
        ga_config=GAConfig(
            population_size=100,
            max_generations=200,
            elitism_count=5,
            early_stopping_generations=40,
            initialization_method=InitializationMethod.SOBOL,
        ),
        pso_config=PSOConfig(
            swarm_size=50,
            max_iterations=100,
            local_search_enabled=True,
        ),
        cv_config=CVConfig(
            outer_cv=10,
            inner_cv=5,
        ),
        verbosity=2,
    )


def get_gpu_optimized_config() -> OptimizationConfig:
    """Get configuration optimized for GPU acceleration.

    Large batch sizes to maximize GPU utilization.
    """
    return OptimizationConfig(
        gpu_config=GPUConfig(
            use_gpu=True,
            use_float32=True,  # Save memory
            memory_limit=0.9,
            min_array_size_for_gpu=500,
        ),
        ga_config=GAConfig(
            population_size=200,
            max_generations=100,
            elitism_count=5,
        ),
        pso_config=PSOConfig(
            swarm_size=100,
            max_iterations=50,
        ),
        verbosity=1,
    )


def get_memory_efficient_config() -> OptimizationConfig:
    """Get configuration for memory-constrained environments."""
    return OptimizationConfig(
        gpu_config=GPUConfig(
            use_gpu=True,
            use_float32=True,
            memory_limit=0.6,
        ),
        ga_config=GAConfig(
            population_size=30,
            max_generations=50,
            elitism_count=2,
        ),
        pso_config=PSOConfig(
            swarm_size=20,
            max_iterations=30,
        ),
        cv_config=CVConfig(
            outer_cv=3,
            inner_cv=2,
        ),
    )


# =============================================================================
# Configuration Validation Utilities
# =============================================================================


def validate_config_for_data(
    config: OptimizationConfig,
    n_samples: int,
    n_features: int,
) -> list[str]:
    """Validate configuration against dataset characteristics.

    Args:
        config: Configuration to validate.
        n_samples: Number of samples in dataset.
        n_features: Number of features in dataset.

    Returns:
        List of warning messages (empty if all OK).
    """
    warnings_list: list[str] = []

    # Check CV configuration
    if config.cv_config.outer_cv > n_samples // 10:
        warnings_list.append(
            f"outer_cv ({config.cv_config.outer_cv}) may be too high "
            f"for {n_samples} samples"
        )

    # Check feature configuration
    if config.feature_config.min_features > n_features:
        warnings_list.append(
            f"min_features ({config.feature_config.min_features}) exceeds "
            f"n_features ({n_features})"
        )

    if config.feature_config.max_features is not None:
        if config.feature_config.max_features > n_features:
            warnings_list.append(
                f"max_features ({config.feature_config.max_features}) exceeds "
                f"n_features ({n_features})"
            )

    # Check GPU memory if enabled
    if config.gpu_config.use_gpu:
        memory_est = config.gpu_config.estimate_memory(
            n_samples=n_samples,
            n_features=n_features,
            pop_size=config.ga_config.population_size,
            swarm_size=config.pso_config.swarm_size,
            n_discrete=10,  # Estimate
            n_continuous=10,  # Estimate
        )
        total_gb = memory_est["total_memory"] / (1024**3)
        if total_gb > 8:  # Arbitrary threshold
            warnings_list.append(
                f"Estimated GPU memory usage ({total_gb:.2f} GB) is high. "
                "Consider reducing population/swarm size or using float32."
            )

    return warnings_list

