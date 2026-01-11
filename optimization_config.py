"""Optimization Configuration Module.

This module provides detailed configuration classes for the hybrid GA-PSO
optimization framework, allowing fine-grained control over:
- Genetic Algorithm behavior (selection, crossover, mutation)
- Particle Swarm Optimization behavior (inertia, velocity, coefficients)
- Feature selection constraints
- Caching and warm-start strategies

Example:
    >>> from optimization_config import (
    ...     GASettings, PSOSettings, FeatureSelectionSettings,
    ...     CrossoverType, MutationType, SelectionType,
    ...     InertiaStrategy, BoundaryHandling,
    ... )
    >>>
    >>> # Custom GA settings
    >>> ga_settings = GASettings(
    ...     population_size=100,
    ...     crossover_type=CrossoverType.BLEND,
    ...     crossover_rate=0.85,
    ...     mutation_type=MutationType.ADAPTIVE,
    ...     selection_type=SelectionType.TOURNAMENT,
    ...     tournament_size=5,
    ... )
    >>>
    >>> # Custom PSO settings
    >>> pso_settings = PSOSettings(
    ...     swarm_size=50,
    ...     inertia_strategy=InertiaStrategy.LINEAR_DECAY,
    ...     inertia_start=0.9,
    ...     inertia_end=0.4,
    ...     cognitive_coef=2.0,
    ...     social_coef=2.0,
    ...     velocity_clamp=0.5,
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence



class CrossoverType(Enum):
    """Crossover operator types for Genetic Algorithm."""

    UNIFORM = "uniform"
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    BLEND = "blend"
    SIMULATED_BINARY = "sbx"
    ARITHMETIC = "arithmetic"

    @classmethod
    def from_string(cls, value: str) -> CrossoverType:
        """Create from string value."""
        mapping = {
            "uniform": cls.UNIFORM,
            "single_point": cls.SINGLE_POINT,
            "two_point": cls.TWO_POINT,
            "blend": cls.BLEND,
            "sbx": cls.SIMULATED_BINARY,
            "simulated_binary": cls.SIMULATED_BINARY,
            "arithmetic": cls.ARITHMETIC,
        }
        return mapping.get(value.lower(), cls.BLEND)


class MutationType(Enum):
    """Mutation operator types for Genetic Algorithm."""

    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    POLYNOMIAL = "polynomial"
    ADAPTIVE = "adaptive"
    BOUNDARY = "boundary"

    @classmethod
    def from_string(cls, value: str) -> MutationType:
        """Create from string value."""
        mapping = {
            "uniform": cls.UNIFORM,
            "gaussian": cls.GAUSSIAN,
            "polynomial": cls.POLYNOMIAL,
            "adaptive": cls.ADAPTIVE,
            "boundary": cls.BOUNDARY,
        }
        return mapping.get(value.lower(), cls.ADAPTIVE)


class SelectionType(Enum):
    """Selection strategy types for Genetic Algorithm."""

    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    TRUNCATION = "truncation"
    ELITIST = "elitist"
    STOCHASTIC_UNIVERSAL = "sus"

    @classmethod
    def from_string(cls, value: str) -> SelectionType:
        """Create from string value."""
        mapping = {
            "tournament": cls.TOURNAMENT,
            "roulette": cls.ROULETTE,
            "rank": cls.RANK,
            "truncation": cls.TRUNCATION,
            "elitist": cls.ELITIST,
            "sus": cls.STOCHASTIC_UNIVERSAL,
            "stochastic_universal": cls.STOCHASTIC_UNIVERSAL,
        }
        return mapping.get(value.lower(), cls.TOURNAMENT)


class InertiaStrategy(Enum):
    """Inertia weight strategies for PSO."""

    CONSTANT = "constant"
    LINEAR_DECAY = "linear_decay"
    NONLINEAR_DECAY = "nonlinear_decay"
    ADAPTIVE = "adaptive"
    CHAOTIC = "chaotic"
    RANDOM = "random"

    @classmethod
    def from_string(cls, value: str) -> InertiaStrategy:
        """Create from string value."""
        mapping = {
            "constant": cls.CONSTANT,
            "linear": cls.LINEAR_DECAY,
            "linear_decay": cls.LINEAR_DECAY,
            "nonlinear": cls.NONLINEAR_DECAY,
            "nonlinear_decay": cls.NONLINEAR_DECAY,
            "adaptive": cls.ADAPTIVE,
            "chaotic": cls.CHAOTIC,
            "random": cls.RANDOM,
        }
        return mapping.get(value.lower(), cls.LINEAR_DECAY)


class BoundaryHandling(Enum):
    """Boundary handling strategies for PSO."""

    CLAMP = "clamp"
    REFLECT = "reflect"
    WRAP = "wrap"
    RANDOM = "random"
    ABSORB = "absorb"

    @classmethod
    def from_string(cls, value: str) -> BoundaryHandling:
        """Create from string value."""
        mapping = {
            "clamp": cls.CLAMP,
            "reflect": cls.REFLECT,
            "wrap": cls.WRAP,
            "random": cls.RANDOM,
            "absorb": cls.ABSORB,
        }
        return mapping.get(value.lower(), cls.CLAMP)


class InitializationMethod(Enum):
    """Population/swarm initialization methods."""

    RANDOM = "random"
    LATIN_HYPERCUBE = "lhs"
    SOBOL = "sobol"
    HALTON = "halton"
    ORTHOGONAL = "orthogonal"
    UNIFORM_GRID = "grid"

    @classmethod
    def from_string(cls, value: str) -> InitializationMethod:
        """Create from string value."""
        mapping = {
            "random": cls.RANDOM,
            "lhs": cls.LATIN_HYPERCUBE,
            "latin_hypercube": cls.LATIN_HYPERCUBE,
            "sobol": cls.SOBOL,
            "halton": cls.HALTON,
            "orthogonal": cls.ORTHOGONAL,
            "grid": cls.UNIFORM_GRID,
        }
        return mapping.get(value.lower(), cls.RANDOM)




@dataclass
class GASettings:
    """Comprehensive Genetic Algorithm configuration.

    Attributes:
        population_size: Number of individuals in population.
        max_generations: Maximum number of generations.
        crossover_type: Type of crossover operator.
        crossover_rate: Probability of crossover.
        crossover_alpha: Alpha parameter for blend crossover (0.0-1.0).
        crossover_eta: Eta parameter for SBX crossover.
        mutation_type: Type of mutation operator.
        mutation_rate: Base probability of mutation per gene.
        mutation_sigma: Standard deviation for Gaussian mutation.
        mutation_eta: Eta parameter for polynomial mutation.
        mutation_adaptive: Whether to use adaptive mutation rates.
        selection_type: Type of selection strategy.
        tournament_size: Tournament size for tournament selection.
        truncation_fraction: Fraction to keep for truncation selection.
        elitism_count: Number of best individuals preserved each generation.
        initialization_method: How to initialize population.
        early_stopping_generations: Generations without improvement to stop.
        early_stopping_tolerance: Minimum improvement to continue.
        diversity_threshold: Minimum diversity before restart.
        restart_on_stagnation: Whether to restart if stuck.

    Example:
        >>> ga = GASettings(
        ...     population_size=100,
        ...     crossover_type=CrossoverType.BLEND,
        ...     mutation_type=MutationType.ADAPTIVE,
        ...     selection_type=SelectionType.TOURNAMENT,
        ...     tournament_size=5,
        ... )
    """

    # Population settings
    population_size: int = 50
    max_generations: int = 100
    initialization_method: InitializationMethod = InitializationMethod.RANDOM

    # Crossover settings
    crossover_type: CrossoverType = CrossoverType.BLEND
    crossover_rate: float = 0.8
    crossover_alpha: float = 0.5  # For blend crossover
    crossover_eta: float = 20.0  # For SBX crossover

    # Mutation settings
    mutation_type: MutationType = MutationType.ADAPTIVE
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.1  # For Gaussian mutation
    mutation_eta: float = 20.0  # For polynomial mutation
    mutation_adaptive: bool = True  # Adapt rates during evolution

    # Selection settings
    selection_type: SelectionType = SelectionType.TOURNAMENT
    tournament_size: int = 3
    truncation_fraction: float = 0.5  # For truncation selection
    elitism_count: int = 2

    # Early stopping
    early_stopping_generations: int = 20
    early_stopping_tolerance: float = 1e-6

    # Diversity maintenance
    diversity_threshold: float = 0.1
    restart_on_stagnation: bool = False

    def __post_init__(self) -> None:
        """Validate settings."""
        if self.population_size < 2:
            msg = f"population_size must be >= 2, got {self.population_size}"
            raise ValueError(msg)

        if self.max_generations < 1:
            msg = f"max_generations must be >= 1, got {self.max_generations}"
            raise ValueError(msg)

        if not 0 <= self.crossover_rate <= 1:
            msg = f"crossover_rate must be in [0, 1], got {self.crossover_rate}"
            raise ValueError(msg)

        if not 0 <= self.mutation_rate <= 1:
            msg = f"mutation_rate must be in [0, 1], got {self.mutation_rate}"
            raise ValueError(msg)

        if self.tournament_size < 2:
            msg = f"tournament_size must be >= 2, got {self.tournament_size}"
            raise ValueError(msg)

        if self.elitism_count >= self.population_size:
            msg = f"elitism_count ({self.elitism_count}) must be < population_size ({self.population_size})"
            raise ValueError(msg)

        if isinstance(self.crossover_type, str):
            self.crossover_type = CrossoverType.from_string(self.crossover_type)
        if isinstance(self.mutation_type, str):
            self.mutation_type = MutationType.from_string(self.mutation_type)
        if isinstance(self.selection_type, str):
            self.selection_type = SelectionType.from_string(self.selection_type)
        if isinstance(self.initialization_method, str):
            self.initialization_method = InitializationMethod.from_string(self.initialization_method)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "initialization_method": self.initialization_method.value,
            "crossover_type": self.crossover_type.value,
            "crossover_rate": self.crossover_rate,
            "crossover_alpha": self.crossover_alpha,
            "crossover_eta": self.crossover_eta,
            "mutation_type": self.mutation_type.value,
            "mutation_rate": self.mutation_rate,
            "mutation_sigma": self.mutation_sigma,
            "mutation_eta": self.mutation_eta,
            "mutation_adaptive": self.mutation_adaptive,
            "selection_type": self.selection_type.value,
            "tournament_size": self.tournament_size,
            "truncation_fraction": self.truncation_fraction,
            "elitism_count": self.elitism_count,
            "early_stopping_generations": self.early_stopping_generations,
            "early_stopping_tolerance": self.early_stopping_tolerance,
            "diversity_threshold": self.diversity_threshold,
            "restart_on_stagnation": self.restart_on_stagnation,
        }



@dataclass
class PSOSettings:
    """Comprehensive Particle Swarm Optimization configuration.

    Attributes:
        swarm_size: Number of particles in swarm.
        max_iterations: Maximum PSO iterations.
        inertia_strategy: Strategy for inertia weight.
        inertia_start: Initial inertia weight.
        inertia_end: Final inertia weight (for decay strategies).
        inertia_constant: Constant inertia value (if constant strategy).
        cognitive_coef: Cognitive (personal best) coefficient (c1).
        social_coef: Social (global best) coefficient (c2).
        velocity_clamp: Velocity clamping factor (fraction of search range).
        max_velocity: Maximum absolute velocity (None for auto).
        boundary_handling: How to handle particles at boundaries.
        use_constriction: Use Clerc's constriction factor.
        local_search_probability: Probability of local search around best.
        initialization_method: How to initialize swarm.
        early_stopping_iterations: Iterations without improvement to stop.
        early_stopping_tolerance: Minimum improvement to continue.

    Example:
        >>> pso = PSOSettings(
        ...     swarm_size=50,
        ...     inertia_strategy=InertiaStrategy.LINEAR_DECAY,
        ...     inertia_start=0.9,
        ...     inertia_end=0.4,
        ...     cognitive_coef=2.0,
        ...     social_coef=2.0,
        ...     velocity_clamp=0.5,
        ... )
    """

    # Swarm settings
    swarm_size: int = 30
    max_iterations: int = 50
    initialization_method: InitializationMethod = InitializationMethod.RANDOM

    # Inertia settings
    inertia_strategy: InertiaStrategy = InertiaStrategy.LINEAR_DECAY
    inertia_start: float = 0.9
    inertia_end: float = 0.4
    inertia_constant: float = 0.7  # For constant strategy

    # Velocity coefficients
    cognitive_coef: float = 2.0  # c1 - personal best attraction
    social_coef: float = 2.0  # c2 - global best attraction

    # Velocity constraints
    velocity_clamp: float = 0.5  # Fraction of search range
    max_velocity: float | None = None  # Absolute max (overrides clamp if set)

    # Boundary handling
    boundary_handling: BoundaryHandling = BoundaryHandling.CLAMP

    # Advanced settings
    use_constriction: bool = False  # Clerc's constriction factor
    local_search_probability: float = 0.0  # Probability of local refinement

    # Early stopping
    early_stopping_iterations: int = 15
    early_stopping_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        """Validate settings."""
        if self.swarm_size < 2:
            msg = f"swarm_size must be >= 2, got {self.swarm_size}"
            raise ValueError(msg)

        if self.max_iterations < 1:
            msg = f"max_iterations must be >= 1, got {self.max_iterations}"
            raise ValueError(msg)

        if not 0 <= self.inertia_start <= 1:
            msg = f"inertia_start must be in [0, 1], got {self.inertia_start}"
            raise ValueError(msg)

        if not 0 <= self.inertia_end <= 1:
            msg = f"inertia_end must be in [0, 1], got {self.inertia_end}"
            raise ValueError(msg)

        if self.cognitive_coef < 0:
            msg = f"cognitive_coef must be >= 0, got {self.cognitive_coef}"
            raise ValueError(msg)

        if self.social_coef < 0:
            msg = f"social_coef must be >= 0, got {self.social_coef}"
            raise ValueError(msg)

        if not 0 < self.velocity_clamp <= 1:
            msg = f"velocity_clamp must be in (0, 1], got {self.velocity_clamp}"
            raise ValueError(msg)

        # Convert string types if needed
        if isinstance(self.inertia_strategy, str):
            self.inertia_strategy = InertiaStrategy.from_string(self.inertia_strategy)
        if isinstance(self.boundary_handling, str):
            self.boundary_handling = BoundaryHandling.from_string(self.boundary_handling)
        if isinstance(self.initialization_method, str):
            self.initialization_method = InitializationMethod.from_string(self.initialization_method)

    def get_inertia(self, iteration: int, max_iterations: int) -> float:
        """Calculate inertia weight for given iteration.

        Args:
            iteration: Current iteration (0-indexed).
            max_iterations: Total iterations.

        Returns:
            Inertia weight value.
        """
        if self.inertia_strategy == InertiaStrategy.CONSTANT:
            return self.inertia_constant

        if self.inertia_strategy == InertiaStrategy.LINEAR_DECAY:
            progress = iteration / max(1, max_iterations - 1)
            return self.inertia_start - progress * (self.inertia_start - self.inertia_end)

        if self.inertia_strategy == InertiaStrategy.NONLINEAR_DECAY:
            progress = iteration / max(1, max_iterations - 1)
            return self.inertia_end + (self.inertia_start - self.inertia_end) * (1 - progress) ** 2

        if self.inertia_strategy == InertiaStrategy.ADAPTIVE:
            # Placeholder
            progress = iteration / max(1, max_iterations - 1)
            return self.inertia_start - progress * (self.inertia_start - self.inertia_end)

        if self.inertia_strategy == InertiaStrategy.CHAOTIC:
            # Logistic map chaotic sequence
            z = 0.7  # Initial value
            for _ in range(iteration + 1):
                z = 4.0 * z * (1.0 - z)
            return self.inertia_end + z * (self.inertia_start - self.inertia_end)

        if self.inertia_strategy == InertiaStrategy.RANDOM:
            return np.random.uniform(self.inertia_end, self.inertia_start)

        return self.inertia_constant

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "swarm_size": self.swarm_size,
            "max_iterations": self.max_iterations,
            "initialization_method": self.initialization_method.value,
            "inertia_strategy": self.inertia_strategy.value,
            "inertia_start": self.inertia_start,
            "inertia_end": self.inertia_end,
            "inertia_constant": self.inertia_constant,
            "cognitive_coef": self.cognitive_coef,
            "social_coef": self.social_coef,
            "velocity_clamp": self.velocity_clamp,
            "max_velocity": self.max_velocity,
            "boundary_handling": self.boundary_handling.value,
            "use_constriction": self.use_constriction,
            "local_search_probability": self.local_search_probability,
            "early_stopping_iterations": self.early_stopping_iterations,
            "early_stopping_tolerance": self.early_stopping_tolerance,
        }




@dataclass
class FeatureSelectionSettings:
    """Configuration for feature selection behavior.

    Attributes:
        enabled: Whether to perform feature selection.
        min_features: Minimum number of features to select.
        max_features: Maximum number of features to select.
        initial_selection_prob: Initial probability of selecting each feature.
        feature_mutation_rate: Mutation rate for feature genes.
        use_feature_importance: Use model-based importance for initialization.
        importance_threshold: Threshold for importance-based selection.
        grouped_features: Groups of features that should be selected together.
        forbidden_combinations: Feature combinations that are not allowed.

    Example:
        >>> fs = FeatureSelectionSettings(
        ...     enabled=True,
        ...     min_features=5,
        ...     max_features=50,
        ...     initial_selection_prob=0.5,
        ... )
    """

    enabled: bool = True
    min_features: int | None = None
    max_features: int | None = None
    initial_selection_prob: float = 0.5
    feature_mutation_rate: float = 0.05
    use_feature_importance: bool = False
    importance_threshold: float = 0.01
    grouped_features: list[list[int]] | None = None
    forbidden_combinations: list[tuple[int, ...]] | None = None

    def __post_init__(self) -> None:
        """Validate settings."""
        if not 0 <= self.initial_selection_prob <= 1:
            msg = f"initial_selection_prob must be in [0, 1], got {self.initial_selection_prob}"
            raise ValueError(msg)

        if not 0 <= self.feature_mutation_rate <= 1:
            msg = f"feature_mutation_rate must be in [0, 1], got {self.feature_mutation_rate}"
            raise ValueError(msg)

        if self.min_features is not None and self.max_features is not None:
            if self.min_features > self.max_features:
                msg = f"min_features ({self.min_features}) cannot exceed max_features ({self.max_features})"
                raise ValueError(msg)

    def get_bounds(self, n_features: int) -> tuple[int, int]:
        """Get actual min/max features for a given total.

        Args:
            n_features: Total number of features available.

        Returns:
            Tuple of (min_features, max_features).
        """
        min_f = self.min_features if self.min_features is not None else max(1, n_features // 10)
        max_f = self.max_features if self.max_features is not None else n_features
        return min_f, min(max_f, n_features)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "min_features": self.min_features,
            "max_features": self.max_features,
            "initial_selection_prob": self.initial_selection_prob,
            "feature_mutation_rate": self.feature_mutation_rate,
            "use_feature_importance": self.use_feature_importance,
            "importance_threshold": self.importance_threshold,
            "grouped_features": self.grouped_features,
            "forbidden_combinations": self.forbidden_combinations,
        }



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
class ParameterSpec:
    """Specification for a single hyperparameter.

    Attributes:
        name: Parameter name.
        param_type: Type ('continuous', 'integer', 'categorical', 'boolean').
        lower: Lower bound (for continuous/integer).
        upper: Upper bound (for continuous/integer).
        choices: List of choices (for categorical).
        log_scale: Whether to use log scale (for continuous/integer).
        default: Default value.

    Example:
        >>> # Continuous parameter
        >>> learning_rate = ParameterSpec(
        ...     name='learning_rate',
        ...     param_type='continuous',
        ...     lower=0.001,
        ...     upper=0.1,
        ...     log_scale=True,
        ... )
        >>>
        >>> # Categorical parameter
        >>> kernel = ParameterSpec(
        ...     name='kernel',
        ...     param_type='categorical',
        ...     choices=['rbf', 'linear', 'poly'],
        ... )
    """

    name: str
    param_type: str  # 'continuous', 'integer', 'categorical', 'boolean'
    lower: float | None = None
    upper: float | None = None
    choices: list[Any] | None = None
    log_scale: bool = False
    default: Any = None

    def __post_init__(self) -> None:
        """Validate specification."""
        valid_types = {"continuous", "integer", "categorical", "boolean"}
        if self.param_type not in valid_types:
            msg = f"param_type must be one of {valid_types}, got {self.param_type}"
            raise ValueError(msg)

        if self.param_type in {"continuous", "integer"}:
            if self.lower is None or self.upper is None:
                msg = f"lower and upper bounds required for {self.param_type} parameter"
                raise ValueError(msg)
            if self.lower >= self.upper:
                msg = f"lower ({self.lower}) must be < upper ({self.upper})"
                raise ValueError(msg)

        if self.param_type == "categorical":
            if not self.choices:
                msg = "choices required for categorical parameter"
                raise ValueError(msg)

    @property
    def is_discrete(self) -> bool:
        """Check if parameter is discrete (GA-optimized)."""
        return self.param_type in {"integer", "categorical", "boolean"}

    @property
    def is_continuous(self) -> bool:
        """Check if parameter is continuous (PSO-optimized)."""
        return self.param_type == "continuous"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "name": self.name,
            "type": self.param_type,
        }

        if self.param_type in {"continuous", "integer"}:
            result["range"] = [self.lower, self.upper]
            if self.log_scale:
                result["log_scale"] = True

        if self.param_type == "categorical":
            result["choices"] = self.choices

        if self.default is not None:
            result["default"] = self.default

        return result


@dataclass
class CustomParameterSpace:
    """Custom parameter space for model optimization.

    Allows defining arbitrary hyperparameters for any model.

    Attributes:
        parameters: List of parameter specifications.
        fixed_params: Parameters with fixed values.

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>>
        >>> space = CustomParameterSpace(
        ...     parameters=[
        ...         ParameterSpec('n_estimators', 'integer', 50, 500),
        ...         ParameterSpec('max_depth', 'integer', 3, 30),
        ...         ParameterSpec('max_features', 'continuous', 0.1, 1.0),
        ...         ParameterSpec('criterion', 'categorical', choices=['gini', 'entropy']),
        ...     ],
        ...     fixed_params={'random_state': 42, 'n_jobs': -1},
        ... )
        >>>
        >>> result = optimize(X, y, model=RandomForestClassifier, param_space=space)
    """

    parameters: list[ParameterSpec] = field(default_factory=list)
    fixed_params: dict[str, Any] = field(default_factory=dict)

    def add_continuous(
        self,
        name: str,
        lower: float,
        upper: float,
        log_scale: bool = False,
        default: float | None = None,
    ) -> CustomParameterSpace:
        """Add a continuous parameter.

        Args:
            name: Parameter name.
            lower: Lower bound.
            upper: Upper bound.
            log_scale: Use log scale.
            default: Default value.

        Returns:
            Self for chaining.
        """
        self.parameters.append(
            ParameterSpec(
                name=name,
                param_type="continuous",
                lower=lower,
                upper=upper,
                log_scale=log_scale,
                default=default,
            )
        )
        return self

    def add_integer(
        self,
        name: str,
        lower: int,
        upper: int,
        log_scale: bool = False,
        default: int | None = None,
    ) -> CustomParameterSpace:
        """Add an integer parameter.

        Args:
            name: Parameter name.
            lower: Lower bound.
            upper: Upper bound.
            log_scale: Use log scale.
            default: Default value.

        Returns:
            Self for chaining.
        """
        self.parameters.append(
            ParameterSpec(
                name=name,
                param_type="integer",
                lower=float(lower),
                upper=float(upper),
                log_scale=log_scale,
                default=default,
            )
        )
        return self

    def add_categorical(
        self,
        name: str,
        choices: list[Any],
        default: Any | None = None,
    ) -> CustomParameterSpace:
        """Add a categorical parameter.

        Args:
            name: Parameter name.
            choices: List of possible values.
            default: Default value.

        Returns:
            Self for chaining.
        """
        self.parameters.append(
            ParameterSpec(
                name=name,
                param_type="categorical",
                choices=choices,
                default=default,
            )
        )
        return self

    def add_boolean(
        self,
        name: str,
        default: bool | None = None,
    ) -> CustomParameterSpace:
        """Add a boolean parameter.

        Args:
            name: Parameter name.
            default: Default value.

        Returns:
            Self for chaining.
        """
        self.parameters.append(
            ParameterSpec(
                name=name,
                param_type="boolean",
                choices=[False, True],
                default=default,
            )
        )
        return self

    def set_fixed(self, **params: Any) -> CustomParameterSpace:
        """Set fixed parameters.

        Args:
            **params: Parameter name-value pairs.

        Returns:
            Self for chaining.
        """
        self.fixed_params.update(params)
        return self

    @property
    def discrete_params(self) -> list[ParameterSpec]:
        """Get discrete (GA-optimized) parameters."""
        return [p for p in self.parameters if p.is_discrete]

    @property
    def continuous_params(self) -> list[ParameterSpec]:
        """Get continuous (PSO-optimized) parameters."""
        return [p for p in self.parameters if p.is_continuous]

    @property
    def n_discrete(self) -> int:
        """Number of discrete parameters."""
        return len(self.discrete_params)

    @property
    def n_continuous(self) -> int:
        """Number of continuous parameters."""
        return len(self.continuous_params)

    def get_discrete_bounds(self) -> list[tuple[int, int]]:
        """Get bounds for discrete parameters.

        Returns:
            List of (lower, upper) tuples for each discrete parameter.
        """
        bounds = []
        for p in self.discrete_params:
            if p.param_type == "integer":
                bounds.append((int(p.lower), int(p.upper)))  # pyright: ignore[reportArgumentType]
            elif p.param_type in {"categorical", "boolean"}:
                bounds.append((0, len(p.choices) - 1))  # pyright: ignore[reportArgumentType]
        return bounds

    def get_continuous_bounds(self) -> list[tuple[float, float]]:
        """Get bounds for continuous parameters.

        Returns:
            List of (lower, upper) tuples for each continuous parameter.
        """
        return [(p.lower, p.upper) for p in self.continuous_params]  # pyright: ignore[reportReturnType]

    def decode_discrete(self, values: NDArray[np.int64]) -> dict[str, Any]:
        """Decode discrete parameter values to actual values.

        Args:
            values: Array of discrete values (indices or integers).

        Returns:
            Dictionary of parameter names to actual values.
        """
        result = {}
        for i, p in enumerate(self.discrete_params):
            if i >= len(values):
                break
            val = int(values[i])
            if p.param_type == "integer":
                result[p.name] = int(p.lower) + val  # pyright: ignore[reportArgumentType]
            elif p.param_type in {"categorical", "boolean"}:
                result[p.name] = p.choices[val]  # pyright: ignore[reportOptionalSubscript]
        return result

    def decode_continuous(self, values: NDArray[np.float64]) -> dict[str, Any]:
        """Decode continuous parameter values.

        Args:
            values: Array of continuous values.

        Returns:
            Dictionary of parameter names to values.
        """
        result = {}
        for i, p in enumerate(self.continuous_params):
            if i >= len(values):
                break
            result[p.name] = float(values[i])
        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "parameters": [p.to_dict() for p in self.parameters],
            "fixed_params": self.fixed_params,
        }




@dataclass
class CachingSettings:
    """Configuration for result caching.

    Attributes:
        use_chromosome_cache: Cache results for (features, discrete params) combinations.
        chromosome_cache_size: Maximum chromosome cache entries.
        use_full_config_cache: Cache results for complete configurations.
        full_config_cache_size: Maximum full config cache entries.
        use_warm_start: Use previous results to warm-start PSO.
        warm_start_threshold: Similarity threshold for warm-starting.
        cache_precision: Decimal precision for cache key rounding.

    Example:
        >>> caching = CachingSettings(
        ...     use_chromosome_cache=True,
        ...     chromosome_cache_size=10000,
        ...     use_warm_start=True,
        ... )
    """

    use_chromosome_cache: bool = True
    chromosome_cache_size: int = 10000
    use_full_config_cache: bool = True
    full_config_cache_size: int = 100000
    use_warm_start: bool = True
    warm_start_threshold: float = 0.1
    cache_precision: int = 6

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "use_chromosome_cache": self.use_chromosome_cache,
            "chromosome_cache_size": self.chromosome_cache_size,
            "use_full_config_cache": self.use_full_config_cache,
            "full_config_cache_size": self.full_config_cache_size,
            "use_warm_start": self.use_warm_start,
            "warm_start_threshold": self.warm_start_threshold,
            "cache_precision": self.cache_precision,
        }



__all__ = [
    # Enums
    "CrossoverType",
    "MutationType",
    "SelectionType",
    "InertiaStrategy",
    "BoundaryHandling",
    "InitializationMethod",
    "GASettings",
    "PSOSettings",
    "FeatureSelectionSettings",
    "ParameterSpec",
    "CustomParameterSpace",
    "CachingSettings",
]