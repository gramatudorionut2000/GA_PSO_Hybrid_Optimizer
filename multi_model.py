from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, TypeVar

from genetic.parameter_space import ModelSpace
import numpy as np
from numpy.typing import NDArray
from .utils.common import TaskType
from .utils.backend import (
    ArrayLike,
    get_backend,
    is_gpu_available,
    to_numpy,
)
from .genetic_algorithm import GPUPopulation
from .utils.data import DeviceInfo

if TYPE_CHECKING:
    from collections.abc import Iterator
    from .evaluation import EvaluationPipeline

# Configure logging
logger = logging.getLogger(__name__)



class ParameterType(Enum):
    """Type of hyperparameter."""

    INTEGER = auto()
    CONTINUOUS = auto()
    CATEGORICAL = auto()
    BOOLEAN = auto()


ModelClassT = TypeVar("ModelClassT", bound=type)



@dataclass
class ParameterDefinition:
    """Definition for a single hyperparameter.

    Attributes:
        name: Parameter name.
        param_type: Type of parameter (INTEGER, CONTINUOUS, etc.).
        lower: Lower bound (for numeric types).
        upper: Upper bound (for numeric types).
        choices: Valid choices (for CATEGORICAL).
        log_scale: Whether to use log scale for sampling.
        default: Default value.
    """

    name: str
    param_type: ParameterType
    lower: float | int | None = None
    upper: float | int | None = None
    choices: list[Any] | None = None
    log_scale: bool = False
    default: Any = None

    def __post_init__(self) -> None:
        """Validate parameter definition."""
        if self.param_type == ParameterType.CATEGORICAL:
            if not self.choices:
                msg = f"Categorical parameter '{self.name}' requires choices"
                raise ValueError(msg)
        elif self.param_type in (ParameterType.INTEGER, ParameterType.CONTINUOUS):
            if self.lower is None or self.upper is None:
                msg = f"Numeric parameter '{self.name}' requires lower and upper bounds"
                raise ValueError(msg)
            if self.lower >= self.upper:
                msg = f"Lower bound must be less than upper bound for '{self.name}'"
                raise ValueError(msg)

    @property
    def n_values(self) -> int:
        """Number of discrete values for this parameter."""
        if self.param_type == ParameterType.CATEGORICAL:
            return len(self.choices) if self.choices else 0
        if self.param_type == ParameterType.INTEGER:
            return int(self.upper) - int(self.lower) + 1  # type: ignore[arg-type]
        if self.param_type == ParameterType.BOOLEAN:
            return 2
        return -1

    @property
    def is_discrete(self) -> bool:
        """Check if parameter is discrete."""
        return self.param_type in (
            ParameterType.INTEGER,
            ParameterType.CATEGORICAL,
            ParameterType.BOOLEAN,
        )

    @property
    def is_continuous(self) -> bool:
        """Check if parameter is continuous."""
        return self.param_type == ParameterType.CONTINUOUS

    def sample(self, rng: np.random.Generator | None = None) -> Any:
        """Sample a random value from this parameter space."""
        if rng is None:
            rng = np.random.default_rng()

        if self.param_type == ParameterType.CATEGORICAL:
            return rng.choice(self.choices)  # pyright: ignore[reportCallIssue, reportArgumentType]
        if self.param_type == ParameterType.BOOLEAN:
            return bool(rng.integers(0, 2))
        if self.param_type == ParameterType.INTEGER:
            return int(rng.integers(int(self.lower), int(self.upper) + 1))  # type: ignore[arg-type]
        if self.param_type == ParameterType.CONTINUOUS:
            if self.log_scale:
                log_val = rng.uniform(np.log(self.lower), np.log(self.upper))  # pyright: ignore[reportCallIssue, reportArgumentType]
                return float(np.exp(log_val))
            return float(rng.uniform(self.lower, self.upper))  # type: ignore[arg-type]
        return self.default

    def index_to_value(self, index: int) -> Any:
        """Convert discrete index to actual value."""
        if self.param_type == ParameterType.CATEGORICAL:
            return self.choices[index] if self.choices else None
        if self.param_type == ParameterType.BOOLEAN:
            return bool(index)
        if self.param_type == ParameterType.INTEGER:
            return int(self.lower) + index  # type: ignore[arg-type]
        msg = f"Cannot convert index for continuous parameter '{self.name}'"
        raise ValueError(msg)

    def value_to_index(self, value: Any) -> int:
        """Convert actual value to discrete index."""
        if self.param_type == ParameterType.CATEGORICAL:
            return self.choices.index(value) if self.choices else 0
        if self.param_type == ParameterType.BOOLEAN:
            return 1 if value else 0
        if self.param_type == ParameterType.INTEGER:
            return int(value) - int(self.lower)  # type: ignore[arg-type]
        msg = f"Cannot convert value for continuous parameter '{self.name}'"
        raise ValueError(msg)

    def normalize(self, value: float) -> float:
        """Normalize continuous value to [0, 1]."""
        if not self.is_continuous:
            msg = f"Cannot normalize non-continuous parameter '{self.name}'"
            raise ValueError(msg)

        if self.log_scale:
            log_val = np.log(max(value, float(self.lower)))  # type: ignore[arg-type]
            log_lower = np.log(float(self.lower))  # type: ignore[arg-type]
            log_upper = np.log(float(self.upper))  # type: ignore[arg-type]
            return float((log_val - log_lower) / (log_upper - log_lower))
        return float((value - self.lower) / (self.upper - self.lower))  # type: ignore[operator]

    def denormalize(self, normalized: float) -> float:
        """Denormalize value from [0, 1] to original range."""
        if not self.is_continuous:
            msg = f"Cannot denormalize non-continuous parameter '{self.name}'"
            raise ValueError(msg)

        if self.log_scale:
            log_lower = np.log(float(self.lower))  # type: ignore[arg-type]
            log_upper = np.log(float(self.upper))  # type: ignore[arg-type]
            log_val = log_lower + normalized * (log_upper - log_lower)
            return float(np.exp(log_val))
        return float(self.lower + normalized * (self.upper - self.lower))  # type: ignore[operator]


@dataclass
class ModelDefinition:
    """Extended model definition with GPU support.

    Attributes:
        name: Human-readable model name.
        model_class: sklearn or compatible model class.
        cuml_class: Optional cuML equivalent class for GPU acceleration.
        discrete_params: Dictionary of discrete parameter definitions.
        continuous_params: Dictionary of continuous parameter definitions.
        fixed_params: Parameters that are always fixed.
        supports_gpu: Whether model can use GPU data directly.
        task_type: Type of ML task this model handles.
        weight: Sampling weight for model selection.
    """

    name: str
    model_class: type
    cuml_class: type | None = None
    discrete_params: dict[str, ParameterDefinition] = field(default_factory=dict)
    continuous_params: dict[str, ParameterDefinition] = field(default_factory=dict)
    fixed_params: dict[str, Any] = field(default_factory=dict)
    supports_gpu: bool = False
    task_type: TaskType = TaskType.CLASSIFICATION
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate and process parameter definitions."""
        for name, param in self.discrete_params.items():
            if not param.is_discrete:
                msg = f"Parameter '{name}' in discrete_params is not discrete"
                raise ValueError(msg)

        for name, param in self.continuous_params.items():
            if not param.is_continuous:
                msg = f"Parameter '{name}' in continuous_params is not continuous"
                raise ValueError(msg)

        if self.cuml_class is not None:
            self.supports_gpu = True

    @property
    def n_discrete_params(self) -> int:
        """Number of discrete parameters."""
        return len(self.discrete_params)

    @property
    def n_continuous_params(self) -> int:
        """Number of continuous parameters."""
        return len(self.continuous_params)

    @property
    def discrete_param_names(self) -> list[str]:
        """Ordered list of discrete parameter names."""
        return list(self.discrete_params.keys())

    @property
    def continuous_param_names(self) -> list[str]:
        """Ordered list of continuous parameter names."""
        return list(self.continuous_params.keys())

    def get_model_class(self, use_gpu: bool = False) -> type:
        """Get appropriate model class based on GPU preference.

        Args:
            use_gpu: Whether to prefer GPU (cuML) model.

        Returns:
            Model class to use.
        """
        if use_gpu and self.cuml_class is not None:
            return self.cuml_class
        return self.model_class

    def get_discrete_bounds(self) -> list[tuple[int, int]]:
        """Get bounds for all discrete parameters as (min, max) tuples."""
        bounds = []
        for param in self.discrete_params.values():
            if param.param_type == ParameterType.CATEGORICAL:
                bounds.append((0, len(param.choices) - 1 if param.choices else 0))
            elif param.param_type == ParameterType.BOOLEAN:
                bounds.append((0, 1))
            elif param.param_type == ParameterType.INTEGER:
                bounds.append((0, param.n_values - 1))
        return bounds

    def get_continuous_bounds(self) -> list[tuple[float, float]]:
        """Get bounds for all continuous parameters."""
        return [
            (float(param.lower), float(param.upper))  # type: ignore[arg-type]
            for param in self.continuous_params.values()
        ]

    def sample_discrete_params(
        self,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """Sample all discrete parameters."""
        if rng is None:
            rng = np.random.default_rng()
        return {name: param.sample(rng) for name, param in self.discrete_params.items()}

    def sample_continuous_params(
        self,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """Sample all continuous parameters."""
        if rng is None:
            rng = np.random.default_rng()
        return {
            name: param.sample(rng) for name, param in self.continuous_params.items()
        }

    def discrete_indices_to_values(
        self,
        indices: NDArray[np.int64],
    ) -> dict[str, Any]:
        """Convert array of discrete indices to parameter values.

        Args:
            indices: Array of shape (n_discrete,) with indices.

        Returns:
            Dictionary mapping parameter names to values.
        """
        values = {}
        for i, (name, param) in enumerate(self.discrete_params.items()):
            if i < len(indices):
                values[name] = param.index_to_value(int(indices[i]))
        return values

    def continuous_array_to_values(
        self,
        values_array: NDArray[np.float64],
    ) -> dict[str, Any]:
        """Convert array of continuous values to parameter dict.

        Args:
            values_array: Array of shape (n_continuous,) with values.

        Returns:
            Dictionary mapping parameter names to values.
        """
        values = {}
        for i, name in enumerate(self.continuous_param_names):
            if i < len(values_array):
                values[name] = float(values_array[i])
        return values

    def instantiate(
        self,
        discrete_values: dict[str, Any] | None = None,
        continuous_values: dict[str, Any] | None = None,
        use_gpu: bool = False,
    ) -> Any:
        """Instantiate the model with given parameters.

        Args:
            discrete_values: Discrete parameter values.
            continuous_values: Continuous parameter values.
            use_gpu: Whether to use GPU model.

        Returns:
            Instantiated model.
        """
        params = dict(self.fixed_params)

        if discrete_values:
            params.update(discrete_values)
        if continuous_values:
            params.update(continuous_values)

        model_class = self.get_model_class(use_gpu)
        return model_class(**params)

    def copy(self) -> ModelDefinition:
        """Create a deep copy of this definition."""
        return ModelDefinition(
            name=self.name,
            model_class=self.model_class,
            cuml_class=self.cuml_class,
            discrete_params={k: v for k, v in self.discrete_params.items()},
            continuous_params={k: v for k, v in self.continuous_params.items()},
            fixed_params=dict(self.fixed_params),
            supports_gpu=self.supports_gpu,
            task_type=self.task_type,
            weight=self.weight,
        )

    def to_model_space(self) -> ModelSpace:
        """Convert ModelDefinition to ModelSpace.

        Returns:
            ModelSpace compatible with model_space module.
        """
        from .parameter_space.model_space import ModelSpace
        from .parameter_space import HyperparameterSpace

        param_space = HyperparameterSpace()

        for name, param in self.discrete_params.items():
            if param.param_type == ParameterType.INTEGER:
                param_space.add_integer(
                    name,
                    int(param.lower),  # type: ignore[arg-type]
                    int(param.upper),  # type: ignore[arg-type]
                    log_scale=param.log_scale,
                )
            elif param.param_type == ParameterType.CATEGORICAL:
                param_space.add_categorical(name, param.choices or [])
            elif param.param_type == ParameterType.BOOLEAN:
                param_space.add_boolean(name)

        for name, param in self.continuous_params.items():
            param_space.add_continuous(
                name,
                float(param.lower),  # type: ignore[arg-type]
                float(param.upper),  # type: ignore[arg-type]
                log_scale=param.log_scale,
            )

        return ModelSpace(
            model_class=self.model_class,
            param_space=param_space,
            fixed_params=self.fixed_params,
            name=self.name,
        )

    def __repr__(self) -> str:
        gpu_str = " (GPU)" if self.supports_gpu else ""
        return (
            f"ModelDefinition({self.name}{gpu_str}, "
            f"discrete={self.n_discrete_params}, "
            f"continuous={self.n_continuous_params})"
        )


class ModelRegistry:
    """Registry for multiple model definitions with GPU awareness.

    Manages a collection of model definitions and provides utilities for
    multi-model optimization including parameter space alignment.

    Attributes:
        models: List of registered model definitions.
        task_type: Task type for all models in registry.
    """

    def __init__(
        self,
        task_type: TaskType = TaskType.CLASSIFICATION,
    ) -> None:
        """Initialize empty model registry.

        Args:
            task_type: Type of ML task for models.
        """
        self.task_type = task_type
        self._models: list[ModelDefinition] = []
        self._name_to_idx: dict[str, int] = {}

        # Cached union spaces
        self._all_discrete_params: list[str] | None = None
        self._all_continuous_params: list[str] | None = None
        self._discrete_mask: NDArray[np.bool_] | None = None
        self._continuous_mask: NDArray[np.bool_] | None = None

    @property
    def models(self) -> list[ModelDefinition]:
        """List of registered models."""
        return self._models

    @property
    def n_models(self) -> int:
        """Number of registered models."""
        return len(self._models)

    @property
    def model_names(self) -> list[str]:
        """List of model names."""
        return [m.name for m in self._models]

    @property
    def max_discrete_params(self) -> int:
        """Maximum number of discrete params across all models."""
        if not self._models:
            return 0
        return max(m.n_discrete_params for m in self._models)

    @property
    def max_continuous_params(self) -> int:
        """Maximum number of continuous params across all models."""
        if not self._models:
            return 0
        return max(m.n_continuous_params for m in self._models)

    @property
    def all_discrete_param_names(self) -> list[str]:
        """Union of all discrete parameter names across models."""
        if self._all_discrete_params is None:
            self._build_param_unions()
        return self._all_discrete_params  # type: ignore[return-value]

    @property
    def all_continuous_param_names(self) -> list[str]:
        """Union of all continuous parameter names across models."""
        if self._all_continuous_params is None:
            self._build_param_unions()
        return self._all_continuous_params  # type: ignore[return-value]

    @property
    def n_all_discrete_params(self) -> int:
        """Total number of unique discrete parameters."""
        return len(self.all_discrete_param_names)

    @property
    def n_all_continuous_params(self) -> int:
        """Total number of unique continuous parameters."""
        return len(self.all_continuous_param_names)

    def _invalidate_cache(self) -> None:
        """Invalidate cached parameter unions."""
        self._all_discrete_params = None
        self._all_continuous_params = None
        self._discrete_mask = None
        self._continuous_mask = None

    def _build_param_unions(self) -> None:
        """Build union of all parameter names across models."""
        discrete_names: list[str] = []
        continuous_names: list[str] = []

        seen_discrete: set[str] = set()
        seen_continuous: set[str] = set()

        for model in self._models:
            for name in model.discrete_param_names:
                if name not in seen_discrete:
                    discrete_names.append(name)
                    seen_discrete.add(name)

            for name in model.continuous_param_names:
                if name not in seen_continuous:
                    continuous_names.append(name)
                    seen_continuous.add(name)

        self._all_discrete_params = discrete_names
        self._all_continuous_params = continuous_names

    def register(self, model: ModelDefinition) -> int:
        """Register a model definition.

        Args:
            model: Model definition to register.

        Returns:
            Index of the registered model.

        Raises:
            ValueError: If model with same name already registered.
        """
        if model.name in self._name_to_idx:
            msg = f"Model '{model.name}' already registered"
            raise ValueError(msg)

        if model.task_type != self.task_type:
            logger.warning(
                "Model '%s' task type %s differs from registry %s",
                model.name,
                model.task_type,
                self.task_type,
            )

        idx = len(self._models)
        self._models.append(model)
        self._name_to_idx[model.name] = idx
        self._invalidate_cache()

        logger.debug("Registered model '%s' at index %d", model.name, idx)
        return idx

    def get_model(self, identifier: int | str) -> ModelDefinition:
        """Get model by index or name.

        Args:
            identifier: Model index or name.

        Returns:
            Model definition.

        Raises:
            KeyError: If model not found.
        """
        if isinstance(identifier, int):
            if 0 <= identifier < len(self._models):
                return self._models[identifier]
            msg = f"Model index {identifier} out of range"
            raise KeyError(msg)

        if identifier in self._name_to_idx:
            return self._models[self._name_to_idx[identifier]]
        msg = f"Model '{identifier}' not found"
        raise KeyError(msg)

    def get_model_index(self, name: str) -> int:
        """Get model index by name.

        Args:
            name: Model name.

        Returns:
            Model index.

        Raises:
            KeyError: If model not found.
        """
        if name not in self._name_to_idx:
            msg = f"Model '{name}' not found"
            raise KeyError(msg)
        return self._name_to_idx[name]

    def get_discrete_mask(self) -> NDArray[np.bool_]:
        """Get mask indicating which discrete params apply to each model.

        Returns:
            Boolean array of shape (n_models, n_all_discrete_params).
            mask[i, j] is True if model i has parameter all_discrete_params[j].
        """
        if self._discrete_mask is not None:
            return self._discrete_mask

        all_names = self.all_discrete_param_names
        mask = np.zeros((self.n_models, len(all_names)), dtype=np.bool_)

        for i, model in enumerate(self._models):
            model_params = set(model.discrete_param_names)
            for j, name in enumerate(all_names):
                mask[i, j] = name in model_params

        self._discrete_mask = mask
        return mask

    def get_continuous_mask(self) -> NDArray[np.bool_]:
        """Get mask indicating which continuous params apply to each model.

        Returns:
            Boolean array of shape (n_models, n_all_continuous_params).
        """
        if self._continuous_mask is not None:
            return self._continuous_mask

        all_names = self.all_continuous_param_names
        mask = np.zeros((self.n_models, len(all_names)), dtype=np.bool_)

        for i, model in enumerate(self._models):
            model_params = set(model.continuous_param_names)
            for j, name in enumerate(all_names):
                mask[i, j] = name in model_params

        self._continuous_mask = mask
        return mask

    def get_discrete_bounds(self) -> NDArray[np.int64]:
        """Get bounds for all discrete parameters in union space.

        Returns:
            Array of shape (n_all_discrete_params, 2) with (min, max) bounds.
            Uses maximum range across all models that have each parameter.
        """
        all_names = self.all_discrete_param_names
        bounds = np.zeros((len(all_names), 2), dtype=np.int64)

        for j, name in enumerate(all_names):
            max_upper = 0
            for model in self._models:
                if name in model.discrete_params:
                    param = model.discrete_params[name]
                    max_upper = max(max_upper, param.n_values - 1)
            bounds[j, 0] = 0
            bounds[j, 1] = max_upper

        return bounds

    def get_continuous_bounds(self) -> NDArray[np.float64]:
        """Get bounds for all continuous parameters in union space.

        Returns:
            Array of shape (n_all_continuous_params, 2) with (lower, upper) bounds.
            Uses union of ranges across all models that have each parameter.
        """
        all_names = self.all_continuous_param_names
        bounds = np.full((len(all_names), 2), [np.inf, -np.inf], dtype=np.float64)

        for j, name in enumerate(all_names):
            for model in self._models:
                if name in model.continuous_params:
                    param = model.continuous_params[name]
                    bounds[j, 0] = min(bounds[j, 0], float(param.lower))  # type: ignore[arg-type]
                    bounds[j, 1] = max(bounds[j, 1], float(param.upper))  # type: ignore[arg-type]

            # Handle case where no model has this parameter
            if np.isinf(bounds[j, 0]):
                bounds[j] = [0.0, 1.0]

        return bounds

    def get_sampling_weights(self) -> NDArray[np.float64]:
        """Get normalized sampling weights for models.

        Returns:
            Array of shape (n_models,) with normalized weights.
        """
        weights = np.array([m.weight for m in self._models], dtype=np.float64)
        return weights / weights.sum()

    def sample_model_index(
        self,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Sample a model index according to weights.

        Args:
            rng: Random number generator.

        Returns:
            Sampled model index.
        """
        if rng is None:
            rng = np.random.default_rng()
        weights = self.get_sampling_weights()
        return int(rng.choice(self.n_models, p=weights))

    def sample_model_indices(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.int64]:
        """Sample n model indices according to weights.

        Args:
            n: Number of indices to sample.
            rng: Random number generator.

        Returns:
            Array of shape (n,) with sampled model indices.
        """
        if rng is None:
            rng = np.random.default_rng()
        weights = self.get_sampling_weights()
        return rng.choice(self.n_models, size=n, p=weights).astype(np.int64)

    def iter_models(self) -> Iterator[tuple[int, ModelDefinition]]:
        """Iterate over (index, model) pairs."""
        yield from enumerate(self._models)

    def __len__(self) -> int:
        return len(self._models)

    def __iter__(self) -> Iterator[ModelDefinition]:
        return iter(self._models)

    def __getitem__(self, key: int | str) -> ModelDefinition:
        return self.get_model(key)

    def __contains__(self, key: str) -> bool:
        return key in self._name_to_idx

    def __repr__(self) -> str:
        return f"ModelRegistry(n_models={self.n_models}, task={self.task_type.name})"




@dataclass
class MultiModelChromosome:
    """Single chromosome with multi-model support.

    Represents a complete configuration including model selection,
    feature selection, and parameter values.

    Attributes:
        model_index: Index of selected model.
        feature_mask: Boolean mask for feature selection.
        discrete_params: Array of discrete parameter indices.
        continuous_params: Array of continuous parameter values.
        fitness: Fitness value (None if not evaluated).
    """

    model_index: int
    feature_mask: NDArray[np.bool_]
    discrete_params: NDArray[np.int64]
    continuous_params: NDArray[np.float64]
    fitness: float | None = None

    @classmethod
    def from_random(
        cls,
        registry: ModelRegistry,
        n_features: int,
        feature_prob: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> MultiModelChromosome:
        """Create a random chromosome.

        Args:
            registry: Model registry.
            n_features: Number of features.
            feature_prob: Probability of selecting each feature.
            rng: Random number generator.

        Returns:
            Randomly initialized chromosome.
        """
        if rng is None:
            rng = np.random.default_rng()

        model_index = registry.sample_model_index(rng)
        model = registry.get_model(model_index)

        feature_mask = rng.random(n_features) < feature_prob
        if not feature_mask.any():
            feature_mask[rng.integers(n_features)] = True

        discrete_bounds = registry.get_discrete_bounds()
        discrete_params = np.array(
            [rng.integers(b[0], b[1] + 1) for b in discrete_bounds],
            dtype=np.int64,
        )

        continuous_bounds = registry.get_continuous_bounds()
        continuous_params = np.array(
            [rng.uniform(b[0], b[1]) for b in continuous_bounds],
            dtype=np.float64,
        )

        return cls(
            model_index=model_index,
            feature_mask=feature_mask,
            discrete_params=discrete_params,
            continuous_params=continuous_params,
        )

    def get_active_discrete_values(
        self,
        registry: ModelRegistry,
    ) -> dict[str, Any]:
        """Get discrete parameter values for the selected model.

        Args:
            registry: Model registry.

        Returns:
            Dictionary of parameter name to value for active parameters.
        """
        model = registry.get_model(self.model_index)
        all_names = registry.all_discrete_param_names
        mask = registry.get_discrete_mask()[self.model_index]

        values = {}
        for i, (name, is_active) in enumerate(zip(all_names, mask, strict=True)):
            if is_active and name in model.discrete_params:
                param = model.discrete_params[name]
                values[name] = param.index_to_value(int(self.discrete_params[i]))

        return values

    def get_active_continuous_values(
        self,
        registry: ModelRegistry,
    ) -> dict[str, Any]:
        """Get continuous parameter values for the selected model.

        Args:
            registry: Model registry.

        Returns:
            Dictionary of parameter name to value for active parameters.
        """
        model = registry.get_model(self.model_index)
        all_names = registry.all_continuous_param_names
        mask = registry.get_continuous_mask()[self.model_index]

        values = {}
        for i, (name, is_active) in enumerate(zip(all_names, mask, strict=True)):
            if is_active and name in model.continuous_params:
                values[name] = float(self.continuous_params[i])

        return values

    def copy(self) -> MultiModelChromosome:
        """Create a deep copy of this chromosome."""
        return MultiModelChromosome(
            model_index=self.model_index,
            feature_mask=self.feature_mask.copy(),
            discrete_params=self.discrete_params.copy(),
            continuous_params=self.continuous_params.copy(),
            fitness=self.fitness,
        )


class MultiModelPopulation:
    """GPU-accelerated population for multi-model optimization.

    Stores all population data as contiguous arrays for efficient GPU operations.
    Supports model selection as part of the evolutionary process.

    Attributes:
        pop_size: Population size.
        n_features: Number of features.
        registry: Model registry.
        use_gpu: Whether to use GPU acceleration.
    """

    def __init__(
        self,
        pop_size: int,
        n_features: int,
        registry: ModelRegistry,
        use_gpu: bool = False,
        feature_prob: float = 0.5,
        random_state: int | None = None,
    ) -> None:
        """Initialize multi-model population.

        Args:
            pop_size: Number of individuals.
            n_features: Number of features.
            registry: Model registry with registered models.
            use_gpu: Whether to use GPU acceleration.
            feature_prob: Initial probability of selecting each feature.
            random_state: Random seed for reproducibility.
        """
        if pop_size <= 0:
            msg = f"pop_size must be positive, got {pop_size}"
            raise ValueError(msg)
        if n_features <= 0:
            msg = f"n_features must be positive, got {n_features}"
            raise ValueError(msg)
        if registry.n_models == 0:
            msg = "Registry must have at least one model"
            raise ValueError(msg)

        self.pop_size = pop_size
        self.n_features = n_features
        self.registry = registry
        self.use_gpu = use_gpu
        self.feature_prob = feature_prob

        self._rng = np.random.default_rng(random_state)

        self._backend = self._get_backend()

        n_discrete = registry.n_all_discrete_params
        n_continuous = registry.n_all_continuous_params

        self._model_indices = self._init_model_indices()

        self._feature_genes = self._init_feature_genes()

        self._discrete_genes = self._init_discrete_genes()

        self._continuous_genes = self._init_continuous_genes()

        xp = self._backend
        self._fitness = xp.full(pop_size, -xp.inf, dtype=xp.float64)

        self._discrete_mask = xp.asarray(registry.get_discrete_mask())
        self._continuous_mask = xp.asarray(registry.get_continuous_mask())

        self._discrete_bounds = xp.asarray(registry.get_discrete_bounds())
        self._continuous_bounds = xp.asarray(registry.get_continuous_bounds())

        self.generation = 0

        self.id = str(uuid.uuid4())[:8]

        logger.debug(
            "Created MultiModelPopulation %s: pop=%d, features=%d, models=%d, gpu=%s",
            self.id,
            pop_size,
            n_features,
            registry.n_models,
            use_gpu,
        )

    def _get_backend(self) -> Any:
        """Get array backend (NumPy or CuPy)."""
        if self.use_gpu:
            try:
                import cupy as cp

                return cp
            except ImportError:
                logger.warning("CuPy not available, falling back to NumPy")
                self.use_gpu = False
        return np

    def _init_model_indices(self) -> NDArray[np.int64]:
        """Initialize model indices according to weights."""
        indices = self.registry.sample_model_indices(self.pop_size, self._rng)
        xp = self._backend
        return xp.asarray(indices, dtype=xp.int64)

    def _init_feature_genes(self) -> NDArray[np.bool_]:
        """Initialize feature selection genes."""
        xp = self._backend
        genes = self._rng.random((self.pop_size, self.n_features)) < self.feature_prob

        no_features = ~genes.any(axis=1)
        for i in np.where(no_features)[0]:
            genes[i, self._rng.integers(self.n_features)] = True

        return xp.asarray(genes, dtype=xp.bool_)

    def _init_discrete_genes(self) -> NDArray[np.int64]:
        """Initialize discrete parameter genes."""
        xp = self._backend
        bounds = self.registry.get_discrete_bounds()
        n_discrete = len(bounds)

        genes = np.zeros((self.pop_size, n_discrete), dtype=np.int64)
        for j, (low, high) in enumerate(bounds):
            genes[:, j] = self._rng.integers(low, high + 1, size=self.pop_size)

        return xp.asarray(genes, dtype=xp.int64)

    def _init_continuous_genes(self) -> NDArray[np.float64]:
        """Initialize continuous parameter genes."""
        xp = self._backend
        bounds = self.registry.get_continuous_bounds()
        n_continuous = len(bounds)

        genes = np.zeros((self.pop_size, n_continuous), dtype=np.float64)
        for j, (low, high) in enumerate(bounds):
            genes[:, j] = self._rng.uniform(low, high, size=self.pop_size)

        return xp.asarray(genes, dtype=xp.float64)


    @property
    def model_indices(self) -> NDArray[np.int64]:
        """Model indices for all individuals."""
        return self._model_indices

    @property
    def feature_genes(self) -> NDArray[np.bool_]:
        """Feature selection genes for all individuals."""
        return self._feature_genes

    @property
    def discrete_genes(self) -> NDArray[np.int64]:
        """Discrete parameter genes for all individuals."""
        return self._discrete_genes

    @property
    def continuous_genes(self) -> NDArray[np.float64]:
        """Continuous parameter genes for all individuals."""
        return self._continuous_genes

    @property
    def fitness(self) -> NDArray[np.float64]:
        """Fitness values for all individuals."""
        return self._fitness

    @fitness.setter
    def fitness(self, values: NDArray[np.float64]) -> None:
        """Set fitness values."""
        xp = self._backend
        if len(values) != self.pop_size:
            msg = f"Expected {self.pop_size} fitness values, got {len(values)}"
            raise ValueError(msg)
        self._fitness = xp.asarray(values, dtype=xp.float64)

    @property
    def best_idx(self) -> int:
        """Index of best individual."""
        xp = self._backend
        if self.use_gpu:
            return int(xp.argmax(self._fitness).get())
        return int(np.argmax(self._fitness))

    @property
    def best_fitness(self) -> float:
        """Fitness of best individual."""
        xp = self._backend
        if self.use_gpu:
            return float(self._fitness[self.best_idx].get())
        return float(self._fitness[self.best_idx])


    def to_numpy(self) -> tuple[
        NDArray[np.int64],
        NDArray[np.bool_],
        NDArray[np.int64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Convert all arrays to NumPy.

        Returns:
            Tuple of (model_indices, feature_genes, discrete_genes,
                     continuous_genes, fitness).
        """
        if self.use_gpu:
            return (
                self._model_indices.get(), # pyright: ignore[reportAttributeAccessIssue]
                self._feature_genes.get(),  # pyright: ignore[reportAttributeAccessIssue]
                self._discrete_genes.get(), # pyright: ignore[reportAttributeAccessIssue]
                self._continuous_genes.get(), # pyright: ignore[reportAttributeAccessIssue]
                self._fitness.get(),
            )
        return (
            self._model_indices,
            self._feature_genes,
            self._discrete_genes,
            self._continuous_genes,
            self._fitness,
        )

    def get_individual(self, idx: int) -> MultiModelChromosome:
        """Get individual as MultiModelChromosome.

        Args:
            idx: Individual index.

        Returns:
            Chromosome object.
        """
        if self.use_gpu:
            return MultiModelChromosome(
                model_index=int(self._model_indices[idx].get()),
                feature_mask=self._feature_genes[idx].get(),
                discrete_params=self._discrete_genes[idx].get(),
                continuous_params=self._continuous_genes[idx].get(),
                fitness=float(self._fitness[idx].get()),
            )
        return MultiModelChromosome(
            model_index=int(self._model_indices[idx]),
            feature_mask=self._feature_genes[idx].copy(),
            discrete_params=self._discrete_genes[idx].copy(),
            continuous_params=self._continuous_genes[idx].copy(),
            fitness=float(self._fitness[idx]),
        )

    def get_best_individual(self) -> MultiModelChromosome:
        """Get the best individual."""
        return self.get_individual(self.best_idx)

    def set_individual(self, idx: int, chromosome: MultiModelChromosome) -> None:
        """Set individual from chromosome.

        Args:
            idx: Individual index.
            chromosome: Chromosome to set.
        """
        xp = self._backend
        self._model_indices[idx] = chromosome.model_index
        self._feature_genes[idx] = xp.asarray(chromosome.feature_mask)
        self._discrete_genes[idx] = xp.asarray(chromosome.discrete_params)
        self._continuous_genes[idx] = xp.asarray(chromosome.continuous_params)
        if chromosome.fitness is not None:
            self._fitness[idx] = chromosome.fitness



    def get_active_discrete_mask(self) -> NDArray[np.bool_]:
        """Get mask of active discrete parameters for each individual.

        Returns:
            Boolean array of shape (pop_size, n_discrete) where
            mask[i, j] indicates if individual i's model uses parameter j.
        """
        xp = self._backend
        return self._discrete_mask[self._model_indices]

    def get_active_continuous_mask(self) -> NDArray[np.bool_]:
        """Get mask of active continuous parameters for each individual.

        Returns:
            Boolean array of shape (pop_size, n_continuous).
        """
        xp = self._backend
        return self._continuous_mask[self._model_indices]

    def get_masked_discrete_genes(self) -> NDArray[np.int64]:
        """Get discrete genes with inactive parameters zeroed.

        Returns:
            Array of shape (pop_size, n_discrete) with inactive params set to 0.
        """
        xp = self._backend
        mask = self.get_active_discrete_mask()
        return self._discrete_genes * mask.astype(xp.int64)

    def get_masked_continuous_genes(self) -> NDArray[np.float64]:
        """Get continuous genes with inactive parameters zeroed.

        Returns:
            Array of shape (pop_size, n_continuous) with inactive params set to 0.
        """
        xp = self._backend
        mask = self.get_active_continuous_mask()
        return self._continuous_genes * mask.astype(xp.float64)


    def mutate_model_indices(
        self,
        mutation_rate: float = 0.1,
        mask: NDArray[np.bool_] | None = None,
    ) -> None:
        """Mutate model indices.

        Args:
            mutation_rate: Probability of mutating each individual.
            mask: Optional mask of individuals to consider for mutation.
        """
        xp = self._backend

        mutate_mask = xp.asarray(
            self._rng.random(self.pop_size) < mutation_rate,
            dtype=xp.bool_,
        )

        if mask is not None:
            mutate_mask = mutate_mask & xp.asarray(mask)

        n_mutate = int(xp.sum(mutate_mask))
        if n_mutate > 0:
            if self.use_gpu:
                mutate_indices = xp.where(mutate_mask)[0].get()
            else:
                mutate_indices = np.where(mutate_mask)[0]

            new_indices = self.registry.sample_model_indices(n_mutate, self._rng)
            self._model_indices[mutate_mask] = xp.asarray(new_indices)

    def mutate_features(
        self,
        mutation_rate: float = 0.05,
        mask: NDArray[np.bool_] | None = None,
    ) -> None:
        """Mutate feature selection genes.

        Args:
            mutation_rate: Probability of flipping each feature bit.
            mask: Optional mask of individuals to consider for mutation.
        """
        xp = self._backend

        flip_mask = xp.asarray(
            self._rng.random((self.pop_size, self.n_features)) < mutation_rate,
            dtype=xp.bool_,
        )

        if mask is not None:
            flip_mask = flip_mask & xp.asarray(mask)[:, xp.newaxis]

        self._feature_genes = self._feature_genes ^ flip_mask

        no_features = ~self._feature_genes.any(axis=1)
        if self.use_gpu:
            no_feature_idx = xp.where(no_features)[0].get()
        else:
            no_feature_idx = np.where(no_features)[0]

        for i in no_feature_idx:
            self._feature_genes[i, self._rng.integers(self.n_features)] = True

    def mutate_discrete_params(
        self,
        mutation_rate: float = 0.1,
        mask: NDArray[np.bool_] | None = None,
    ) -> None:
        """Mutate discrete parameter genes respecting bounds.

        Args:
            mutation_rate: Probability of mutating each parameter.
            mask: Optional mask of individuals to consider for mutation.
        """
        xp = self._backend
        n_discrete = self.registry.n_all_discrete_params

        mutate_mask = xp.asarray(
            self._rng.random((self.pop_size, n_discrete)) < mutation_rate,
            dtype=xp.bool_,
        )

        if mask is not None:
            mutate_mask = mutate_mask & xp.asarray(mask)[:, xp.newaxis]

        active_mask = self.get_active_discrete_mask()
        mutate_mask = mutate_mask & active_mask

        bounds_np = self.registry.get_discrete_bounds()
        for j in range(n_discrete):
            low, high = bounds_np[j]
            col_mask = mutate_mask[:, j]
            if self.use_gpu:
                n_mutate = int(xp.sum(col_mask).get())
            else:
                n_mutate = int(np.sum(col_mask))

            if n_mutate > 0:
                new_values = self._rng.integers(low, high + 1, size=n_mutate)
                self._discrete_genes[col_mask, j] = xp.asarray(new_values)

    def mutate_continuous_params(
        self,
        mutation_strength: float = 0.1,
        mutation_rate: float = 0.2,
        mask: NDArray[np.bool_] | None = None,
    ) -> None:
        """Mutate continuous parameter genes with Gaussian perturbation.

        Args:
            mutation_strength: Standard deviation as fraction of range.
            mutation_rate: Probability of mutating each parameter.
            mask: Optional mask of individuals to consider for mutation.
        """
        xp = self._backend
        n_continuous = self.registry.n_all_continuous_params

        mutate_mask = xp.asarray(
            self._rng.random((self.pop_size, n_continuous)) < mutation_rate,
            dtype=xp.bool_,
        )

        if mask is not None:
            mutate_mask = mutate_mask & xp.asarray(mask)[:, xp.newaxis]

        active_mask = self.get_active_continuous_mask()
        mutate_mask = mutate_mask & active_mask

        bounds_np = self.registry.get_continuous_bounds()
        ranges = bounds_np[:, 1] - bounds_np[:, 0]

        noise = self._rng.normal(0, mutation_strength, (self.pop_size, n_continuous))
        noise = noise * ranges  # Scale by parameter ranges

        noise_gpu = xp.asarray(noise, dtype=xp.float64)
        self._continuous_genes = xp.where(
            mutate_mask,
            self._continuous_genes + noise_gpu,
            self._continuous_genes,
        )

        lower = xp.asarray(bounds_np[:, 0], dtype=xp.float64)
        upper = xp.asarray(bounds_np[:, 1], dtype=xp.float64)
        self._continuous_genes = xp.clip(self._continuous_genes, lower, upper)



    def crossover_features(
        self,
        parent1_idx: NDArray[np.intp],
        parent2_idx: NDArray[np.intp],
        crossover_rate: float = 0.8,
    ) -> NDArray[np.bool_]:
        """Perform uniform crossover on feature genes.

        Args:
            parent1_idx: Indices of first parents.
            parent2_idx: Indices of second parents.
            crossover_rate: Probability of performing crossover.

        Returns:
            Offspring feature genes of shape (n_offspring, n_features).
        """
        xp = self._backend
        n_offspring = len(parent1_idx)

        p1_features = self._feature_genes[parent1_idx]
        p2_features = self._feature_genes[parent2_idx]

        do_crossover = xp.asarray(
            self._rng.random(n_offspring) < crossover_rate,
            dtype=xp.bool_,
        )

        swap_mask = xp.asarray(
            self._rng.random((n_offspring, self.n_features)) < 0.5,
            dtype=xp.bool_,
        )

        offspring = xp.where(
            do_crossover[:, xp.newaxis] & swap_mask,
            p2_features,
            p1_features,
        )

        return offspring

    def crossover_discrete(
        self,
        parent1_idx: NDArray[np.intp],
        parent2_idx: NDArray[np.intp],
        crossover_rate: float = 0.8,
    ) -> NDArray[np.int64]:
        """Perform uniform crossover on discrete parameter genes.

        Args:
            parent1_idx: Indices of first parents.
            parent2_idx: Indices of second parents.
            crossover_rate: Probability of performing crossover.

        Returns:
            Offspring discrete genes.
        """
        xp = self._backend
        n_offspring = len(parent1_idx)
        n_discrete = self.registry.n_all_discrete_params

        p1_discrete = self._discrete_genes[parent1_idx]
        p2_discrete = self._discrete_genes[parent2_idx]

        do_crossover = xp.asarray(
            self._rng.random(n_offspring) < crossover_rate,
            dtype=xp.bool_,
        )

        swap_mask = xp.asarray(
            self._rng.random((n_offspring, n_discrete)) < 0.5,
            dtype=xp.bool_,
        )

        offspring = xp.where(
            do_crossover[:, xp.newaxis] & swap_mask,
            p2_discrete,
            p1_discrete,
        )

        return offspring

    def crossover_continuous(
        self,
        parent1_idx: NDArray[np.intp],
        parent2_idx: NDArray[np.intp],
        crossover_rate: float = 0.8,
        blend_alpha: float = 0.5,
    ) -> NDArray[np.float64]:
        """Perform BLX-alpha crossover on continuous parameter genes.

        Args:
            parent1_idx: Indices of first parents.
            parent2_idx: Indices of second parents.
            crossover_rate: Probability of performing crossover.
            blend_alpha: Blending parameter (0.5 = uniform blend).

        Returns:
            Offspring continuous genes.
        """
        xp = self._backend
        n_offspring = len(parent1_idx)
        n_continuous = self.registry.n_all_continuous_params

        p1_cont = self._continuous_genes[parent1_idx]
        p2_cont = self._continuous_genes[parent2_idx]

        do_crossover = xp.asarray(
            self._rng.random(n_offspring) < crossover_rate,
            dtype=xp.bool_,
        )

        # BLX-alpha: offspring = p1 + gamma * (p2 - p1)
        # where gamma is uniform in [-alpha, 1 + alpha]
        gamma = self._rng.uniform(
            -blend_alpha, 1 + blend_alpha, (n_offspring, n_continuous)
        )
        gamma = xp.asarray(gamma, dtype=xp.float64)

        blended = p1_cont + gamma * (p2_cont - p1_cont)

        # Apply crossover where appropriate
        offspring = xp.where(do_crossover[:, xp.newaxis], blended, p1_cont)

        # Clip to bounds
        bounds_np = self.registry.get_continuous_bounds()
        lower = xp.asarray(bounds_np[:, 0], dtype=xp.float64)
        upper = xp.asarray(bounds_np[:, 1], dtype=xp.float64)
        offspring = xp.clip(offspring, lower, upper)

        return offspring


    def get_model_distribution(self) -> dict[str, int]:
        """Get distribution of models in population.

        Returns:
            Dictionary mapping model name to count.
        """
        if self.use_gpu:
            indices = self._model_indices.get()  # pyright: ignore[reportAttributeAccessIssue]
        else:
            indices = self._model_indices

        distribution = {}
        for model in self.registry:
            idx = self.registry.get_model_index(model.name)
            distribution[model.name] = int(np.sum(indices == idx))

        return distribution

    def get_feature_selection_stats(self) -> dict[str, float]:
        """Get feature selection statistics.

        Returns:
            Dictionary with min, max, mean features selected.
        """
        xp = self._backend
        n_selected = xp.sum(self._feature_genes, axis=1)

        if self.use_gpu:
            n_selected = n_selected.get()

        return {
            "min_features": int(np.min(n_selected)),
            "max_features": int(np.max(n_selected)),
            "mean_features": float(np.mean(n_selected)),
            "std_features": float(np.std(n_selected)),
        }

    def get_fitness_stats(self) -> dict[str, float]:
        """Get fitness statistics.

        Returns:
            Dictionary with fitness statistics.
        """
        if self.use_gpu:
            fitness = self._fitness.get()
        else:
            fitness = self._fitness

        # Exclude -inf values
        valid_fitness = fitness[np.isfinite(fitness)]
        if len(valid_fitness) == 0:
            return {
                "best": float("-inf"),
                "worst": float("-inf"),
                "mean": float("-inf"),
                "std": 0.0,
                "n_evaluated": 0,
            }

        return {
            "best": float(np.max(valid_fitness)),
            "worst": float(np.min(valid_fitness)),
            "mean": float(np.mean(valid_fitness)),
            "std": float(np.std(valid_fitness)),
            "n_evaluated": len(valid_fitness),
        }

    def summary(self) -> str:
        """Get population summary string."""
        model_dist = self.get_model_distribution()
        feature_stats = self.get_feature_selection_stats()
        fitness_stats = self.get_fitness_stats()

        lines = [
            f"MultiModelPopulation {self.id}:",
            f"  Generation: {self.generation}",
            f"  Size: {self.pop_size}",
            f"  Models: {model_dist}",
            f"  Features: {feature_stats['mean_features']:.1f} Â± {feature_stats['std_features']:.1f}",
            f"  Best fitness: {fitness_stats['best']:.4f}",
            f"  Mean fitness: {fitness_stats['mean']:.4f}",
            f"  Evaluated: {fitness_stats['n_evaluated']}/{self.pop_size}",
        ]

        return "\n".join(lines)

    def __repr__(self) -> str:
        gpu_str = " (GPU)" if self.use_gpu else ""
        return (
            f"MultiModelPopulation({self.pop_size} individuals, "
            f"{self.n_features} features, "
            f"{self.registry.n_models} models{gpu_str})"
        )

    def to_gpu_population(self, model_index: int | None = None) -> GPUPopulation:
        """Convert to GPUPopulation format for single-model optimization.

        This allows using HybridGAPSOOptimizer with a specific model from
        the multi-model population.

        Args:
            model_index: Index of model to extract population for.
                        If None, uses the most common model.

        Returns:
            GPUPopulation with individuals for the specified model.
        """
        if model_index is None:
            # Use the most common model
            dist = self.get_model_distribution()
            model_name = max(dist, key=dist.get)  # type: ignore[arg-type]
            model_index = self.registry.get_model_index(model_name)

        if self.use_gpu:
            model_indices_np = to_numpy(self._model_indices)
        else:
            model_indices_np = self._model_indices

        mask = model_indices_np == model_index
        indices = np.where(mask)[0]

        if len(indices) == 0:
            msg = f"No individuals found for model index {model_index}"
            raise ValueError(msg)

        if self.use_gpu:
            feature_genes = to_numpy(self._feature_genes[indices])
            discrete_genes = to_numpy(self._discrete_genes[indices])
            fitness = to_numpy(self._fitness[indices])
        else:
            feature_genes = self._feature_genes[indices].copy()
            discrete_genes = self._discrete_genes[indices].copy()
            fitness = self._fitness[indices].copy()

        needs_eval = ~np.isfinite(fitness)
        generation_arr = np.full(len(indices), self.generation, dtype=np.int64)

        return GPUPopulation(
            feature_genes=feature_genes,
            discrete_genes=discrete_genes,
            fitness=fitness,
            needs_evaluation=needs_eval,
            generation=generation_arr,
            device_info=DeviceInfo.gpu() if self.use_gpu else DeviceInfo.cpu(),
        )

    @classmethod
    def from_gpu_population(
        cls,
        population: GPUPopulation,
        registry: ModelRegistry,
        model_index: int,
        use_gpu: bool = False,
    ) -> MultiModelPopulation:
        """Create MultiModelPopulation from GPUPopulation.

        Args:
            population: Source GPUPopulation.
            registry: Model registry.
            model_index: Model index to assign to all individuals.
            use_gpu: Whether to use GPU.

        Returns:
            MultiModelPopulation initialized from GPUPopulation.
        """
        pop_size = population.pop_size
        n_features = population.n_features

        instance = cls(
            pop_size=pop_size,
            n_features=n_features,
            registry=registry,
            use_gpu=use_gpu,
        )

        feature_genes = to_numpy(population.feature_genes)
        discrete_genes = to_numpy(population.discrete_genes)
        fitness = to_numpy(population.fitness)

        xp = instance._backend
        instance._feature_genes = xp.asarray(feature_genes)
        instance._discrete_genes = xp.asarray(discrete_genes)
        instance._fitness = xp.asarray(fitness)
        instance._model_indices = xp.full(pop_size, model_index, dtype=xp.int64)
        instance.generation = int(np.max(to_numpy(population.generation)))

        return instance


def create_param(
    name: str,
    lower: float | int | None = None,
    upper: float | int | None = None,
    choices: list[Any] | None = None,
    log_scale: bool = False,
    param_type: ParameterType | None = None,
) -> ParameterDefinition:
    """Create a parameter definition with type inference.

    Args:
        name: Parameter name.
        lower: Lower bound (for numeric types).
        upper: Upper bound (for numeric types).
        choices: Valid choices (for CATEGORICAL).
        log_scale: Whether to use log scale.
        param_type: Explicit parameter type (inferred if None).

    Returns:
        ParameterDefinition instance.
    """
    if param_type is not None:
        return ParameterDefinition(
            name=name,
            param_type=param_type,
            lower=lower,
            upper=upper,
            choices=choices,
            log_scale=log_scale,
        )

    if choices is not None:
        return ParameterDefinition(
            name=name,
            param_type=ParameterType.CATEGORICAL,
            choices=choices,
        )

    if lower is not None and upper is not None:
        if isinstance(lower, int) and isinstance(upper, int) and not log_scale:
            return ParameterDefinition(
                name=name,
                param_type=ParameterType.INTEGER,
                lower=lower,
                upper=upper,
            )
        return ParameterDefinition(
            name=name,
            param_type=ParameterType.CONTINUOUS,
            lower=float(lower),
            upper=float(upper),
            log_scale=log_scale,
        )

    msg = f"Cannot infer parameter type for '{name}'"
    raise ValueError(msg)


def create_sklearn_registry(
    task: str = "classification",
    include_models: list[str] | None = None,
) -> ModelRegistry:
    """Create a registry with common sklearn models.

    Args:
        task: "classification" or "regression".
        include_models: Optional list of model names to include.
            If None, includes all available models for the task.

    Returns:
        ModelRegistry with sklearn models.
    """
    from sklearn.ensemble import (
        AdaBoostClassifier,
        AdaBoostRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import (
        ElasticNet,
        Lasso,
        LogisticRegression,
        Ridge,
    )
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    task_type = TaskType.CLASSIFICATION if task == "classification" else TaskType.REGRESSION
    registry = ModelRegistry(task_type=task_type)

    if task == "classification":
        models = _get_classification_models(
            RandomForestClassifier,
            GradientBoostingClassifier,
            AdaBoostClassifier,
            LogisticRegression,
            SVC,
            KNeighborsClassifier,
            DecisionTreeClassifier,
        )
    else:
        models = _get_regression_models(
            RandomForestRegressor,
            GradientBoostingRegressor,
            AdaBoostRegressor,
            Ridge,
            Lasso,
            ElasticNet,
            SVR,
            KNeighborsRegressor,
            DecisionTreeRegressor,
        )

    for model in models:
        if include_models is None or model.name in include_models:
            registry.register(model)

    return registry


def _get_classification_models(*classes: type) -> list[ModelDefinition]:
    """Get classification model definitions."""
    models = []

    class_map = {cls.__name__: cls for cls in classes}

    if "RandomForestClassifier" in class_map:
        models.append(
            ModelDefinition(
                name="RandomForest",
                model_class=class_map["RandomForestClassifier"],
                discrete_params={
                    "n_estimators": create_param("n_estimators", 10, 500),
                    "max_depth": create_param("max_depth", 2, 50),
                    "min_samples_split": create_param("min_samples_split", 2, 20),
                    "criterion": create_param("criterion", choices=["gini", "entropy"]),
                },
                continuous_params={
                    "max_features": create_param("max_features", 0.1, 1.0),
                },
                fixed_params={"random_state": 42, "n_jobs": -1},
                task_type=TaskType.CLASSIFICATION,
            )
        )

    if "GradientBoostingClassifier" in class_map:
        models.append(
            ModelDefinition(
                name="GradientBoosting",
                model_class=class_map["GradientBoostingClassifier"],
                discrete_params={
                    "n_estimators": create_param("n_estimators", 50, 500),
                    "max_depth": create_param("max_depth", 2, 20),
                },
                continuous_params={
                    "learning_rate": create_param(
                        "learning_rate", 0.01, 0.3, log_scale=True
                    ),
                    "subsample": create_param("subsample", 0.5, 1.0),
                },
                fixed_params={"random_state": 42},
                task_type=TaskType.CLASSIFICATION,
            )
        )

    if "AdaBoostClassifier" in class_map:
        models.append(
            ModelDefinition(
                name="AdaBoost",
                model_class=class_map["AdaBoostClassifier"],
                discrete_params={
                    "n_estimators": create_param("n_estimators", 50, 500),
                },
                continuous_params={
                    "learning_rate": create_param(
                        "learning_rate", 0.01, 2.0, log_scale=True
                    ),
                },
                fixed_params={"random_state": 42},
                task_type=TaskType.CLASSIFICATION,
            )
        )

    if "LogisticRegression" in class_map:
        models.append(
            ModelDefinition(
                name="LogisticRegression",
                model_class=class_map["LogisticRegression"],
                discrete_params={
                    "solver": create_param(
                        "solver", choices=["lbfgs", "liblinear", "saga"]
                    ),
                },
                continuous_params={
                    "C": create_param("C", 0.001, 100.0, log_scale=True),
                },
                fixed_params={"random_state": 42, "max_iter": 1000},
                task_type=TaskType.CLASSIFICATION,
            )
        )

    if "SVC" in class_map:
        models.append(
            ModelDefinition(
                name="SVC",
                model_class=class_map["SVC"],
                discrete_params={
                    "kernel": create_param("kernel", choices=["rbf", "linear", "poly"]),
                },
                continuous_params={
                    "C": create_param("C", 0.01, 100.0, log_scale=True),
                    "gamma": create_param("gamma", 0.0001, 1.0, log_scale=True),
                },
                fixed_params={"random_state": 42, "probability": True},
                task_type=TaskType.CLASSIFICATION,
            )
        )

    if "KNeighborsClassifier" in class_map:
        models.append(
            ModelDefinition(
                name="KNN",
                model_class=class_map["KNeighborsClassifier"],
                discrete_params={
                    "n_neighbors": create_param("n_neighbors", 1, 50),
                    "weights": create_param("weights", choices=["uniform", "distance"]),
                    "metric": create_param(
                        "metric", choices=["euclidean", "manhattan", "minkowski"]
                    ),
                },
                continuous_params={},
                fixed_params={"n_jobs": -1},
                task_type=TaskType.CLASSIFICATION,
            )
        )

    if "DecisionTreeClassifier" in class_map:
        models.append(
            ModelDefinition(
                name="DecisionTree",
                model_class=class_map["DecisionTreeClassifier"],
                discrete_params={
                    "max_depth": create_param("max_depth", 2, 50),
                    "min_samples_split": create_param("min_samples_split", 2, 20),
                    "criterion": create_param("criterion", choices=["gini", "entropy"]),
                },
                continuous_params={
                    "min_impurity_decrease": create_param(
                        "min_impurity_decrease", 0.0, 0.5
                    ),
                },
                fixed_params={"random_state": 42},
                task_type=TaskType.CLASSIFICATION,
            )
        )

    return models


def _get_regression_models(*classes: type) -> list[ModelDefinition]:
    """Get regression model definitions."""
    models = []

    class_map = {cls.__name__: cls for cls in classes}

    if "RandomForestRegressor" in class_map:
        models.append(
            ModelDefinition(
                name="RandomForest",
                model_class=class_map["RandomForestRegressor"],
                discrete_params={
                    "n_estimators": create_param("n_estimators", 10, 500),
                    "max_depth": create_param("max_depth", 2, 50),
                    "min_samples_split": create_param("min_samples_split", 2, 20),
                },
                continuous_params={
                    "max_features": create_param("max_features", 0.1, 1.0),
                },
                fixed_params={"random_state": 42, "n_jobs": -1},
                task_type=TaskType.REGRESSION,
            )
        )

    if "GradientBoostingRegressor" in class_map:
        models.append(
            ModelDefinition(
                name="GradientBoosting",
                model_class=class_map["GradientBoostingRegressor"],
                discrete_params={
                    "n_estimators": create_param("n_estimators", 50, 500),
                    "max_depth": create_param("max_depth", 2, 20),
                },
                continuous_params={
                    "learning_rate": create_param(
                        "learning_rate", 0.01, 0.3, log_scale=True
                    ),
                    "subsample": create_param("subsample", 0.5, 1.0),
                },
                fixed_params={"random_state": 42},
                task_type=TaskType.REGRESSION,
            )
        )

    if "Ridge" in class_map:
        models.append(
            ModelDefinition(
                name="Ridge",
                model_class=class_map["Ridge"],
                discrete_params={},
                continuous_params={
                    "alpha": create_param("alpha", 0.001, 100.0, log_scale=True),
                },
                fixed_params={"random_state": 42},
                task_type=TaskType.REGRESSION,
            )
        )

    if "Lasso" in class_map:
        models.append(
            ModelDefinition(
                name="Lasso",
                model_class=class_map["Lasso"],
                discrete_params={},
                continuous_params={
                    "alpha": create_param("alpha", 0.001, 100.0, log_scale=True),
                },
                fixed_params={"random_state": 42, "max_iter": 10000},
                task_type=TaskType.REGRESSION,
            )
        )

    if "ElasticNet" in class_map:
        models.append(
            ModelDefinition(
                name="ElasticNet",
                model_class=class_map["ElasticNet"],
                discrete_params={},
                continuous_params={
                    "alpha": create_param("alpha", 0.001, 100.0, log_scale=True),
                    "l1_ratio": create_param("l1_ratio", 0.0, 1.0),
                },
                fixed_params={"random_state": 42, "max_iter": 10000},
                task_type=TaskType.REGRESSION,
            )
        )

    if "SVR" in class_map:
        models.append(
            ModelDefinition(
                name="SVR",
                model_class=class_map["SVR"],
                discrete_params={
                    "kernel": create_param("kernel", choices=["rbf", "linear", "poly"]),
                },
                continuous_params={
                    "C": create_param("C", 0.01, 100.0, log_scale=True),
                    "gamma": create_param("gamma", 0.0001, 1.0, log_scale=True),
                    "epsilon": create_param("epsilon", 0.01, 1.0),
                },
                fixed_params={},
                task_type=TaskType.REGRESSION,
            )
        )

    if "KNeighborsRegressor" in class_map:
        models.append(
            ModelDefinition(
                name="KNN",
                model_class=class_map["KNeighborsRegressor"],
                discrete_params={
                    "n_neighbors": create_param("n_neighbors", 1, 50),
                    "weights": create_param("weights", choices=["uniform", "distance"]),
                },
                continuous_params={},
                fixed_params={"n_jobs": -1},
                task_type=TaskType.REGRESSION,
            )
        )

    if "DecisionTreeRegressor" in class_map:
        models.append(
            ModelDefinition(
                name="DecisionTree",
                model_class=class_map["DecisionTreeRegressor"],
                discrete_params={
                    "max_depth": create_param("max_depth", 2, 50),
                    "min_samples_split": create_param("min_samples_split", 2, 20),
                },
                continuous_params={
                    "min_impurity_decrease": create_param(
                        "min_impurity_decrease", 0.0, 0.5
                    ),
                },
                fixed_params={"random_state": 42},
                task_type=TaskType.REGRESSION,
            )
        )

    return models




class MultiModelEvaluator:
    """Evaluates fitness for multi-model population.

    Handles model-conditional parameter extraction and evaluation.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        scoring: str = "accuracy",
        use_gpu: bool = False,
    ) -> None:
        """Initialize evaluator.

        Args:
            registry: Model registry.
            scoring: Scoring metric name.
            use_gpu: Whether to use GPU models.
        """
        self.registry = registry
        self.scoring = scoring
        self.use_gpu = use_gpu

    def evaluate_individual(
        self,
        chromosome: MultiModelChromosome,
        X: NDArray,
        y: NDArray,
        cv_splits: list[tuple[NDArray, NDArray]] | None = None,
    ) -> float:
        """Evaluate a single individual.

        Args:
            chromosome: Individual to evaluate.
            X: Feature matrix.
            y: Target array.
            cv_splits: Optional pre-computed CV splits.

        Returns:
            Fitness score.
        """
        model_def = self.registry.get_model(chromosome.model_index)

        discrete_values = chromosome.get_active_discrete_values(self.registry)
        continuous_values = chromosome.get_active_continuous_values(self.registry)

        X_selected = X[:, chromosome.feature_mask]

        if X_selected.shape[1] == 0:
            return float("-inf")

        model = model_def.instantiate(
            discrete_values=discrete_values,
            continuous_values=continuous_values,
            use_gpu=self.use_gpu,
        )

        if cv_splits is None:
            n_samples = len(y)
            n_train = int(0.8 * n_samples)
            indices = np.random.permutation(n_samples)
            train_idx, test_idx = indices[:n_train], indices[n_train:]

            try:
                model.fit(X_selected[train_idx], y[train_idx])
                y_pred = model.predict(X_selected[test_idx])
                return self._score(y[test_idx], y_pred)
            except Exception as e:  # noqa: BLE001
                logger.debug("Evaluation failed: %s", e)
                return float("-inf")

        scores = []
        for train_idx, test_idx in cv_splits:
            try:
                model_copy = model_def.instantiate(
                    discrete_values=discrete_values,
                    continuous_values=continuous_values,
                    use_gpu=self.use_gpu,
                )
                model_copy.fit(X_selected[train_idx], y[train_idx])
                y_pred = model_copy.predict(X_selected[test_idx])
                scores.append(self._score(y[test_idx], y_pred))
            except Exception:  # noqa: BLE001
                scores.append(float("-inf"))

        valid_scores = [s for s in scores if np.isfinite(s)]
        return float(np.mean(valid_scores)) if valid_scores else float("-inf")

    def _score(self, y_true: NDArray, y_pred: NDArray) -> float:
        """Compute score based on metric.

        Uses the scoring module for consistent metric computation.
        """
        from .utils.scoring import get_scorer, is_minimization_metric

        try:
            scorer = get_scorer(self.scoring)
            score = scorer(y_true, y_pred)
            # For minimization metrics, negate to ensure higher is better
            if is_minimization_metric(self.scoring):
                return -float(score)
            return float(score)
        except (KeyError, ValueError):
            # Fall back to basic metrics
            if self.scoring == "accuracy":
                return float(np.mean(y_true == y_pred))
            if self.scoring == "mse":
                return -float(np.mean((y_true - y_pred) ** 2))
            if self.scoring == "r2":
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            msg = f"Unknown scoring metric: {self.scoring}"
            raise ValueError(msg)