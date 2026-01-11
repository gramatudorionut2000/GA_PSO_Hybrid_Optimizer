from __future__ import annotations

import copy as copy_module
import inspect
from typing import TYPE_CHECKING, Any

import numpy as np

from .parameter_space import (
    HyperparameterSpace,
    ParameterSpace,
    ContinuousSpace,
    IntegerSpace,
    CategoricalSpace,
    BooleanSpace,
)
from ..utils.utils import get_rng
from ..utils.common import TaskType

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from ..multi_model import ModelDefinition, ModelRegistry, ParameterDefinition


class ModelInstantiationError(Exception):
    """Raised when model instantiation fails."""

    def __init__(
        self,
        model_class: type,
        params: dict[str, Any],
        original_error: Exception,
    ) -> None:
        """

        Args:
            model_class: The model class that failed to instantiate.
            params: The parameters that were passed.
            original_error: The original exception that was raised.
        """
        self.model_class = model_class
        self.params = params
        self.original_error = original_error
        super().__init__(
            f"Failed to instantiate {model_class.__name__} with params {params}: "
            f"{type(original_error).__name__}: {original_error}"
        )


class ModelSpace:
    """Combines a model class with a hyperparameter space for instantiation.

    """

    def __init__(
        self,
        model_class: type,
        param_space: HyperparameterSpace | None = None,
        fixed_params: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize model space.

        Args:
            model_class: The model class to instantiate.
            param_space: Optional existing hyperparameter space.
            fixed_params: Parameters that are always fixed.
            name: Optional name for this model space.

        Raises:
            ValueError: If model_class is not a class.
        """
        if not inspect.isclass(model_class):
            msg = f"model_class must be a class, got {type(model_class)}"
            raise ValueError(msg)

        self.model_class = model_class
        self.param_space = (
            param_space.copy() if param_space is not None else HyperparameterSpace()
        )
        self.fixed_params = dict(fixed_params) if fixed_params else {}
        self.name = name or model_class.__name__

        self._on_instantiate: Callable[[Any, dict[str, Any]], None] | None = None
        self._param_validator: Callable[[dict[str, Any]], bool] | None = None

    def add(self, space: ParameterSpace) -> ModelSpace:
        """Add a parameter space."""
        self.param_space.add(space)
        return self

    def add_continuous(
        self,
        name: str,
        lower: float,
        upper: float,
        log_scale: bool = False,
    ) -> ModelSpace:
        self.param_space.add_continuous(name, lower, upper, log_scale)
        return self

    def add_integer(
        self,
        name: str,
        lower: int,
        upper: int,
        log_scale: bool = False,
    ) -> ModelSpace:
        self.param_space.add_integer(name, lower, upper, log_scale)
        return self

    def add_categorical(
        self,
        name: str,
        choices: list[Any],
        weights: list[float] | None = None,
    ) -> ModelSpace:
        self.param_space.add_categorical(name, choices, weights)
        return self

    def add_boolean(
        self,
        name: str,
        true_probability: float = 0.5,
    ) -> ModelSpace:
        self.param_space.add_boolean(name, true_probability)
        return self

    def set_fixed(self, **params: Any) -> ModelSpace:
        self.fixed_params.update(params)
        return self

    def remove_fixed(self, *param_names: str) -> ModelSpace:
        for name in param_names:
            self.fixed_params.pop(name, None)
        return self

    def set_param_validator(
        self,
        validator: Callable[[dict[str, Any]], bool],
    ) -> ModelSpace:
        """Set a parameter validator function."""
        self._param_validator = validator
        return self

    def set_on_instantiate(
        self,
        callback: Callable[[Any, dict[str, Any]], None],
    ) -> ModelSpace:
        """Set a callback to run after instantiation."""
        self._on_instantiate = callback
        return self

    def validate_params(self, params: dict[str, Any]) -> bool:
        """Validate parameters against the space and validator."""
        for name, space in self.param_space.spaces.items():
            if name in params and not space.contains(params[name]):
                return False

        if self._param_validator is not None:
            full_params = {**self.fixed_params, **params}
            if not self._param_validator(full_params):
                return False

        return True

    def sample_params(self) -> dict[str, Any]:
        params = self.param_space.sample()
        params.update(self.fixed_params)
        return params

    def sample_params_batch(self, n: int) -> list[dict[str, Any]]:
        params_list = self.param_space.sample_n(n)
        for params in params_list:
            params.update(self.fixed_params)
        return params_list

    def sample_valid_params(self, max_attempts: int = 100) -> dict[str, Any]:
        """Sample valid parameters, retrying if necessary."""
        for _ in range(max_attempts):
            params = self.sample_params()
            if self.validate_params(params):
                return params

        msg = f"Could not sample valid parameters within {max_attempts} attempts."
        raise RuntimeError(msg)

    def instantiate(self, **override_params: Any) -> Any:
        params = self.sample_params()
        params.update(override_params)

        try:
            model = self.model_class(**params)

            if self._on_instantiate is not None:
                self._on_instantiate(model, params)

            return model

        except Exception as e:
            raise ModelInstantiationError(self.model_class, params, e) from e

    def instantiate_with_params(self, params: dict[str, Any]) -> Any:
        """Instantiate the model with specific parameters."""
        full_params = {**self.fixed_params, **params}

        try:
            model = self.model_class(**full_params)

            if self._on_instantiate is not None:
                self._on_instantiate(model, full_params)

            return model

        except Exception as e:
            raise ModelInstantiationError(self.model_class, full_params, e) from e

    def try_instantiate_with_params(self, params: dict[str, Any]) -> Any | None:
        """Try to instantiate with params. Returns None on failure."""
        try:
            return self.instantiate_with_params(params)
        except (ModelInstantiationError, Exception):
            return None

    def instantiate_n(self, n: int, **override_params: Any) -> list[Any]:
        """Instantiate n models."""
        return [self.instantiate(**override_params) for _ in range(n)]

    def instantiate_grid(self, n_samples_per_dim: int = 10) -> list[Any]:
        """Instantiate models from a parameter grid."""
        param_grid = self.param_space.grid_sample(n_samples_per_dim)
        models = []

        for params in param_grid:
            model = self.try_instantiate_with_params(params)
            if model is not None:
                models.append(model)

        return models

    def generate_models(self, n: int, **override_params: Any) -> Iterator[Any]:
        """Generate n models (lasy initialization)."""
        for _ in range(n):
            yield self.instantiate(**override_params)

    def get_param_names(self) -> list[str]:
        """Get names of tunable parameters."""
        return list(self.param_space.names)

    def get_fixed_param_names(self) -> list[str]:
        """Get names of fixed parameters."""
        return list(self.fixed_params.keys())

    def get_all_param_names(self) -> list[str]:
        """Get names of all parameters."""
        return list(set(self.param_space.names) | set(self.fixed_params.keys()))

    def copy(self) -> ModelSpace:
        """Create a deep copy of the model space."""
        new_space = ModelSpace(
            self.model_class,
            self.param_space.copy(),
            copy_module.deepcopy(self.fixed_params),
            self.name,
        )
        new_space._on_instantiate = self._on_instantiate
        new_space._param_validator = self._param_validator
        return new_space

    def __len__(self) -> int:
        return len(self.param_space)

    def __repr__(self) -> str:
        fixed_str = f", fixed={self.fixed_params}" if self.fixed_params else ""
        return f"ModelSpace({self.model_class.__name__}, {self.param_space}{fixed_str})"

    def to_model_definition(
        self,
        task_type: TaskType = TaskType.CLASSIFICATION,
        weight: float = 1.0,
    ) -> ModelDefinition:
        """Convert ModelSpace to ModelDefinition for use with ModelRegistry.

        This enables integration between the ModelSpace and multi-model
        optimization framework.

        Args:
            task_type: Type of ML task.
            weight: Sampling weight for this model.

        Returns:
            ModelDefinition compatible with ModelRegistry.
        """
        from ..multi_model import ModelDefinition, ParameterDefinition, ParameterType

        discrete_params: dict[str, ParameterDefinition] = {}
        continuous_params: dict[str, ParameterDefinition] = {}

        for name, space in self.param_space.spaces.items():
            if isinstance(space, IntegerSpace):
                discrete_params[name] = ParameterDefinition(
                    name=name,
                    param_type=ParameterType.INTEGER,
                    lower=space.lower,
                    upper=space.upper,
                    log_scale=space.log_scale,
                )
            elif isinstance(space, CategoricalSpace):
                discrete_params[name] = ParameterDefinition(
                    name=name,
                    param_type=ParameterType.CATEGORICAL,
                    choices=list(space.choices),
                )
            elif isinstance(space, BooleanSpace):
                discrete_params[name] = ParameterDefinition(
                    name=name,
                    param_type=ParameterType.BOOLEAN,
                )
            elif isinstance(space, ContinuousSpace):
                continuous_params[name] = ParameterDefinition(
                    name=name,
                    param_type=ParameterType.CONTINUOUS,
                    lower=space.lower,
                    upper=space.upper,
                    log_scale=space.log_scale,
                )

        return ModelDefinition(
            name=self.name,
            model_class=self.model_class,
            discrete_params=discrete_params,
            continuous_params=continuous_params,
            fixed_params=self.fixed_params,
            task_type=task_type,
            weight=weight,
        )


def create_model_space(model_class: type, **param_specs: Any) -> ModelSpace:
    """Quick way to create a ModelSpace with parameter specifications.

    Parameter specs can be:
        - tuple of (lower, upper) for continuous
        - tuple of (lower, upper, 'int') for integer
        - tuple of (lower, upper, 'log') for log-scale continuous
        - tuple of (lower, upper, 'int', 'log') for log-scale integer
        - list for categorical
        - 'bool' for boolean
        - Any other value is treated as a fixed parameter
    """
    space = ModelSpace(model_class)

    for name, spec in param_specs.items():
        if spec == "bool":
            space.add_boolean(name)

        elif isinstance(spec, list):
            space.add_categorical(name, spec)

        elif isinstance(spec, tuple):
            if len(spec) == 2:
                space.add_continuous(name, float(spec[0]), float(spec[1]))

            elif len(spec) == 3:
                if spec[2] == "int":
                    space.add_integer(name, int(spec[0]), int(spec[1]))
                elif spec[2] == "log":
                    space.add_continuous(
                        name, float(spec[0]), float(spec[1]), log_scale=True
                    )
                else:
                    msg = f"Unknown spec modifier: {spec[2]}"
                    raise ValueError(msg)

            elif len(spec) == 4:
                modifiers = set(spec[2:])
                if modifiers == {"int", "log"}:
                    space.add_integer(
                        name, int(spec[0]), int(spec[1]), log_scale=True
                    )
                else:
                    msg = f"Unknown spec modifiers: {spec[2:]}"
                    raise ValueError(msg)
            else:
                msg = f"Invalid tuple spec for {name}: {spec}"
                raise ValueError(msg)

        else:
            space.set_fixed(**{name: spec})

    return space


class ModelSpaceUnion:
    """Union of multiple ModelSpaces for algorithm selection."""

    def __init__(
        self,
        spaces: list[ModelSpace],
        weights: list[float] | None = None,
    ) -> None:
        """Initialize model space union.

        Args:
            spaces: List of model spaces.
            weights: Optional weights for each space.

        Raises:
            ValueError: If no spaces provided or weights don't match.
        """
        if not spaces:
            msg = "Must provide at least one ModelSpace"
            raise ValueError(msg)

        self.spaces = list(spaces)

        if weights is not None:
            weights_arr = np.asarray(weights, dtype=np.float64)
            if len(weights_arr) != len(spaces):
                msg = "Weights must match number of spaces"
                raise ValueError(msg)
            self._weights = weights_arr / weights_arr.sum()
        else:
            self._weights = np.ones(len(spaces)) / len(spaces)

    def sample_space(self) -> ModelSpace:
        """Sample a model space according to weights."""
        rng = get_rng()
        idx = rng.choice(len(self.spaces), p=self._weights)
        return self.spaces[idx]

    def sample_model(self) -> tuple[Any, int]:
        """Sample a model and return it with its space index."""
        rng = get_rng()
        idx = rng.choice(len(self.spaces), p=self._weights)
        return self.spaces[idx].instantiate(), idx

    def sample_params(self) -> tuple[dict[str, Any], int, type]:
        """Sample parameters and return with space index and model class."""
        rng = get_rng()
        idx = rng.choice(len(self.spaces), p=self._weights)
        space = self.spaces[idx]
        return space.sample_params(), idx, space.model_class

    def __len__(self) -> int:
        return len(self.spaces)

    def __repr__(self) -> str:
        names = [s.name for s in self.spaces]
        return f"ModelSpaceUnion({names})"

    def to_model_registry(
        self,
        task_type: TaskType = TaskType.CLASSIFICATION,
    ) -> ModelRegistry:
        """Convert ModelSpaceUnion to ModelRegistry for multi-model optimization.

        Args:
            task_type: Type of ML task.

        Returns:
            ModelRegistry with all model spaces registered.
        """
        from ..multi_model import ModelRegistry

        registry = ModelRegistry(task_type=task_type)

        for space, weight in zip(self.spaces, self._weights):
            model_def = space.to_model_definition(
                task_type=task_type,
                weight=float(weight),
            )
            registry.register(model_def)

        return registry

    @classmethod
    def from_model_registry(cls, registry: ModelRegistry) -> ModelSpaceUnion:
        """Create ModelSpaceUnion from ModelRegistry.

        Args:
            registry: Source ModelRegistry.

        Returns:
            ModelSpaceUnion containing all models from registry.
        """
        spaces: list[ModelSpace] = []
        weights: list[float] = []

        for model_def in registry:
            space = model_def.to_model_space()
            spaces.append(space)
            weights.append(model_def.weight)

        return cls(spaces, weights)