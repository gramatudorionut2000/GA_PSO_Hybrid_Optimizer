from __future__ import annotations

import copy as copy_module
import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from scipy.stats import qmc

from ..utils.utils import (
    get_rng,
    validate_positive_int,
    validate_probability,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class ParameterSpace(ABC):
    """Abstract base class for parameter spaces.

    A parameter space defines the domain of valid values for a hyperparameter,
    along with methods for sampling, validation, and manipulation.
    """

    def __init__(self, name: str) -> None:
        """Initialize parameter space.

        Args:
            name: Unique identifier for this parameter.

        Raises:
            ValueError: If name is empty.
        """
        if not name or not name.strip():
            msg = "Parameter name cannot be empty"
            raise ValueError(msg)
        self.name = name.strip()

    @abstractmethod
    def sample(self, size: int | None = None) -> Any:
        """Sample random values from this parameter space.

        Args:
            size: Number of samples (None for single value).

        Returns:
            Single value if size is None, else array of values.
        """

    @abstractmethod
    def contains(self, value: Any) -> bool:
        """Check if a value is within this parameter space."""

    @abstractmethod
    def copy(self) -> ParameterSpace:
        """Create a deep copy of this parameter space."""

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    def denormalize(self, normalized: int | float | NDArray) -> int | float | NDArray:  # pyright: ignore[reportReturnType]
        """Convert normalized [0, 1] values back to the original scale."""
        pass


    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.name))


class ContinuousSpace(ParameterSpace):
    """Continuous parameter space for float values.

    Supports both uniform and log-uniform sampling.

    """

    def __init__(
        self,
        name: str,
        lower: float,
        upper: float,
        log_scale: bool = False,
    ) -> None:
        """Initialize continuous space.

        Args:
            name: Parameter name.
            lower: Lower bound (inclusive).
            upper: Upper bound (inclusive).
            log_scale: If True, sample uniformly in log space.
        """
        super().__init__(name)

        if lower >= upper:
            msg = f"Lower bound ({lower}) must be less than upper bound ({upper})"
            raise ValueError(msg)

        if log_scale and (lower <= 0 or upper <= 0):
            msg = "Log scale requires positive bounds"
            raise ValueError(msg)

        self.lower = float(lower)
        self.upper = float(upper)
        self.log_scale = log_scale

        # Pre-computed log bounds
        if log_scale:
            self._log_lower = np.log(self.lower)
            self._log_upper = np.log(self.upper)
        else:
            self._log_lower = 0.0
            self._log_upper = 0.0

    @property
    def range_size(self) -> float:
        """The size of the parameter range."""
        return self.upper - self.lower

    def sample(self, size: int | None = None) -> float | NDArray:
        """Sample random values."""
        rng = get_rng()

        if self.log_scale:
            log_val = rng.uniform(self._log_lower, self._log_upper, size=size)
            result = np.exp(log_val)
        else:
            result = rng.uniform(self.lower, self.upper, size=size)

        return float(result) if size is None else result

    def sample_batch(self, n: int) -> NDArray:
        """Explicitly sample n values as array."""
        return self.sample(size=n)  # pyright: ignore[reportReturnType]

    def contains(self, value: Any) -> bool:
        """Check if a value is within the bounds."""
        if not isinstance(value, (int, float, np.number)):
            return False
        return self.lower <= value <= self.upper  # pyright: ignore[reportReturnType]

    def clamp(self, value: float | NDArray) -> float | NDArray:
        """Clamp values to the bounds ."""
        result = np.clip(value, self.lower, self.upper)
        if np.ndim(result) == 0:
            return float(result)
        return result

    def normalize(self, value: float | NDArray) -> float | NDArray:
        """Normalize values to [0, 1] range"""
        value = np.asarray(value, dtype=np.float64)

        if self.log_scale:
            log_val = np.log(np.maximum(value, self.lower))
            result = (log_val - self._log_lower) / (self._log_upper - self._log_lower)
        else:
            result = (value - self.lower) / self.range_size

        return float(result) if result.ndim == 0 else result

    def denormalize(self, normalized: float | NDArray) -> float | NDArray:
        """Convert normalized [0, 1] values back to the original scale."""
        normalized = np.asarray(normalized, dtype=np.float64)

        if self.log_scale:
            log_val = self._log_lower + normalized * (self._log_upper - self._log_lower)
            result = np.exp(log_val)
        else:
            result = self.lower + normalized * self.range_size

        return float(result) if result.ndim == 0 else result

    def perturb(
        self,
        value: float | NDArray,
        strength: float = 0.1,
    ) -> float | NDArray:
        """Apply random perturbation."""
        rng = get_rng()
        value = np.asarray(value, dtype=np.float64)
        is_scalar = value.ndim == 0

        if self.log_scale:
            log_val = np.log(value)
            log_range = self._log_upper - self._log_lower
            noise = rng.normal(
                0, strength * log_range, size=value.shape if not is_scalar else None
            )
            result = self.clamp(np.exp(log_val + noise))
        else:
            noise = rng.normal(
                0, strength * self.range_size, size=value.shape if not is_scalar else None
            )
            result = self.clamp(value + noise)

        return float(result) if is_scalar else result

    def distance(
        self,
        a: float | NDArray,
        b: float | NDArray,
        *,
        normalized: bool = True,
    ) -> float | NDArray:
        """Compute distance between values."""
        if normalized:
            return np.abs(self.normalize(a) - self.normalize(b))
        if self.log_scale:
            return np.abs(np.log(a) - np.log(b))
        return np.abs(a - b)

    def get_bounds(self) -> tuple[float, float]:
        """Return (lower, upper) bounds."""
        return (self.lower, self.upper)

    def linspace(self, n: int) -> NDArray:
        """Generate n evenly spaced values."""
        if self.log_scale:
            return np.exp(np.linspace(self._log_lower, self._log_upper, n))
        return np.linspace(self.lower, self.upper, n)

    def copy(self) -> ContinuousSpace:
        """Create a copy of this space."""
        return ContinuousSpace(self.name, self.lower, self.upper, self.log_scale)

    def with_bounds(
        self,
        lower: float | None = None,
        upper: float | None = None,
    ) -> ContinuousSpace:
        """Create a copy with modified bounds."""
        new_lower = lower if lower is not None else self.lower
        new_upper = upper if upper is not None else self.upper
        return ContinuousSpace(self.name, new_lower, new_upper, self.log_scale)

    def __repr__(self) -> str:
        scale = " (log)" if self.log_scale else ""
        return f"ContinuousSpace('{self.name}', [{self.lower}, {self.upper}]{scale})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContinuousSpace):
            return False
        return (
            self.name == other.name
            and self.lower == other.lower
            and self.upper == other.upper
            and self.log_scale == other.log_scale
        )

    def __hash__(self) -> int:
        return hash(
            (self.__class__.__name__, self.name, self.lower, self.upper, self.log_scale)
        )


class IntegerSpace(ParameterSpace):
    """Integer parameter space with sampling."""

    def __init__(
        self,
        name: str,
        lower: int,
        upper: int,
        log_scale: bool = False,
    ) -> None:
        """Initialize integer space.

        Args:
            name: Parameter name.
            lower: Lower bound (inclusive).
            upper: Upper bound (inclusive).
            log_scale: If True, sample uniformly in log space.
        """
        super().__init__(name)

        if not isinstance(lower, (int, np.integer)) or not isinstance(
            upper, (int, np.integer)
        ):
            msg = "IntegerSpace bounds must be integers"
            raise ValueError(msg)

        if lower >= upper:
            msg = f"Lower bound ({lower}) must be less than upper bound ({upper})"
            raise ValueError(msg)

        if log_scale and (lower <= 0 or upper <= 0):
            msg = "Log scale requires positive bounds"
            raise ValueError(msg)

        self.lower = int(lower)
        self.upper = int(upper)
        self.log_scale = log_scale

        if log_scale:
            self._log_lower = np.log(self.lower)
            self._log_upper = np.log(self.upper)
        else:
            self._log_lower = 0.0
            self._log_upper = 0.0

    @property
    def range_size(self) -> int:
        """The number of possible values."""
        return self.upper - self.lower + 1

    def sample(self, size: int | None = None) -> int | NDArray:
        """Sample random integers."""
        rng = get_rng()

        if self.log_scale:
            log_val = rng.uniform(self._log_lower, self._log_upper, size=size)
            result = np.rint(np.exp(log_val)).astype(np.int64)
            result = np.clip(result, self.lower, self.upper)
        else:
            result = rng.integers(self.lower, self.upper + 1, size=size)

        return int(result) if size is None else result

    def sample_batch(self, n: int) -> NDArray:
        """Explicitly sample n values as array."""
        return self.sample(size=n)  # pyright: ignore[reportReturnType]

    def contains(self, value: Any) -> bool:
        """Check if value is a valid integer in bounds."""
        if not isinstance(value, (int, np.integer)):
            return False
        return self.lower <= value <= self.upper  # pyright: ignore[reportReturnType]

    def clamp(self, value: int | NDArray) -> int | NDArray:
        """Clamp values to bounds."""
        result = np.clip(value, self.lower, self.upper)
        if np.ndim(result) == 0:
            return int(result)
        return result.astype(np.int64)

    def normalize(self, value: int | NDArray) -> float | NDArray:
        """Normalize values to [0, 1]."""
        value = np.asarray(value, dtype=np.float64)

        if self.log_scale:
            log_val = np.log(np.maximum(value, self.lower))
            result = (log_val - self._log_lower) / (self._log_upper - self._log_lower)
        else:
            result = (value - self.lower) / (self.upper - self.lower)

        return float(result) if result.ndim == 0 else result

    def denormalize(self, normalized: float | NDArray) -> int | NDArray:
        """Convert normalized [0, 1] values back to integers."""
        normalized = np.asarray(normalized, dtype=np.float64)

        if self.log_scale:
            log_val = self._log_lower + normalized * (self._log_upper - self._log_lower)
            result = np.rint(np.exp(log_val)).astype(np.int64)
        else:
            result = np.rint(
                self.lower + normalized * (self.upper - self.lower)
            ).astype(np.int64)

        result = np.clip(result, self.lower, self.upper)
        return int(result) if result.ndim == 0 else result

    def perturb(
        self,
        value: int | NDArray,
        strength: float = 0.1,
    ) -> int | NDArray:
        """Apply random perturbation."""
        rng = get_rng()
        value = np.asarray(value)
        is_scalar = value.ndim == 0

        if self.log_scale:
            log_val = np.log(value.astype(np.float64))
            log_range = self._log_upper - self._log_lower
            noise = rng.normal(
                0, strength * log_range, size=value.shape if not is_scalar else None
            )
            result = self.clamp(np.rint(np.exp(log_val + noise)).astype(np.int64))
        else:
            range_size = self.upper - self.lower
            noise = rng.normal(
                0, strength * range_size, size=value.shape if not is_scalar else None
            )
            result = self.clamp(np.rint(value + noise).astype(np.int64))

        return int(result) if is_scalar else result

    def distance(
        self,
        a: int | NDArray,
        b: int | NDArray,
        *,
        normalized: bool = True,
    ) -> float | NDArray:
        """Compute distance between values."""
        if normalized:
            return np.abs(self.normalize(a) - self.normalize(b))
        return np.abs(a - b)

    def get_bounds(self) -> tuple[int, int]:
        """Return (lower, upper) bounds."""
        return (self.lower, self.upper)

    def arange(self) -> NDArray:
        """Return all possible values as array."""
        return np.arange(self.lower, self.upper + 1)

    def linspace(self, n: int) -> NDArray:
        """Generate n evenly spaced integer values."""
        if n >= self.range_size:
            return self.arange()

        if self.log_scale:
            log_vals = np.linspace(self._log_lower, self._log_upper, n)
            values = np.rint(np.exp(log_vals)).astype(np.int64)
        else:
            values = np.rint(np.linspace(self.lower, self.upper, n)).astype(np.int64)

        return np.unique(values)

    def copy(self) -> IntegerSpace:
        """Create a copy of this space."""
        return IntegerSpace(self.name, self.lower, self.upper, self.log_scale)

    def with_bounds(
        self,
        lower: int | None = None,
        upper: int | None = None,
    ) -> IntegerSpace:
        """Create a copy with modified bounds."""
        new_lower = lower if lower is not None else self.lower
        new_upper = upper if upper is not None else self.upper
        return IntegerSpace(self.name, new_lower, new_upper, self.log_scale)

    def __repr__(self) -> str:
        scale = " (log)" if self.log_scale else ""
        return f"IntegerSpace('{self.name}', [{self.lower}, {self.upper}]{scale})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntegerSpace):
            return False
        return (
            self.name == other.name
            and self.lower == other.lower
            and self.upper == other.upper
            and self.log_scale == other.log_scale
        )

    def __hash__(self) -> int:
        return hash(
            (self.__class__.__name__, self.name, self.lower, self.upper, self.log_scale)
        )


class CategoricalSpace(ParameterSpace):
    """Categorical parameter space."""

    def __init__(
        self,
        name: str,
        choices: list[Any],
        weights: list[float] | None = None,
    ) -> None:
        """Initialize categorical space.

        Args:
            name: Parameter name.
            choices: List of possible values.
            weights: Optional probability weights for each choice.
        """
        super().__init__(name)

        if not choices:
            msg = "Choices cannot be empty"
            raise ValueError(msg)

        self.choices = list(choices)

        if weights is not None:
            weights_arr = np.asarray(weights, dtype=np.float64)
            if len(weights_arr) != len(choices):
                msg = "Weights must match choices length"
                raise ValueError(msg)
            if np.any(weights_arr < 0):
                msg = "Weights must be non-negative"
                raise ValueError(msg)
            if weights_arr.sum() == 0:
                msg = "At least one weight must be positive"
                raise ValueError(msg)
            self._weights: NDArray | None = weights_arr / weights_arr.sum()
        else:
            self._weights = None

    @property
    def n_choices(self) -> int:
        """Number of choices."""
        return len(self.choices)

    @property
    def weights(self) -> NDArray | None:
        """Probability weights."""
        return self._weights

    def sample(self, size: int | None = None) -> Any:
        """Sample random choices"""
        rng = get_rng()
        indices = rng.choice(self.n_choices, size=size, p=self._weights)

        if size is None:
            return self.choices[indices]
        return [self.choices[i] for i in indices]  # pyright: ignore[reportGeneralTypeIssues]

    def sample_batch(self, n: int) -> list[Any]:
        """Explicitly sample n values as list."""
        return self.sample(size=n)

    def contains(self, value: Any) -> bool:
        """Check if value is in choices."""
        return value in self.choices

    def index_of(self, value: Any) -> int:
        """Get index of value in choices."""
        return self.choices.index(value)

    def one_hot(self, value: Any) -> NDArray:
        """Return one-hot encoding of value."""
        encoding = np.zeros(self.n_choices, dtype=np.float64)
        encoding[self.index_of(value)] = 1.0
        return encoding

    def distance(self, a: Any, b: Any) -> float:
        """Binary distance: 0 if same, 1 if different."""
        return 0.0 if a == b else 1.0

    def copy(self) -> CategoricalSpace:
        """Create a copy of this space."""
        weights_list = self._weights.tolist() if self._weights is not None else None
        return CategoricalSpace(
            self.name,
            copy_module.deepcopy(self.choices),
            weights_list,
        )


    def __repr__(self) -> str:
        if self._weights is not None:
            return (
                f"CategoricalSpace('{self.name}', {self.choices}, "
                f"weights={self._weights.tolist()})"
            )
        return f"CategoricalSpace('{self.name}', {self.choices})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CategoricalSpace):
            return False
        if self._weights is not None:
            weights_eq = np.array_equal(self._weights, other._weights)  # pyright: ignore[reportArgumentType]
        else:
            weights_eq = other._weights is None
        return self.name == other.name and self.choices == other.choices and weights_eq

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.name, tuple(self.choices)))


class BooleanSpace(ParameterSpace):
    """Boolean parameter space with sampling."""

    def __init__(self, name: str, true_probability: float = 0.5) -> None:
        """Initialize boolean space.

        Args:
            name: Parameter name.
            true_probability: Probability of sampling True.
        """
        super().__init__(name)
        validate_probability(true_probability, "true_probability")
        self.true_probability = true_probability

    def sample(self, size: int | None = None) -> bool | NDArray:
        """Sample random booleans using NumPy."""
        rng = get_rng()
        result = rng.random(size=size) < self.true_probability

        if size is None:
            return bool(result)
        return result

    def sample_batch(self, n: int) -> NDArray:
        """Sample n booleans as array."""
        return self.sample(size=n)  # pyright: ignore[reportReturnType]

    def contains(self, value: Any) -> bool:
        """Check if value is a boolean."""
        return isinstance(value, (bool, np.bool_))

    def distance(self, a: bool, b: bool) -> float:
        """Binary distance: 0 if same, 1 if different."""
        return 0.0 if a == b else 1.0

    def copy(self) -> BooleanSpace:
        """Create a copy of this space."""
        return BooleanSpace(self.name, self.true_probability)

    def __repr__(self) -> str:
        if self.true_probability != 0.5:
            return f"BooleanSpace('{self.name}', p_true={self.true_probability})"
        return f"BooleanSpace('{self.name}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BooleanSpace):
            return False
        return (
            self.name == other.name
            and self.true_probability == other.true_probability
        )

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.name, self.true_probability))


class HyperparameterSpace:
    """Container for multiple parameter spaces operations.

    Supports advanced sampling methods:
    - Random sampling
    - Latin Hypercube sampling (SciPy)
    - Sobol sequence sampling (SciPy)
    - Grid sampling
    """

    def __init__(self) -> None:
        """Initialize empty hyperparameter space."""
        self._spaces: dict[str, ParameterSpace] = {}
        self._order: list[str] = []

    @property
    def spaces(self) -> dict[str, ParameterSpace]:
        """Dictionary of parameter spaces."""
        return self._spaces

    @property
    def names(self) -> list[str]:
        """List of parameter names in order."""
        return list(self._order)

    @property
    def n_continuous(self) -> int:
        """Number of continuous parameters."""
        return sum(1 for s in self._spaces.values() if isinstance(s, ContinuousSpace))

    @property
    def n_integer(self) -> int:
        """Number of integer parameters."""
        return sum(1 for s in self._spaces.values() if isinstance(s, IntegerSpace))

    @property
    def n_numeric(self) -> int:
        """Number of numeric (continuous + integer) parameters."""
        return self.n_continuous + self.n_integer

    def add(self, space: ParameterSpace) -> HyperparameterSpace:
        """Add a parameter space.

        Args:
            space: Parameter space to add.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If parameter name already exists.
        """
        if space.name in self._spaces:
            msg = f"Parameter '{space.name}' already exists"
            raise ValueError(msg)

        self._spaces[space.name] = space
        self._order.append(space.name)
        return self

    def add_continuous(
        self,
        name: str,
        lower: float,
        upper: float,
        log_scale: bool = False,
    ) -> HyperparameterSpace:
        """Add a continuous parameter space."""
        return self.add(ContinuousSpace(name, lower, upper, log_scale))

    def add_integer(
        self,
        name: str,
        lower: int,
        upper: int,
        log_scale: bool = False,
    ) -> HyperparameterSpace:
        """Add an integer parameter space."""
        return self.add(IntegerSpace(name, lower, upper, log_scale))

    def add_categorical(
        self,
        name: str,
        choices: list[Any],
        weights: list[float] | None = None,
    ) -> HyperparameterSpace:
        """Add a categorical parameter space."""
        return self.add(CategoricalSpace(name, choices, weights))

    def add_boolean(
        self,
        name: str,
        true_probability: float = 0.5,
    ) -> HyperparameterSpace:
        """Add a boolean parameter space."""
        return self.add(BooleanSpace(name, true_probability))

    def get(self, name: str) -> ParameterSpace | None:
        """Get a parameter space by name."""
        return self._spaces.get(name)

    def __getitem__(self, name: str) -> ParameterSpace:
        if name not in self._spaces:
            msg = f"Parameter '{name}' not found"
            raise KeyError(msg)
        return self._spaces[name]

    def __contains__(self, name: str) -> bool:
        return name in self._spaces

    def __len__(self) -> int:
        return len(self._spaces)

    def __iter__(self) -> Iterator[str]:
        return iter(self._order)

    def sample(self) -> dict[str, Any]:
        """Sample a single configuration."""
        return {name: self._spaces[name].sample() for name in self._order}

    def sample_n(self, n: int) -> list[dict[str, Any]]:
        """Generate n random samples using batch sampling."""
        validate_positive_int(n, "n")

        # Batch sample from each space
        samples_per_param: dict[str, Any] = {}
        for name in self._order:
            space = self._spaces[name]
            samples_per_param[name] = space.sample(size=n)

        # Convert to list of dicts
        return [
            {
                name: (
                    samples_per_param[name][i]
                    if hasattr(samples_per_param[name], "__getitem__")
                    and not isinstance(samples_per_param[name], str)
                    else samples_per_param[name]
                )
                for name in self._order
            }
            for i in range(n)
        ]

    def sample_latin_hypercube(
        self,
        n: int,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Generate n samples using Latin Hypercube Sampling.

        LHS provides better coverage of the parameter space than random sampling.
        Only applies to numeric parameters; categorical/boolean are sampled randomly.
        """
        validate_positive_int(n, "n")

        # Get numeric parameter names
        numeric_names = [
            name
            for name in self._order
            if isinstance(self._spaces[name], (ContinuousSpace, IntegerSpace))
        ]
        n_numeric = len(numeric_names)

        samples = []

        if n_numeric > 0:
            # Generate LHS samples in [0, 1]^d
            sampler = qmc.LatinHypercube(d=n_numeric, rng=seed)
            lhs_samples = sampler.random(n=n)

            # Transform to parameter ranges
            for i in range(n):
                config: dict[str, Any] = {}
                for j, name in enumerate(numeric_names):
                    space = self._spaces[name]
                    config[name] = space.denormalize(lhs_samples[i, j])

                # Sample non-numeric parameters randomly
                for name in self._order:
                    if name not in config:
                        config[name] = self._spaces[name].sample()

                samples.append(config)
        else:
            samples = self.sample_n(n)

        return samples

    def sample_sobol(
        self,
        n: int,
        seed: int | None = None,
        *,
        scramble: bool = True,
    ) -> list[dict[str, Any]]:
        """Generate n samples using Sobol sequence.

        n should ideally be a power of 2.
        """
        validate_positive_int(n, "n")

        numeric_names = [
            name
            for name in self._order
            if isinstance(self._spaces[name], (ContinuousSpace, IntegerSpace))
        ]
        n_numeric = len(numeric_names)

        samples = []

        if n_numeric > 0:
            sampler = qmc.Sobol(d=n_numeric, scramble=scramble, rng=seed)
            sobol_samples = sampler.random(n=n)

            for i in range(n):
                config: dict[str, Any] = {}
                for j, name in enumerate(numeric_names):
                    space = self._spaces[name]
                    config[name] = space.denormalize(sobol_samples[i, j])

                for name in self._order:
                    if name not in config:
                        config[name] = self._spaces[name].sample()

                samples.append(config)
        else:
            samples = self.sample_n(n)

        return samples

    def contains(self, config: dict[str, Any]) -> bool:
        """Check if a configuration is valid."""
        if set(config.keys()) != set(self._spaces.keys()):
            return False
        return all(
            self._spaces[name].contains(config[name]) for name in self._spaces
        )

    def clamp(self, config: dict[str, Any]) -> dict[str, Any]:
        """Clamp all values to valid ranges."""
        result = {}
        for name in self._order:
            space = self._spaces[name]
            value = config.get(name)

            if value is None:
                result[name] = space.sample()
            elif isinstance(space, (ContinuousSpace, IntegerSpace)):
                result[name] = space.clamp(value)
            elif space.contains(value):
                result[name] = value
            else:
                result[name] = space.sample()

        return result

    def to_array(self, config: dict[str, Any]) -> NDArray:
        """Convert a configuration to a numeric array (normalized [0, 1]).

        Only includes numeric parameters. Useful for distance calculations.
        """
        values = []
        for name in self._order:
            space = self._spaces[name]
            if isinstance(space, (ContinuousSpace, IntegerSpace)):
                values.append(space.normalize(config[name]))
            elif isinstance(space, CategoricalSpace):
                values.append(space.index_of(config[name]) / max(1, space.n_choices - 1))
            elif isinstance(space, BooleanSpace):
                values.append(1.0 if config[name] else 0.0)
        return np.array(values, dtype=np.float64)

    def configs_to_array(self, configs: list[dict[str, Any]]) -> NDArray:
        """Convert multiple configurations to 2D array."""
        return np.vstack([self.to_array(c) for c in configs])

    def distance(
        self,
        a: dict[str, Any],
        b: dict[str, Any],
        *,
        normalized: bool = True,
    ) -> float:
        """Compute Euclidean distance between configurations."""
        if normalized:
            arr_a = self.to_array(a)
            arr_b = self.to_array(b)
            return float(np.linalg.norm(arr_a - arr_b))

        squared_sum = 0.0
        for name, space in self._spaces.items():
            d = space.distance(a[name], b[name], normalized=False)  # pyright: ignore[reportAttributeAccessIssue]
            squared_sum += d**2
        return float(np.sqrt(squared_sum))

    def pairwise_distances(self, configs: list[dict[str, Any]]) -> NDArray:
        """Compute pairwise distance matrix for configurations."""
        x = self.configs_to_array(configs)
        return cdist(x, x, metric="euclidean")

    def get_numeric_bounds(self) -> dict[str, tuple[float, float]]:
        """Get bounds for all numeric parameters."""
        bounds = {}
        for name, space in self._spaces.items():
            if isinstance(space, (ContinuousSpace, IntegerSpace)):
                bounds[name] = space.get_bounds()
        return bounds

    def grid_sample(self, n_samples_per_dim: int = 10) -> list[dict[str, Any]]:
        """Generate a grid of samples"""
        validate_positive_int(n_samples_per_dim, "n_samples_per_dim")

        param_values: dict[str, list[Any]] = {}
        for name in self._order:
            space = self._spaces[name]

            if isinstance(space, (ContinuousSpace, IntegerSpace)):
                param_values[name] = space.linspace(n_samples_per_dim).tolist()
            elif isinstance(space, CategoricalSpace):
                param_values[name] = space.choices
            elif isinstance(space, BooleanSpace):
                param_values[name] = [True, False]

        keys = list(param_values.keys())
        values = [param_values[k] for k in keys]
        combinations = list(itertools.product(*values))

        return [dict(zip(keys, combo, strict=True)) for combo in combinations]

    def copy(self) -> HyperparameterSpace:
        """Create a deep copy of this space."""
        new_space = HyperparameterSpace()
        for name in self._order:
            new_space.add(self._spaces[name].copy())
        return new_space

    def __repr__(self) -> str:
        if not self._spaces:
            return "HyperparameterSpace(empty)"

        spaces_str = "\n  ".join(str(self._spaces[name]) for name in self._order)
        return f"HyperparameterSpace(\n  {spaces_str}\n)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HyperparameterSpace):
            return False
        return self._order == other._order and all(
            self._spaces[n] == other._spaces[n] for n in self._order
        )