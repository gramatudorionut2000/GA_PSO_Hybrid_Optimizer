
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from collections.abc import Hashable

Numeric = int | float

_RNG: np.random.Generator | None = None




def set_random_seed(seed: int | None) -> None:
    """Set the global random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    global _RNG  # noqa: PLW0603
    _RNG = np.random.default_rng(seed)


def get_rng() -> np.random.Generator:
    """Get the global random number generator.

    Returns:
         random generator instance.
    """
    global _RNG  # noqa: PLW0603
    if _RNG is None:
        _RNG = np.random.default_rng()
    return _RNG




def validate_probability(value: float, name: str = "probability") -> None:
    """Validate that a value is a valid probability in [0, 1].

    Args:
        value: Value to validate.
        name: Parameter name for error messages.

    Raises:
        ValueError: If value is not in [0, 1].
    """
    if not 0.0 <= value <= 1.0:
        msg = f"{name} must be in [0, 1], got {value}"
        raise ValueError(msg)


def validate_positive(value: float, name: str = "value") -> None:
    """Validate that a value is positive (> 0).

    Args:
        value: Value to validate.
        name: Parameter name for error messages.

    Raises:
        ValueError: If value is not positive.
    """
    if value <= 0:
        msg = f"{name} must be positive, got {value}"
        raise ValueError(msg)


def validate_non_negative(value: float, name: str = "value") -> None:
    """Validate that a value is non-negative (>= 0).

    Args:
        value: Value to validate.
        name: Parameter name for error messages.

    Raises:
        ValueError: If value is negative.
    """
    if value < 0:
        msg = f"{name} must be non-negative, got {value}"
        raise ValueError(msg)


def validate_positive_int(value: int, name: str = "value") -> None:
    """Validate that a value is a positive integer.

    Args:
        value: Value to validate.
        name: Parameter name for error messages.

    Raises:
        ValueError: If value is not a positive integer.
    """
    if not isinstance(value, (int, np.integer)) or value <= 0:
        msg = f"{name} must be a positive integer, got {value}"
        raise ValueError(msg)





def clamp(
    value: Numeric | NDArray,
    min_val: Numeric,
    max_val: Numeric,
) -> Numeric | NDArray:
    """Clamp a value to be within [min_val, max_val].

    Args:
        value: The value to clamp (scalar or array).
        min_val: Minimum boundary.
        max_val: Maximum boundary.

    Returns:
        The clamped value

    Raises:
        ValueError: If min_val > max_val.

    Examples:
        >>> clamp(15, 0, 10)
        10
        >>> clamp(-5, 0, 10)
        0
        >>> clamp(np.array([15, -5, 5]), 0, 10)
        array([10,  0,  5])
    """
    if min_val > max_val:
        msg = f"min_val ({min_val}) must be <= max_val ({max_val})"
        raise ValueError(msg)

    result = np.clip(value, min_val, max_val)
    if np.ndim(result) == 0:
        return type(value)(result) if isinstance(value, (int, float)) else float(result)
    return result


def reflect(
    value: Numeric | NDArray,
    min_val: Numeric,
    max_val: Numeric,
) -> float | NDArray:
    """Reflect a value back into [min_val, max_val] when it exceeds boundaries.

    

    Args:
        value: The value to reflect (scalar or array).
        min_val: Minimum boundary.
        max_val: Maximum boundary.

    Returns:
        The reflected value as a float or array.

    Raises:
        ValueError: If min_val >= max_val.

    Examples:
        >>> reflect(12, 0, 10)
        8.0
        >>> reflect(-3, 0, 10)
        3.0
    """
    if min_val >= max_val:
        msg = f"min_val ({min_val}) must be < max_val ({max_val})"
        raise ValueError(msg)

    value = np.asarray(value, dtype=np.float64)
    range_size = max_val - min_val

    # Normalize value to [0, range_size] reference frame
    normalized = value - min_val

    cycle_length = 2 * range_size

    position_in_cycle = normalized % cycle_length

    # If in first half of cycle, going forward; if in second half, going backward
    result = np.where(
        position_in_cycle <= range_size,
        min_val + position_in_cycle,
        max_val - (position_in_cycle - range_size),
    )

    return float(result) if result.ndim == 0 else result


def wrap(
    value: Numeric | NDArray,
    min_val: Numeric,
    max_val: Numeric,
) -> float | NDArray:
    """Wrap a value around the boundaries [min_val, max_val).

    When exceeding max, wraps to min, and vice versa.
    Note: The upper bound is exclusive to avoid ambiguity at boundaries.

    Args:
        value: The value to wrap (scalar or array).
        min_val: Minimum boundary (inclusive).
        max_val: Maximum boundary (exclusive).

    Returns:
        The wrapped value as a float or array.

    Raises:
        ValueError: If min_val >= max_val.

    Examples:
        >>> wrap(12, 0, 10)
        2.0
        >>> wrap(-3, 0, 10)
        7.0
    """
    if min_val >= max_val:
        msg = f"min_val ({min_val}) must be < max_val ({max_val})"
        raise ValueError(msg)

    value = np.asarray(value, dtype=np.float64)
    range_size = max_val - min_val
    normalized = (value - min_val) % range_size
    result = min_val + normalized

    return float(result) if result.ndim == 0 else result



def lerp(
    a: Numeric | NDArray,
    b: Numeric | NDArray,
    t: Numeric | NDArray,
) -> float | NDArray:
    """Linear interpolation between set values.

    Args:
        a: Start value.
        b: End value.
        t: Interpolation factor (0 = a, 1 = b).

    Returns:
        Interpolated value.

    Examples:
        >>> lerp(0, 10, 0.5)
        5.0
        >>> lerp(0, 10, 0.25)
        2.5
    """
    result = np.asarray(a) + np.asarray(t) * (np.asarray(b) - np.asarray(a))
    return float(result) if result.ndim == 0 else result


def inverse_lerp(
    a: Numeric,
    b: Numeric,
    value: Numeric | NDArray,
) -> float | NDArray:
    """Inverse linear interpolation - find t such that lerp(a, b, t) = value.

    Args:
        a: Start value.
        b: End value.
        value: Value to find the interpolation factor for.

    Returns:
        Interpolation factor t.

    Examples:
        >>> inverse_lerp(0, 10, 5)
        0.5
    """
    if a == b:
        return 0.0 if np.isscalar(value) else np.zeros_like(value, dtype=np.float64)
    result = (np.asarray(value) - a) / (b - a)
    return float(result) if result.ndim == 0 else result


def remap(
    value: Numeric | NDArray,
    in_min: Numeric,
    in_max: Numeric,
    out_min: Numeric,
    out_max: Numeric,
) -> float | NDArray:
    """Remap a value from one range to another.

    Args:
        value: Value to remap.
        in_min: Input range minimum.
        in_max: Input range maximum.
        out_min: Output range minimum.
        out_max: Output range maximum.

    Returns:
        Remapped value.

    Examples:
        >>> remap(5, 0, 10, 0, 100)
        50.0
    """
    t = inverse_lerp(in_min, in_max, value)
    return lerp(out_min, out_max, t)




def random_in_range(low: float, high: float, size: int | None = None) -> float | NDArray:
    """Generate random float(s) in [low, high).

    Args:
        low: Lower bound (inclusive).
        high: Upper bound (exclusive).
        size: Number of samples (None for single value).

    Returns:
        Random value(s).
    """
    rng = get_rng()
    result = rng.uniform(low, high, size=size)
    return float(result) if size is None else result


def random_int_in_range(
    low: int,
    high: int,
    size: int | None = None,
) -> int | NDArray:
    """Generate random integer(s) in [low, high].

    Args:
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
        size: Number of samples (None for single value).

    Returns:
        Random integer(s).
    """
    rng = get_rng()
    result = rng.integers(low, high + 1, size=size)
    return int(result) if size is None else result


def random_log_uniform(
    low: float,
    high: float,
    size: int | None = None,
) -> float | NDArray:
    """Generate random value(s) from log-uniform distribution.

    Args:
        low: Lower bound (must be positive).
        high: Upper bound (must be positive).
        size: Number of samples (None for single value).

    Returns:
        Random value(s).

    Raises:
        ValueError: If bounds are not positive.
    """
    if low <= 0 or high <= 0:
        msg = "Bounds must be positive for log-uniform distribution"
        raise ValueError(msg)

    rng = get_rng()
    log_low, log_high = np.log(low), np.log(high)
    result = np.exp(rng.uniform(log_low, log_high, size=size))
    return float(result) if size is None else result


def random_choice(
    n: int,
    size: int | None = None,
    p: NDArray | None = None,
    *,
    replace: bool = True,
) -> int | NDArray:
    """Random choice from range(n) with optional weights.

    Args:
        n: Upper bound (exclusive).
        size: Number of samples.
        p: Probability weights.
        replace: Whether to sample with replacement.

    Returns:
        Random index/indices.
    """
    rng = get_rng()
    result = rng.choice(n, size=size, p=p, replace=replace)
    return int(result) if size is None else result





def mean(values: NDArray) -> float:
    return float(np.mean(values))


def variance(values: NDArray, ddof: int = 0) -> float:
    return float(np.var(values, ddof=ddof))


def std_dev(values: NDArray, ddof: int = 0) -> float:
    return float(np.std(values, ddof=ddof))


def percentile(values: NDArray, q: float) -> float:
    return float(np.percentile(values, q))


def median(values: NDArray) -> float:
    return float(np.median(values))


def iqr(values: NDArray) -> float:
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def describe(values: NDArray) -> dict[str, float]:
    """Compute descriptive statistics

    Args:
        values: Array of values.

    Returns:
        Dictionary with mean, variance, skewness, kurtosis.
    """
    result = scipy_stats.describe(values)
    return {
        "nobs": float(result.nobs),
        "mean": float(result.mean),
        "variance": float(result.variance),
        "skewness": float(result.skewness),
        "kurtosis": float(result.kurtosis),
        "min": float(result.minmax[0]),
        "max": float(result.minmax[1]),
    }





def hash_dict(d: dict[str, Any]) -> str:
    """Compute a deterministic hash for a dictionary.

    Args:
        d: Dictionary to hash.

    Returns:
        Hexadecimal hash string.
    """

    def make_hashable(obj: Any) -> Hashable:
        if isinstance(obj, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
        if isinstance(obj, list):
            return tuple(make_hashable(x) for x in obj)
        if isinstance(obj, np.ndarray):
            return tuple(obj.flatten().tolist())
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    hashable = make_hashable(d)
    json_str = json.dumps(hashable, sort_keys=True, default=str)
    return hashlib.md5(json_str.encode(), usedforsecurity=False).hexdigest()




def euclidean_distance(a: NDArray, b: NDArray) -> float:
    """Compute Euclidean distance between two vectors."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def manhattan_distance(a: NDArray, b: NDArray) -> float:
    """Compute Manhattan distance between two vectors."""
    return float(np.sum(np.abs(np.asarray(a) - np.asarray(b))))


def cosine_distance(a: NDArray, b: NDArray) -> float:
    """Compute cosine distance between two vectors."""
    a, b = np.asarray(a), np.asarray(b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def pairwise_distances(x: NDArray, metric: str = "euclidean") -> NDArray:
    """Compute pairwise distance matrix

    Args:
        x: 2D array of shape (n_samples, n_features).
        metric: Distance metric ('euclidean', 'manhattan', 'cosine').

    Returns:
        Pairwise distance matrix of shape (n_samples, n_samples).
    """
    return cdist(x, x, metric=metric)  # pyright: ignore[reportCallIssue, reportArgumentType]


def normalized_euclidean_distance(
    a: NDArray,
    b: NDArray,
    ranges: NDArray,
) -> float:
    """Compute normalized Euclidean distance.

    Args:
        a: First vector.
        b: Second vector.
        ranges: Range sizes for each dimension.

    Returns:
        Normalized distance.
    """
    a, b, ranges = np.asarray(a), np.asarray(b), np.asarray(ranges)
    safe_ranges = np.where(ranges == 0, 1, ranges)
    normalized_diff = (a - b) / safe_ranges
    return float(np.linalg.norm(normalized_diff))




def softmax(x: NDArray, temperature: float = 1.0) -> NDArray:
    """Compute softmax probabilities with temperature scaling.

    Args:
        x: Input logits.
        temperature: Temperature parameter (higher = more uniform).

    Returns:
        Softmax probabilities.
    """
    x = np.asarray(x, dtype=np.float64)
    scaled = x / temperature
    # Subtract max for numerical stability
    shifted = scaled - np.max(scaled)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x)


def weighted_sample_indices(
    weights: NDArray,
    n: int,
    *,
    replace: bool = True,
) -> NDArray:
    """Sample indices according to weights.

    Args:
        weights: Probability weights (will be normalized).
        n: Number of samples.
        replace: Whether to sample with replacement.

    Returns:
        Array of sampled indices.
    """
    rng = get_rng()
    weights = np.asarray(weights, dtype=np.float64)
    probs = weights / np.sum(weights)
    return rng.choice(len(weights), size=n, p=probs, replace=replace)


def argsort_descending(x: NDArray) -> NDArray:
    """Return indices that would sort array in descending order.

    Args:
        x: Input array.

    Returns:
        Indices for descending sort.
    """
    return np.argsort(-np.asarray(x))


def rank_array(x: NDArray, method: str = "average") -> NDArray:
    """Compute ranks of array elements.

    Args:
        x: Input array.
        method: Ranking method ('average', 'min', 'max', 'dense', 'ordinal').

    Returns:
        Array of ranks (1-based).
    """
    return scipy_stats.rankdata(x, method=method)