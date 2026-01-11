
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .backend import (
    ArrayLike,
    DeviceType,
    get_backend,
    is_gpu_available,
    to_numpy,
)

if TYPE_CHECKING:
    from collections.abc import Sequence




@dataclass
class DeviceInfo:
    """Information about where data is stored."""

    device_type: DeviceType
    device_id: int = 0
    backend_name: str = "numpy"

    @classmethod
    def cpu(cls) -> DeviceInfo:
        """Create CPU device info."""
        return cls(DeviceType.CPU, 0, "numpy")

    @classmethod
    def gpu(cls, device_id: int = 0) -> DeviceInfo:
        """Create GPU device info."""
        return cls(DeviceType.GPU, device_id, "cupy")

    @property
    def is_gpu(self) -> bool:
        """Check if on GPU."""
        return self.device_type == DeviceType.GPU

    @property
    def is_cpu(self) -> bool:
        """Check if on CPU."""
        return self.device_type == DeviceType.CPU




@dataclass
class FeatureMask:
    """Boolean mask for feature selection.

    Handles feature selection operations on CPU or GPU.

    Attributes:
        mask: Boolean array indicating selected features.
        device_info: Device where mask is stored.
        feature_names: Optional names for features.
    """

    mask: ArrayLike
    device_info: DeviceInfo = field(default_factory=DeviceInfo.cpu)
    feature_names: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate mask."""
        if self.mask.ndim != 1:
            msg = f"Mask must be 1D, got {self.mask.ndim}D"
            raise ValueError(msg)

    @property
    def n_features(self) -> int:
        """Total number of features."""
        return len(self.mask)

    @property
    def n_selected(self) -> int:
        """Number of selected features."""
        return int(np.sum(to_numpy(self.mask)))

    @property
    def selected_indices(self) -> NDArray:
        """Indices of selected features."""
        return np.where(to_numpy(self.mask))[0]

    @property
    def selected_names(self) -> list[str] | None:
        """Names of selected features."""
        if self.feature_names is None:
            return None
        indices = self.selected_indices
        return [self.feature_names[i] for i in indices]

    def to_cpu(self) -> FeatureMask:
        """Transfer mask to CPU."""
        if self.device_info.is_cpu:
            return self
        return FeatureMask(
            mask=to_numpy(self.mask),
            device_info=DeviceInfo.cpu(),
            feature_names=self.feature_names,
        )

    def to_gpu(self) -> FeatureMask:
        """Transfer mask to GPU."""
        if not is_gpu_available():
            msg = "GPU not available"
            raise RuntimeError(msg)

        if self.device_info.is_gpu:
            return self

        backend = get_backend()
        return FeatureMask(
            mask=backend.asarray(self.mask),
            device_info=DeviceInfo.gpu(),
            feature_names=self.feature_names,
        )

    def to_device(self, device_info: DeviceInfo) -> FeatureMask:
        """Transfer to specified device."""
        if device_info.is_gpu:
            return self.to_gpu()
        return self.to_cpu()

    def copy(self) -> FeatureMask:
        """Create a copy of the mask."""
        if self.device_info.is_gpu:
            backend = get_backend()
            mask_copy = backend.copy(self.mask)
        else:
            mask_copy = np.copy(self.mask)

        return FeatureMask(
            mask=mask_copy,
            device_info=self.device_info,
            feature_names=copy.copy(self.feature_names) if self.feature_names else None,
        )

    @classmethod
    def all_selected(
        cls,
        n_features: int,
        feature_names: list[str] | None = None,
        device_info: DeviceInfo | None = None,
    ) -> FeatureMask:
        """Create mask with all features selected."""
        device_info = device_info or DeviceInfo.cpu()
        if device_info.is_gpu and is_gpu_available():
            backend = get_backend()
            mask = backend.ones(n_features, dtype="bool")
        else:
            mask = np.ones(n_features, dtype=bool)
            device_info = DeviceInfo.cpu()

        return cls(mask=mask, device_info=device_info, feature_names=feature_names)

    @classmethod
    def none_selected(
        cls,
        n_features: int,
        feature_names: list[str] | None = None,
        device_info: DeviceInfo | None = None,
    ) -> FeatureMask:
        """Create mask with no features selected."""
        device_info = device_info or DeviceInfo.cpu()
        if device_info.is_gpu and is_gpu_available():
            backend = get_backend()
            mask = backend.zeros(n_features, dtype="bool")
        else:
            mask = np.zeros(n_features, dtype=bool)
            device_info = DeviceInfo.cpu()

        return cls(mask=mask, device_info=device_info, feature_names=feature_names)

    @classmethod
    def from_indices(
        cls,
        indices: Sequence[int],
        n_features: int,
        feature_names: list[str] | None = None,
        device_info: DeviceInfo | None = None,
    ) -> FeatureMask:
        """Create mask from selected indices."""
        mask = np.zeros(n_features, dtype=bool)
        for idx in indices:
            if 0 <= idx < n_features:
                mask[idx] = True

        result = cls(mask=mask, feature_names=feature_names)
        if device_info is not None and device_info.is_gpu:
            return result.to_gpu()
        return result

    def __and__(self, other: FeatureMask) -> FeatureMask:
        """Logical AND of two masks."""
        if self.device_info.is_gpu or other.device_info.is_gpu:
            backend = get_backend()
            mask1 = backend.asarray(self.mask)
            mask2 = backend.asarray(other.mask)
            result = backend.logical_and(mask1, mask2)
            return FeatureMask(mask=result, device_info=DeviceInfo.gpu())

        result = np.logical_and(self.mask, other.mask)
        return FeatureMask(mask=result, feature_names=self.feature_names)

    def __or__(self, other: FeatureMask) -> FeatureMask:
        """Logical OR of two masks."""
        if self.device_info.is_gpu or other.device_info.is_gpu:
            backend = get_backend()
            mask1 = backend.asarray(self.mask)
            mask2 = backend.asarray(other.mask)
            result = backend.logical_or(mask1, mask2)
            return FeatureMask(mask=result, device_info=DeviceInfo.gpu())

        result = np.logical_or(self.mask, other.mask)
        return FeatureMask(mask=result, feature_names=self.feature_names)

    def __invert__(self) -> FeatureMask:
        """Logical NOT (invert selection)."""
        if self.device_info.is_gpu:
            backend = get_backend()
            result = backend.logical_not(self.mask)
            return FeatureMask(
                mask=result,
                device_info=self.device_info,
                feature_names=self.feature_names,
            )

        result = np.logical_not(self.mask)
        return FeatureMask(mask=result, feature_names=self.feature_names)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FeatureMask(n_features={self.n_features}, "
            f"n_selected={self.n_selected}, "
            f"device={self.device_info.backend_name})"
        )


@dataclass
class DataBundle:
    """Container for feature matrix and target vector.


    Attributes:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target vector of shape (n_samples,).
        feature_names: Optional list of feature names.
        sample_weights: Optional sample weights.
        groups: Optional group labels for grouped CV.
        device_info: Device where data is stored.
        metadata: Additional metadata dictionary.
    """

    X: ArrayLike
    y: ArrayLike
    feature_names: list[str] | None = None
    sample_weights: ArrayLike | None = None
    groups: ArrayLike | None = None
    device_info: DeviceInfo = field(default_factory=DeviceInfo.cpu)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate data bundle."""
        if self.X.ndim != 2:
            msg = f"X must be 2D, got {self.X.ndim}D"
            raise ValueError(msg)

        if self.y.ndim != 1:
            msg = f"y must be 1D, got {self.y.ndim}D"
            raise ValueError(msg)

        if self.X.shape[0] != self.y.shape[0]:
            msg = f"X and y must have same number of samples: {self.X.shape[0]} vs {self.y.shape[0]}"
            raise ValueError(msg)

        if self.feature_names is not None and len(self.feature_names) != self.X.shape[1]:
            msg = f"feature_names length ({len(self.feature_names)}) must match n_features ({self.X.shape[1]})"
            raise ValueError(msg)

        if self.sample_weights is not None and len(self.sample_weights) != self.n_samples:
            msg = "sample_weights length must match n_samples"
            raise ValueError(msg)

        if self.groups is not None and len(self.groups) != self.n_samples:
            msg = "groups length must match n_samples"
            raise ValueError(msg)

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.X.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of feature matrix (n_samples, n_features)."""
        return (self.n_samples, self.n_features)

    @property
    def dtype(self) -> np.dtype:
        """Data type of feature matrix."""
        return self.X.dtype

    @property
    def is_gpu(self) -> bool:
        """Check if data is on GPU."""
        return self.device_info.is_gpu

    @property
    def is_cpu(self) -> bool:
        """Check if data is on CPU."""
        return self.device_info.is_cpu

    def to_cpu(self) -> DataBundle:
        """Transfer all data to CPU.

        Returns:
            New DataBundle with all data on CPU.
        """
        if self.is_cpu:
            return self

        return DataBundle(
            X=to_numpy(self.X),
            y=to_numpy(self.y),
            feature_names=self.feature_names,
            sample_weights=to_numpy(self.sample_weights) if self.sample_weights is not None else None,
            groups=to_numpy(self.groups) if self.groups is not None else None,
            device_info=DeviceInfo.cpu(),
            metadata=copy.copy(self.metadata),
        )

    def to_gpu(self, device_id: int = 0) -> DataBundle:
        """Transfer all data to GPU.

        Args:
            device_id: CUDA device ID.

        Returns:
            New DataBundle with all data on GPU.

        Raises:
            RuntimeError: If GPU is not available.
        """
        if not is_gpu_available():
            msg = "GPU not available"
            raise RuntimeError(msg)

        if self.is_gpu:
            return self

        backend = get_backend()

        return DataBundle(
            X=backend.asarray(self.X),
            y=backend.asarray(self.y),
            feature_names=self.feature_names,
            sample_weights=backend.asarray(self.sample_weights) if self.sample_weights is not None else None,
            groups=backend.asarray(self.groups) if self.groups is not None else None,
            device_info=DeviceInfo.gpu(device_id),
            metadata=copy.copy(self.metadata),
        )

    def to_device(self, device_info: DeviceInfo) -> DataBundle:
        """Transfer to specified device.

        Args:
            device_info: Target device information.

        Returns:
            DataBundle on the target device.
        """
        if device_info.is_gpu:
            return self.to_gpu(device_info.device_id)
        return self.to_cpu()

    def get_subset(
        self,
        sample_indices: ArrayLike | None = None,
        feature_indices: ArrayLike | None = None,
        feature_mask: FeatureMask | None = None,
    ) -> DataBundle:
        """Get a subset of the data.

        Args:
            sample_indices: Indices of samples to select.
            feature_indices: Indices of features to select.
            feature_mask: Boolean mask for feature selection.

        Returns:
            New DataBundle with selected data.
        """
        X_subset = self.X
        y_subset = self.y
        weights_subset = self.sample_weights
        groups_subset = self.groups
        new_feature_names = self.feature_names

        # Apply sample selection
        if sample_indices is not None:
            if self.is_gpu:
                backend = get_backend()
                indices = backend.asarray(sample_indices)
            else:
                indices = np.asarray(sample_indices)

            X_subset = X_subset[indices]
            y_subset = y_subset[indices]
            if weights_subset is not None:
                weights_subset = weights_subset[indices]
            if groups_subset is not None:
                groups_subset = groups_subset[indices]

        # Apply feature selection
        if feature_mask is not None:
            # Ensure mask is on same device
            if self.is_gpu:
                mask = feature_mask.to_gpu().mask
            else:
                mask = feature_mask.to_cpu().mask
            X_subset = X_subset[:, mask]
            if self.feature_names is not None:
                new_feature_names = feature_mask.selected_names
        elif feature_indices is not None:
            if self.is_gpu:
                backend = get_backend()
                f_indices = backend.asarray(feature_indices)
            else:
                f_indices = np.asarray(feature_indices)
            X_subset = X_subset[:, f_indices]
            if self.feature_names is not None:
                new_feature_names = [self.feature_names[i] for i in to_numpy(f_indices)]

        return DataBundle(
            X=X_subset,
            y=y_subset,
            feature_names=new_feature_names,
            sample_weights=weights_subset,
            groups=groups_subset,
            device_info=self.device_info,
            metadata=copy.copy(self.metadata),
        )

    def apply_feature_mask(self, mask: FeatureMask) -> DataBundle:
        """Apply feature mask to select features.

        Args:
            mask: Boolean feature mask.

        Returns:
            New DataBundle with selected features.
        """
        return self.get_subset(feature_mask=mask)

    def get_train_val_split(
        self,
        train_indices: ArrayLike,
        val_indices: ArrayLike,
    ) -> tuple[DataBundle, DataBundle]:
        """Split data into training and validation sets.

        Args:
            train_indices: Indices for training set.
            val_indices: Indices for validation set.

        Returns:
            Tuple of (train_bundle, val_bundle).
        """
        train_bundle = self.get_subset(sample_indices=train_indices)
        val_bundle = self.get_subset(sample_indices=val_indices)
        return train_bundle, val_bundle

    def copy(self) -> DataBundle:
        """Create a deep copy of the data bundle."""
        if self.is_gpu:
            backend = get_backend()
            return DataBundle(
                X=backend.copy(self.X),
                y=backend.copy(self.y),
                feature_names=copy.copy(self.feature_names) if self.feature_names else None,
                sample_weights=backend.copy(self.sample_weights) if self.sample_weights is not None else None,
                groups=backend.copy(self.groups) if self.groups is not None else None,
                device_info=self.device_info,
                metadata=copy.deepcopy(self.metadata),
            )

        return DataBundle(
            X=np.copy(self.X),
            y=np.copy(self.y),
            feature_names=copy.copy(self.feature_names) if self.feature_names else None,
            sample_weights=np.copy(self.sample_weights) if self.sample_weights is not None else None,
            groups=np.copy(self.groups) if self.groups is not None else None,
            device_info=self.device_info,
            metadata=copy.deepcopy(self.metadata),
        )

    @classmethod
    def from_arrays(
        cls,
        X: ArrayLike,
        y: ArrayLike,
        feature_names: list[str] | None = None,
        use_gpu: bool = False,
    ) -> DataBundle:
        """Create DataBundle from arrays.

        Args:
            X: Feature matrix.
            y: Target vector.
            feature_names: Optional feature names.
            use_gpu: Whether to store on GPU.

        Returns:
            DataBundle instance.
        """
        # Ensure numpy arrays first
        X_np = np.asarray(X)
        y_np = np.asarray(y)

        # Ensure 2D
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        bundle = cls(
            X=X_np,
            y=y_np,
            feature_names=feature_names,
            device_info=DeviceInfo.cpu(),
        )

        if use_gpu:
            return bundle.to_gpu()
        return bundle

    def get_statistics(self) -> dict[str, Any]:
        """Compute basic statistics about the data.

        Returns:
            Dictionary with statistics.
        """
        # Transfer to CPU for statistics if needed
        X_cpu = to_numpy(self.X)
        y_cpu = to_numpy(self.y)

        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "X_mean": float(np.mean(X_cpu)),
            "X_std": float(np.std(X_cpu)),
            "X_min": float(np.min(X_cpu)),
            "X_max": float(np.max(X_cpu)),
            "y_unique": len(np.unique(y_cpu)),
            "y_dtype": str(y_cpu.dtype),
            "device": self.device_info.backend_name,
            "memory_bytes": X_cpu.nbytes + y_cpu.nbytes,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DataBundle(n_samples={self.n_samples}, "
            f"n_features={self.n_features}, "
            f"dtype={self.dtype}, "
            f"device={self.device_info.backend_name})"
        )




@dataclass
class CVFoldData:
    """Data container for a single cross-validation fold.

    Stores train and validation indices rather than copies of data
    for memory efficiency.

    Attributes:
        train_indices: Indices for training samples.
        val_indices: Indices for validation samples.
        fold_id: Identifier for this fold.
        device_info: Device where indices are stored.
    """

    train_indices: ArrayLike
    val_indices: ArrayLike
    fold_id: int
    device_info: DeviceInfo = field(default_factory=DeviceInfo.cpu)

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return len(self.train_indices)

    @property
    def n_val(self) -> int:
        """Number of validation samples."""
        return len(self.val_indices)

    def to_cpu(self) -> CVFoldData:
        """Transfer indices to CPU."""
        if self.device_info.is_cpu:
            return self

        return CVFoldData(
            train_indices=to_numpy(self.train_indices),
            val_indices=to_numpy(self.val_indices),
            fold_id=self.fold_id,
            device_info=DeviceInfo.cpu(),
        )

    def to_gpu(self) -> CVFoldData:
        """Transfer indices to GPU."""
        if not is_gpu_available():
            msg = "GPU not available"
            raise RuntimeError(msg)

        if self.device_info.is_gpu:
            return self

        backend = get_backend()
        return CVFoldData(
            train_indices=backend.asarray(self.train_indices),
            val_indices=backend.asarray(self.val_indices),
            fold_id=self.fold_id,
            device_info=DeviceInfo.gpu(),
        )

    def get_train_val_data(self, data: DataBundle) -> tuple[DataBundle, DataBundle]:
        """Extract train and validation data from a DataBundle.

        Args:
            data: Full dataset.

        Returns:
            Tuple of (train_data, val_data).
        """
        # Ensure indices are on same device as data
        if data.is_gpu:
            fold = self.to_gpu()
        else:
            fold = self.to_cpu()

        return data.get_train_val_split(fold.train_indices, fold.val_indices)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CVFoldData(fold_id={self.fold_id}, "
            f"n_train={self.n_train}, "
            f"n_val={self.n_val})"
        )



@dataclass
class CVSplits:
    """Container for all cross-validation fold data.

    Attributes:
        folds: List of CVFoldData for each fold.
        n_folds: Number of folds.
        stratified: Whether stratified splitting was used.
        shuffle: Whether data was shuffled.
        random_state: Random state used for splitting.
    """

    folds: list[CVFoldData]
    n_folds: int
    stratified: bool = False
    shuffle: bool = True
    random_state: int | None = None

    def __post_init__(self) -> None:
        """Validate splits."""
        if len(self.folds) != self.n_folds:
            msg = f"Number of folds ({len(self.folds)}) must match n_folds ({self.n_folds})"
            raise ValueError(msg)

    def to_cpu(self) -> CVSplits:
        """Transfer all fold indices to CPU."""
        return CVSplits(
            folds=[f.to_cpu() for f in self.folds],
            n_folds=self.n_folds,
            stratified=self.stratified,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

    def to_gpu(self) -> CVSplits:
        """Transfer all fold indices to GPU."""
        return CVSplits(
            folds=[f.to_gpu() for f in self.folds],
            n_folds=self.n_folds,
            stratified=self.stratified,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

    def __iter__(self):
        """Iterate over folds."""
        return iter(self.folds)

    def __len__(self) -> int:
        """Number of folds."""
        return self.n_folds

    def __getitem__(self, idx: int) -> CVFoldData:
        """Get fold by index."""
        return self.folds[idx]

    @classmethod
    def from_sklearn_splitter(
        cls,
        splitter: Any,
        X: ArrayLike,
        y: ArrayLike,
        groups: ArrayLike | None = None,
    ) -> CVSplits:
        """Create CVSplits from sklearn splitter.

        Args:
            splitter: sklearn splitter object (KFold, StratifiedKFold, etc.).
            X: Feature matrix.
            y: Target vector.
            groups: Optional group labels.

        Returns:
            CVSplits instance.
        """
        X_cpu = to_numpy(X)
        y_cpu = to_numpy(y)
        groups_cpu = to_numpy(groups) if groups is not None else None

        folds = []
        for fold_id, (train_idx, val_idx) in enumerate(splitter.split(X_cpu, y_cpu, groups_cpu)):
            folds.append(
                CVFoldData(
                    train_indices=train_idx.astype(np.int64),
                    val_indices=val_idx.astype(np.int64),
                    fold_id=fold_id,
                )
            )

        # Detect splitter properties
        stratified = hasattr(splitter, "stratify") or "Stratified" in type(splitter).__name__
        shuffle = getattr(splitter, "shuffle", True)
        random_state = getattr(splitter, "random_state", None)

        return cls(
            folds=folds,
            n_folds=len(folds),
            stratified=stratified,
            shuffle=shuffle,
            random_state=random_state,
        )




@dataclass
class PopulationData:
    """Container for GA population arrays.

    Stores population data in array format for efficient GPU operations.

    Attributes:
        feature_genes: Boolean array of shape (pop_size, n_features).
        discrete_genes: Array of shape (pop_size, n_discrete_params).
        fitness: Array of shape (pop_size,).
        needs_evaluation: Boolean array indicating which need evaluation.
        generation: Generation number for each individual.
        device_info: Device where data is stored.
    """

    feature_genes: ArrayLike
    discrete_genes: ArrayLike
    fitness: ArrayLike
    needs_evaluation: ArrayLike
    generation: ArrayLike
    device_info: DeviceInfo = field(default_factory=DeviceInfo.cpu)

    def __post_init__(self) -> None:
        """Validate population data."""
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

    def to_cpu(self) -> PopulationData:
        """Transfer to CPU."""
        if self.device_info.is_cpu:
            return self

        return PopulationData(
            feature_genes=to_numpy(self.feature_genes),
            discrete_genes=to_numpy(self.discrete_genes),
            fitness=to_numpy(self.fitness),
            needs_evaluation=to_numpy(self.needs_evaluation),
            generation=to_numpy(self.generation),
            device_info=DeviceInfo.cpu(),
        )

    def to_gpu(self) -> PopulationData:
        """Transfer to GPU."""
        if not is_gpu_available():
            msg = "GPU not available"
            raise RuntimeError(msg)

        if self.device_info.is_gpu:
            return self

        backend = get_backend()
        return PopulationData(
            feature_genes=backend.asarray(self.feature_genes),
            discrete_genes=backend.asarray(self.discrete_genes),
            fitness=backend.asarray(self.fitness),
            needs_evaluation=backend.asarray(self.needs_evaluation),
            generation=backend.asarray(self.generation),
            device_info=DeviceInfo.gpu(),
        )

    @classmethod
    def create_empty(
        cls,
        pop_size: int,
        n_features: int,
        n_discrete: int,
        use_gpu: bool = False,
        dtype: str = "float64",
    ) -> PopulationData:
        """Create empty population data.

        Args:
            pop_size: Population size.
            n_features: Number of features.
            n_discrete: Number of discrete parameters.
            use_gpu: Whether to use GPU.
            dtype: Data type for fitness array.

        Returns:
            Empty PopulationData.
        """
        if use_gpu and is_gpu_available():
            backend = get_backend()
            return cls(
                feature_genes=backend.zeros((pop_size, n_features), dtype="bool"),
                discrete_genes=backend.zeros((pop_size, n_discrete), dtype="int64"),
                fitness=backend.full((pop_size,), float("-inf"), dtype=dtype),  # pyright: ignore[reportArgumentType]
                needs_evaluation=backend.ones((pop_size,), dtype="bool"),
                generation=backend.zeros((pop_size,), dtype="int32"),
                device_info=DeviceInfo.gpu(),
            )

        return cls(
            feature_genes=np.zeros((pop_size, n_features), dtype=bool),
            discrete_genes=np.zeros((pop_size, n_discrete), dtype=np.int64),
            fitness=np.full(pop_size, -np.inf, dtype=dtype),
            needs_evaluation=np.ones(pop_size, dtype=bool),
            generation=np.zeros(pop_size, dtype=np.int32),
            device_info=DeviceInfo.cpu(),
        )



@dataclass
class SwarmData:
    """Container for PSO swarm arrays.

    Stores swarm data in array format for efficient GPU operations.

    Attributes:
        positions: Array of shape (swarm_size, n_continuous).
        velocities: Array of shape (swarm_size, n_continuous).
        personal_best_positions: Array of shape (swarm_size, n_continuous).
        personal_best_fitness: Array of shape (swarm_size,).
        fitness: Array of shape (swarm_size,).
        global_best_position: Array of shape (n_continuous,).
        global_best_fitness: Scalar value.
        device_info: Device where data is stored.
    """

    positions: ArrayLike
    velocities: ArrayLike
    personal_best_positions: ArrayLike
    personal_best_fitness: ArrayLike
    fitness: ArrayLike
    global_best_position: ArrayLike
    global_best_fitness: float
    device_info: DeviceInfo = field(default_factory=DeviceInfo.cpu)

    def __post_init__(self) -> None:
        """Validate swarm data."""
        if self.positions.ndim != 2:
            msg = f"positions must be 2D, got {self.positions.ndim}D"
            raise ValueError(msg)

    @property
    def swarm_size(self) -> int:
        """Number of particles."""
        return self.positions.shape[0]

    @property
    def n_continuous(self) -> int:
        """Number of continuous parameters."""
        return self.positions.shape[1]

    def to_cpu(self) -> SwarmData:
        """Transfer to CPU."""
        if self.device_info.is_cpu:
            return self

        return SwarmData(
            positions=to_numpy(self.positions),
            velocities=to_numpy(self.velocities),
            personal_best_positions=to_numpy(self.personal_best_positions),
            personal_best_fitness=to_numpy(self.personal_best_fitness),
            fitness=to_numpy(self.fitness),
            global_best_position=to_numpy(self.global_best_position),
            global_best_fitness=self.global_best_fitness,
            device_info=DeviceInfo.cpu(),
        )

    def to_gpu(self) -> SwarmData:
        """Transfer to GPU."""
        if not is_gpu_available():
            msg = "GPU not available"
            raise RuntimeError(msg)

        if self.device_info.is_gpu:
            return self

        backend = get_backend()
        return SwarmData(
            positions=backend.asarray(self.positions),
            velocities=backend.asarray(self.velocities),
            personal_best_positions=backend.asarray(self.personal_best_positions),
            personal_best_fitness=backend.asarray(self.personal_best_fitness),
            fitness=backend.asarray(self.fitness),
            global_best_position=backend.asarray(self.global_best_position),
            global_best_fitness=self.global_best_fitness,
            device_info=DeviceInfo.gpu(),
        )

    @classmethod
    def create_empty(
        cls,
        swarm_size: int,
        n_continuous: int,
        use_gpu: bool = False,
        dtype: str = "float64",
    ) -> SwarmData:
        """Create empty swarm data.

        Args:
            swarm_size: Number of particles.
            n_continuous: Number of continuous parameters.
            use_gpu: Whether to use GPU.
            dtype: Data type for arrays.

        Returns:
            Empty SwarmData.
        """
        if use_gpu and is_gpu_available():
            backend = get_backend()
            return cls(
                positions=backend.zeros((swarm_size, n_continuous), dtype=dtype),  # pyright: ignore[reportArgumentType]
                velocities=backend.zeros((swarm_size, n_continuous), dtype=dtype),  # pyright: ignore[reportArgumentType]
                personal_best_positions=backend.zeros((swarm_size, n_continuous), dtype=dtype),  # pyright: ignore[reportArgumentType]
                personal_best_fitness=backend.full((swarm_size,), float("-inf"), dtype=dtype),  # pyright: ignore[reportArgumentType]
                fitness=backend.full((swarm_size,), float("-inf"), dtype=dtype),  # pyright: ignore[reportArgumentType]
                global_best_position=backend.zeros((n_continuous,), dtype=dtype),  # pyright: ignore[reportArgumentType]
                global_best_fitness=float("-inf"),
                device_info=DeviceInfo.gpu(),
            )

        return cls(
            positions=np.zeros((swarm_size, n_continuous), dtype=dtype),
            velocities=np.zeros((swarm_size, n_continuous), dtype=dtype),
            personal_best_positions=np.zeros((swarm_size, n_continuous), dtype=dtype),
            personal_best_fitness=np.full(swarm_size, -np.inf, dtype=dtype),
            fitness=np.full(swarm_size, -np.inf, dtype=dtype),
            global_best_position=np.zeros(n_continuous, dtype=dtype),
            global_best_fitness=float("-inf"),
            device_info=DeviceInfo.cpu(),
        )
