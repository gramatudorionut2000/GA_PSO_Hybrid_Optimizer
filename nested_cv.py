from __future__ import annotations

import logging
import time

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.model_selection import KFold, StratifiedKFold
from .utils.common import TaskType, ScoringMetric, ensure_numpy
from .utils.scoring import (
    SCORING_FUNCTIONS,
    get_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    needs_probability_predictions,
)
from .hybrid_optimizer import (
    HybridGAPSOOptimizer,
    HybridConfig,
    HybridResult,
)
from .optimization_config import GASettings, PSOSettings
from .evaluation import GPUScorer, EvaluationPipeline, PipelineConfig, BatchEvaluator
from .results import GenerationStats, OptimizationResults, FeatureImportance
from .utils.performance import TransferOptimizer, create_transfer_optimizer


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

ArrayType = TypeVar("ArrayType", bound=np.ndarray)




@dataclass
class CVSplit:
    """Single cross-validation split.

    Attributes:
        fold: Fold index.
        train_idx: Training indices.
        val_idx: Validation indices.
    """

    fold: int
    train_idx: NDArray[np.intp]
    val_idx: NDArray[np.intp]

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return len(self.train_idx)

    @property
    def n_val(self) -> int:
        """Number of validation samples."""
        return len(self.val_idx)


class CVSplitter:
    """Cross-validation splitter with GPU-friendly index management.

    Generates and caches CV split indices for efficient reuse during
    optimization iterations.
    """

    def __init__(
        self,
        n_splits: int = 5,
        stratified: bool = True,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initialize CV splitter.

        Args:
            n_splits: Number of CV folds.
            stratified: Use stratified splitting.
            shuffle: Shuffle data before splitting.
            random_state: Random seed for reproducibility.
        """
        self.n_splits = n_splits
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state

        self._cached_splits: list[CVSplit] | None = None
        self._cached_n_samples: int | None = None

    def get_splits(
        self,
        n_samples: int,
        y: NDArray | None = None,
    ) -> list[CVSplit]:
        """Get CV splits.

        Args:
            n_samples: Number of samples.
            y: Target array (required for stratified splitting).

        Returns:
            List of CVSplit objects.
        """
        if (
            self._cached_splits is not None
            and self._cached_n_samples == n_samples
        ):
            return self._cached_splits

        if self.stratified and y is not None:
            splitter: KFold | StratifiedKFold = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        else:
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )

        X_dummy = np.arange(n_samples).reshape(-1, 1)
        y_for_split = y if y is not None else np.zeros(n_samples)

        splits = []
        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(X_dummy, y_for_split)
        ):
            splits.append(
                CVSplit(
                    fold=fold,
                    train_idx=train_idx.astype(np.intp),
                    val_idx=val_idx.astype(np.intp),
                )
            )

        self._cached_splits = splits
        self._cached_n_samples = n_samples

        return splits

    def iter_splits(
        self,
        n_samples: int,
        y: NDArray | None = None,
    ) -> Iterator[CVSplit]:
        """Iterate over CV splits.

        Args:
            n_samples: Number of samples.
            y: Target array (required for stratified splitting).

        Yields:
            CVSplit objects.
        """
        yield from self.get_splits(n_samples, y)



class CVDataManager:
    """Manages data for cross-validation with GPU support.

    Implements memory-efficient CV by using index arrays instead of copying
    data for each fold. Supports both CPU and GPU data with optimized transfers.
    """

    def __init__(
        self,
        X: NDArray[np.floating],
        y: NDArray,
        use_gpu: bool = False,
        cache_folds: bool = True,
    ) -> None:
        """Initialize CV data manager.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target array of shape (n_samples,).
            use_gpu: Whether to store data on GPU.
            cache_folds: Cache fold subsets for reuse.
        """
        self.use_gpu = use_gpu
        self.cache_folds = cache_folds

        self._X_cpu = np.asarray(X)
        self._y_cpu = np.asarray(y)

        self._X_gpu: Any = None
        self._y_gpu: Any = None
        self._backend: Any = None
        
        self._transfer_optimizer: TransferOptimizer | None = None

        self._fold_cache: dict[tuple[int, str], tuple[Any, Any, Any, Any]] = {}

        if use_gpu:
            self._init_gpu()

    def _init_gpu(self) -> None:
        """Initialize GPU arrays using TransferOptimizer."""
        try:
            import cupy as cp

            self._backend = cp
            
            self._transfer_optimizer = create_transfer_optimizer()
            
            if self._transfer_optimizer is not None:
                try:
                    result_x = self._transfer_optimizer.to_gpu_async(self._X_cpu)
                    result_y = self._transfer_optimizer.to_gpu_async(self._y_cpu)
                    
                    if isinstance(result_x, tuple):
                        self._X_gpu = result_x[0]
                    else:
                        self._X_gpu = result_x
                        
                    if isinstance(result_y, tuple):
                        self._y_gpu = result_y[0]
                    else:
                        self._y_gpu = result_y
                    
                    self._transfer_optimizer.synchronize()
                except Exception:
                    self._X_gpu = cp.asarray(self._X_cpu)
                    self._y_gpu = cp.asarray(self._y_cpu)
            else:
                self._X_gpu = cp.asarray(self._X_cpu)
                self._y_gpu = cp.asarray(self._y_cpu)
            
            if isinstance(self._X_gpu, tuple):
                self._X_gpu = self._X_gpu[0] if self._X_gpu else cp.asarray(self._X_cpu)
            if isinstance(self._y_gpu, tuple):
                self._y_gpu = self._y_gpu[0] if self._y_gpu else cp.asarray(self._y_cpu)
                
            logger.debug("GPU data initialized: X=%s, y=%s", self._X_gpu.shape, self._y_gpu.shape)
        except ImportError:
            logger.warning("CuPy not available, falling back to CPU")
            self.use_gpu = False
            
    def to_gpu(self, arr: NDArray) -> Any:
        """Transfer array to GPU using optimized transfer.
        
        Args:
            arr: NumPy array to transfer.
            
        Returns:
            GPU array.
        """
        if self._transfer_optimizer is not None:
            try:
                result = self._transfer_optimizer.to_gpu_async(arr)
                self._transfer_optimizer.synchronize()
                if isinstance(result, tuple):
                    return result[0]
                return result
            except Exception:
                pass
        
        if self._backend is not None:
            return self._backend.asarray(arr)
        return arr
    
    def to_cpu(self, arr: Any) -> NDArray:
        """Transfer array to CPU using optimized transfer.
        
        Args:
            arr: GPU array to transfer.
            
        Returns:
            NumPy array.
        """
        if self._transfer_optimizer is not None:
            try:
                result = self._transfer_optimizer.to_cpu_async(arr)
                self._transfer_optimizer.synchronize()
                if isinstance(result, tuple):
                    return result[0]
                return result
            except Exception:
                pass
        
        if hasattr(arr, "get"):
            return arr.get()
        return np.asarray(arr)

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self._X_cpu.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self._X_cpu.shape[1]

    @property
    def X(self) -> NDArray[np.floating]:
        """Get feature matrix (CPU)."""
        return self._X_cpu

    @property
    def y(self) -> NDArray:
        """Get target array (CPU)."""
        return self._y_cpu

    @property
    def X_gpu(self) -> Any | None:
        """Get feature matrix (GPU, if available)."""
        return self._X_gpu

    @property
    def y_gpu(self) -> Any | None:
        """Get target array (GPU, if available)."""
        return self._y_gpu

    def _compute_feature_hash(self, feature_mask: NDArray[np.bool_] | None) -> str:
        """Compute hash for feature mask."""
        if feature_mask is None:
            return "all"
        return str(hash(feature_mask.tobytes()))[:16]

    def get_fold_data(
        self,
        split: CVSplit,
        feature_mask: NDArray[np.bool_] | None = None,
        prefer_gpu: bool = True,
    ) -> tuple[Any, Any, Any, Any]:
        """Get data for a CV fold with optional feature selection.

        Uses fancy indexing to avoid copying full arrays.

        Args:
            split: CV split containing train/val indices.
            feature_mask: Optional boolean mask for feature selection.
            prefer_gpu: Prefer GPU arrays if available.

        Returns:
            Tuple of (X_train, X_val, y_train, y_val).
        """
        cache_key = (split.fold, self._compute_feature_hash(feature_mask))
        if self.cache_folds and cache_key in self._fold_cache:
            return self._fold_cache[cache_key]

        if prefer_gpu and self.use_gpu and self._X_gpu is not None:
            X = self._X_gpu
            y = self._y_gpu
            cp = self._backend

            train_idx = cp.asarray(split.train_idx)
            val_idx = cp.asarray(split.val_idx)

            if feature_mask is not None:
                feature_mask_gpu = cp.asarray(feature_mask)
                X_train = X[train_idx][:, feature_mask_gpu]
                X_val = X[val_idx][:, feature_mask_gpu]
            else:
                X_train = X[train_idx]
                X_val = X[val_idx]

            y_train = y[train_idx]
            y_val = y[val_idx]
        else:
            X = self._X_cpu
            y = self._y_cpu

            if feature_mask is not None:
                X_train = X[split.train_idx][:, feature_mask]
                X_val = X[split.val_idx][:, feature_mask]
            else:
                X_train = X[split.train_idx]
                X_val = X[split.val_idx]

            y_train = y[split.train_idx]
            y_val = y[split.val_idx]

        result = (X_train, X_val, y_train, y_val)

        if self.cache_folds:
            self._fold_cache[cache_key] = result

        return result

    def clear_cache(self) -> None:
        """Clear the fold cache."""
        self._fold_cache.clear()



class ModelProtocol(Protocol):

    def fit(self, X: Any, y: Any) -> Any:
        ...

    def predict(self, X: Any) -> Any:
        ...


class ModelWithProba(ModelProtocol, Protocol):
    """Protocol for models with probability prediction."""

    def predict_proba(self, X: Any) -> Any:
        """Predict class probabilities."""
        ...


@dataclass
class ModelConfig:
    """Configuration for a machine learning model.

    Attributes:
        model_class: Model class (sklearn or cuML).
        cuml_class: Optional cuML equivalent class.
        fixed_params: Fixed model parameters.
        discrete_params: Discrete parameter definitions.
        continuous_params: Continuous parameter definitions.
        task_type: Type of ML task.
        name: Model name.
        supports_gpu: Whether model supports GPU data.
    """

    model_class: type
    cuml_class: type | None = None
    fixed_params: dict[str, Any] = field(default_factory=dict)
    discrete_params: dict[str, Any] = field(default_factory=dict)
    continuous_params: dict[str, Any] = field(default_factory=dict)
    task_type: TaskType = TaskType.CLASSIFICATION
    name: str = "model"
    supports_gpu: bool = False


class ModelWrapper:
    """Wrapper for ML models with parameter management.

    Provides a unified interface for sklearn and cuML models.
    """

    def __init__(
        self,
        config: ModelConfig,
        use_gpu: bool = False,
    ) -> None:
        """Initialize model wrapper.

        Args:
            config: Model configuration.
            use_gpu: Whether to use GPU model if available.
        """
        self.config = config
        self.use_gpu = use_gpu and config.cuml_class is not None
        self._model: Any = None

    def build_model(
        self,
        discrete_values: dict[str, Any] | None = None,
        continuous_values: dict[str, Any] | None = None,
    ) -> Any:
        """Build model with specified parameters.

        Args:
            discrete_values: Discrete parameter values.
            continuous_values: Continuous parameter values.

        Returns:
            Instantiated model.
        """
        params = dict(self.config.fixed_params)

        if discrete_values:
            params.update(discrete_values)

        if continuous_values:
            params.update(continuous_values)

        model_class = (
            self.config.cuml_class
            if self.use_gpu and self.config.cuml_class is not None
            else self.config.model_class
        )

        logger.debug(
            "Building model %s with params: %s",
            model_class.__name__ if model_class else "None",
            params
        )

        try:
            self._model = model_class(**params)
        except Exception as e:
            logger.error(
                "Failed to build model %s with params %s: %s",
                model_class.__name__ if model_class else "None",
                params,
                e
            )
            raise
        return self._model

    def fit(self, X: Any, y: Any) -> Any:
        """Fit the model.

        Args:
            X: Feature matrix.
            y: Target array.

        Returns:
            Fitted model.
        """
        if self._model is None:
            self.build_model()
        

        if hasattr(X, "get"):  # CuPy array
            X = X.get()
        if hasattr(y, "get"):  # CuPy array
            y = y.get()
        
        return self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        """Make predictions.

        Args:
            X: Feature matrix.

        Returns:
            Predictions.
        """
        if self._model is None:
            msg = "Model not built or fitted"
            raise RuntimeError(msg)
        
        if hasattr(X, "get"):  # CuPy array
            X = X.get()
        
        return self._model.predict(X)

    def predict_proba(self, X: Any) -> Any | None:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Class probabilities or None if not supported.
        """
        if self._model is None:
            msg = "Model not built or fitted"
            raise RuntimeError(msg)

        if hasattr(X, "get"):  # CuPy array
            X = X.get()

        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return None



@dataclass
class InnerCVResult:
    """Result of inner CV evaluation.

    Attributes:
        mean_score: Mean CV score across folds.
        std_score: Standard deviation of CV scores.
        fold_scores: Individual fold scores.
        fit_time: Total fitting time.
        score_time: Total scoring time.
    """

    mean_score: float
    std_score: float
    fold_scores: list[float]
    fit_time: float = 0.0
    score_time: float = 0.0


class InnerCVEvaluator:
    """Inner cross-validation evaluator.

    Performs k-fold CV evaluation for a single configuration during
    the optimization process. Uses GPUScorer for GPU-accelerated scoring
    and EvaluationPipeline for structured evaluation when enabled.
    """

    def __init__(
        self,
        data_manager: CVDataManager,
        cv_splitter: CVSplitter,
        scoring: str | ScoringMetric = "accuracy",
        use_gpu: bool = False,
        n_jobs: int = 1,
        use_pipeline: bool = True,
    ) -> None:
        """Initialize inner CV evaluator.

        Args:
            data_manager: CV data manager.
            cv_splitter: CV splitter for inner folds.
            scoring: Scoring metric.
            use_gpu: Whether to use GPU for evaluation.
            n_jobs: Number of parallel jobs (CPU only).
            use_pipeline: Whether to use EvaluationPipeline for evaluation.
        """
        self.data_manager = data_manager
        self.cv_splitter = cv_splitter
        self.scoring = scoring
        self.use_gpu = use_gpu
        self.n_jobs = n_jobs
        self.use_pipeline = use_pipeline

        self._gpu_scorer = GPUScorer(use_gpu=use_gpu, min_samples_for_gpu=500)
        
        pipeline_config = PipelineConfig(
            use_gpu=use_gpu,
            scoring=str(scoring),
            min_samples_for_gpu=500,
        )
        self._evaluation_pipeline = EvaluationPipeline(
            config=pipeline_config,
            scorer=self._gpu_scorer,
        )
        
        self._batch_evaluator = BatchEvaluator(
            pipeline=self._evaluation_pipeline,
            config=pipeline_config,
        )
        
        self._scorer = get_scorer(scoring)
        self._needs_proba = needs_probability_predictions(scoring)

        self._splits = cv_splitter.get_splits(
            data_manager.n_samples,
            data_manager.y,
        )

    def _score(
        self,
        y_true: NDArray,
        y_pred: NDArray,
    ) -> float:
        """Score predictions using GPUScorer or fallback.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels or probabilities.
            
        Returns:
            Score value.
        """
        if self.use_gpu:
            try:
                return self._gpu_scorer.score(
                    y_true, y_pred,
                    metric=str(self.scoring),
                )
            except Exception:
                pass
        
        y_true_np = ensure_numpy(y_true)
        y_pred_np = ensure_numpy(y_pred)
        return self._scorer(y_true_np, y_pred_np)
    
    def _evaluate_fold_with_pipeline(
        self,
        model: Any,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        feature_mask: NDArray[np.bool_] | None = None,
    ) -> tuple[float, float, float]:
        """Evaluate a single fold using EvaluationPipeline.
        
        Args:
            model: Model instance.
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            feature_mask: Optional feature selection mask.
            
        Returns:
            Tuple of (score, fit_time, score_time).
        """
        result = self._evaluation_pipeline.evaluate(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_mask=feature_mask,
        )
        
        return result.score, result.fit_time, result.score_time

    def evaluate(
        self,
        model_wrapper: ModelWrapper,
        feature_mask: NDArray[np.bool_] | None = None,
        discrete_params: dict[str, Any] | None = None,
        continuous_params: dict[str, Any] | None = None,
        train_idx: NDArray[np.intp] | None = None,
    ) -> InnerCVResult:
        """Evaluate configuration using inner CV.

        Uses EvaluationPipeline for structured evaluation when use_pipeline=True,
        otherwise falls back to manual evaluation.

        Args:
            model_wrapper: Model wrapper with configuration.
            feature_mask: Optional feature selection mask.
            discrete_params: Discrete parameter values.
            continuous_params: Continuous parameter values.
            train_idx: Optional subset of training indices (for nested CV).

        Returns:
            InnerCVResult with scores and timing.
        """
        fold_scores = []
        total_fit_time = 0.0
        total_score_time = 0.0

        if train_idx is not None:
            n_subset = len(train_idx)
            y_subset = self.data_manager.y[train_idx]
            splits = self.cv_splitter.get_splits(n_subset, y_subset)

            for split in splits:
                split.train_idx = train_idx[split.train_idx]
                split.val_idx = train_idx[split.val_idx]
        else:
            splits = self._splits

        for split in splits:
            X_train, X_val, y_train, y_val = self.data_manager.get_fold_data(
                split,
                feature_mask=feature_mask,
                prefer_gpu=self.use_gpu,
            )

            model_wrapper.build_model(discrete_params, continuous_params)
            
            if self.use_pipeline:
                try:
                    current_model = model_wrapper._model
                    
                    result = self._batch_evaluator.evaluate_cv_fold(
                        model_builder=lambda m=current_model: m,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        feature_mask=None,
                    )
                    fold_scores.append(result.score)
                    total_fit_time += result.fit_time
                    total_score_time += result.score_time
                    continue
                except Exception as e:
                    logger.debug("Pipeline evaluation failed, using fallback: %s", e)

            fit_start = time.perf_counter()
            model_wrapper.fit(X_train, y_train)
            total_fit_time += time.perf_counter() - fit_start

            score_start = time.perf_counter()

            if self._needs_proba:
                y_pred = model_wrapper.predict_proba(X_val)
                if y_pred is None:
                    y_pred = model_wrapper.predict(X_val)
            else:
                y_pred = model_wrapper.predict(X_val)

            score = self._score(y_val, y_pred)
            fold_scores.append(score)
            total_score_time += time.perf_counter() - score_start

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))

        return InnerCVResult(
            mean_score=mean_score,
            std_score=std_score,
            fold_scores=fold_scores,
            fit_time=total_fit_time,
            score_time=total_score_time,
        )

    def evaluate_batch(
        self,
        model_wrapper: ModelWrapper,
        feature_masks: NDArray[np.bool_] | None = None,
        discrete_params_batch: list[dict[str, Any]] | None = None,
        continuous_params_batch: NDArray[np.floating] | None = None,
        continuous_param_names: list[str] | None = None,
        train_idx: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        """Batch evaluate multiple configurations.

        Args:
            model_wrapper: Model wrapper with base configuration.
            feature_masks: Feature masks of shape (batch_size, n_features) or None.
            discrete_params_batch: List of discrete param dicts.
            continuous_params_batch: Continuous params of shape (batch_size, n_continuous).
            continuous_param_names: Names of continuous parameters.
            train_idx: Optional subset of training indices.

        Returns:
            Array of mean CV scores for each configuration.
        """
        if continuous_params_batch is not None:
            batch_size = len(continuous_params_batch)
        elif discrete_params_batch is not None:
            batch_size = len(discrete_params_batch)
        elif feature_masks is not None:
            batch_size = len(feature_masks)
        else:
            batch_size = 1

        scores = np.zeros(batch_size, dtype=np.float64)

        for i in range(batch_size):
            # Get configuration for this sample
            feature_mask = (
                feature_masks[i] if feature_masks is not None else None
            )

            discrete_params = (
                discrete_params_batch[i]
                if discrete_params_batch is not None
                else None
            )

            if (
                continuous_params_batch is not None
                and continuous_param_names is not None
            ):
                continuous_params = dict(
                    zip(
                        continuous_param_names,
                        continuous_params_batch[i],
                        strict=True,
                    )
                )
            else:
                continuous_params = None

            # Evaluate
            result = self.evaluate(
                model_wrapper=model_wrapper,
                feature_mask=feature_mask,
                discrete_params=discrete_params,
                continuous_params=continuous_params,
                train_idx=train_idx,
            )
            scores[i] = result.mean_score

        return scores



@dataclass
class OuterFoldResult:
    """Result for a single outer CV fold.

    Attributes:
        fold: Fold index.
        test_score: Score on outer test set.
        best_config: Best configuration found in inner optimization.
        best_inner_score: Best inner CV score.
        n_features_selected: Number of features selected.
        optimization_time: Time spent on optimization.
        evaluation_time: Time for final evaluation.
    """

    fold: int
    test_score: float
    best_config: dict[str, Any]
    best_inner_score: float
    n_features_selected: int
    optimization_time: float = 0.0
    evaluation_time: float = 0.0


@dataclass
class NestedCVResult:
    """Complete nested CV result.

    Attributes:
        outer_scores: Scores on outer test folds.
        mean_score: Mean outer score.
        std_score: Standard deviation of outer scores.
        ci_lower: Lower bound of 95% confidence interval.
        ci_upper: Upper bound of 95% confidence interval.
        fold_results: Detailed results for each outer fold.
        total_time: Total optimization time.
        best_overall_config: Best configuration across all folds.
    """

    outer_scores: NDArray[np.float64]
    mean_score: float
    std_score: float
    ci_lower: float
    ci_upper: float
    fold_results: list[OuterFoldResult]
    total_time: float
    best_overall_config: dict[str, Any] | None = None

    def summary(self) -> str:
        """Generate summary string.

        Returns:
            Formatted summary of nested CV results.
        """
        lines = [
            "=" * 60,
            "Nested Cross-Validation Results",
            "=" * 60,
            f"Mean Score: {self.mean_score:.4f} Ã‚Â± {self.std_score:.4f}",
            f"95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"Total Time: {self.total_time:.2f}s",
            "",
            "Outer Fold Scores:",
        ]

        for fold_result in self.fold_results:
            lines.append(
                f"  Fold {fold_result.fold}: {fold_result.test_score:.4f} "
                f"(inner: {fold_result.best_inner_score:.4f}, "
                f"features: {fold_result.n_features_selected})"
            )

        lines.extend(["", "=" * 60])
        return "\n".join(lines)


class OuterCVManager:
    """Manager for outer cross-validation loop.

    Orchestrates the nested CV process, managing data splitting,
    inner optimization, and final evaluation.
    """

    def __init__(
        self,
        X: NDArray[np.floating],
        y: NDArray,
        outer_cv: int = 5,
        inner_cv: int = 3,
        stratified: bool = True,
        shuffle: bool = True,
        scoring: str = "accuracy",
        use_gpu: bool = False,
        random_state: int | None = None,
        n_jobs: int = 1,
        verbose: bool = True,
    ) -> None:
        """Initialize outer CV manager.

        Args:
            X: Feature matrix.
            y: Target array.
            outer_cv: Number of outer CV folds.
            inner_cv: Number of inner CV folds.
            stratified: Use stratified splitting.
            shuffle: Shuffle data before splitting.
            scoring: Scoring metric.
            use_gpu: Whether to use GPU.
            random_state: Random seed.
            n_jobs: Number of parallel jobs.
            verbose: Print progress.
        """
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.scoring = scoring
        self.use_gpu = use_gpu
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.data_manager = CVDataManager(X, y, use_gpu=use_gpu)

        self.outer_splitter = CVSplitter(
            n_splits=outer_cv,
            stratified=stratified,
            shuffle=shuffle,
            random_state=random_state,
        )

        self._outer_splits = self.outer_splitter.get_splits(
            len(y), y
        )

        self._scorer = get_scorer(scoring)
        self._needs_proba = scoring in ("roc_auc", "log_loss")

    def run(
        self,
        optimizer_factory: Callable[[NDArray[np.intp]], Any],
        model_config: ModelConfig,
    ) -> NestedCVResult:
        """Run nested cross-validation.

        Args:
            optimizer_factory: Factory function that creates an optimizer
                              given training indices.
            model_config: Model configuration.

        Returns:
            NestedCVResult with scores and details.
        """
        start_time = time.perf_counter()
        fold_results: list[OuterFoldResult] = []
        outer_scores: list[float] = []

        for split in self._outer_splits:
            if self.verbose:
                logger.info(
                    "Outer fold %d/%d",
                    split.fold + 1,
                    self.outer_cv,
                )

            opt_start = time.perf_counter()
            optimizer = optimizer_factory(split.train_idx)
            inner_result = optimizer.optimize()
            opt_time = time.perf_counter() - opt_start

            best_config = self._extract_config(inner_result)

            eval_start = time.perf_counter()
            test_score = self._evaluate_on_test(
                split,
                best_config,
                model_config,
            )
            eval_time = time.perf_counter() - eval_start

            n_features = np.sum(best_config.get("feature_mask", []))

            fold_result = OuterFoldResult(
                fold=split.fold,
                test_score=test_score,
                best_config=best_config,
                best_inner_score=inner_result.best_fitness,
                n_features_selected=int(n_features),
                optimization_time=opt_time,
                evaluation_time=eval_time,
            )

            fold_results.append(fold_result)
            outer_scores.append(test_score)

            if self.verbose:
                logger.info(
                    "  Fold %d: test=%.4f, inner=%.4f, time=%.2fs",
                    split.fold + 1,
                    test_score,
                    inner_result.best_fitness,
                    opt_time + eval_time,
                )

        total_time = time.perf_counter() - start_time

        outer_scores_arr = np.array(outer_scores)
        mean_score = float(np.mean(outer_scores_arr))
        std_score = float(np.std(outer_scores_arr, ddof=1))

        n_folds = len(outer_scores_arr)
        t_value = stats.t.ppf(0.975, df=n_folds - 1)
        ci_margin = t_value * std_score / np.sqrt(n_folds)
        ci_lower = mean_score - ci_margin
        ci_upper = mean_score + ci_margin

        best_idx = np.argmax([r.best_inner_score for r in fold_results])
        best_overall_config = fold_results[best_idx].best_config

        return NestedCVResult(
            outer_scores=outer_scores_arr,
            mean_score=mean_score,
            std_score=std_score,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            fold_results=fold_results,
            total_time=total_time,
            best_overall_config=best_overall_config,
        )

    def _extract_config(self, result: Any) -> dict[str, Any]:
        """Extract configuration from optimization result.

        Args:
            result: Optimization result object.

        Returns:
            Dictionary with configuration.
        """
        config: dict[str, Any] = {}

        if hasattr(result, "best_feature_mask"):
            config["feature_mask"] = ensure_numpy(result.best_feature_mask)

        if hasattr(result, "best_discrete_params"):
            config["discrete_params"] = ensure_numpy(result.best_discrete_params)

        if hasattr(result, "best_continuous_params"):
            config["continuous_params"] = ensure_numpy(result.best_continuous_params)

        if hasattr(result, "best_fitness"):
            config["best_fitness"] = result.best_fitness

        return config

    def _evaluate_on_test(
        self,
        split: CVSplit,
        config: dict[str, Any],
        model_config: ModelConfig,
    ) -> float:
        """Evaluate configuration on outer test set.

        Args:
            split: Outer CV split.
            config: Best configuration from inner optimization.
            model_config: Model configuration.

        Returns:
            Test score.
        """
        feature_mask = config.get("feature_mask")

        X_train, X_test, y_train, y_test = self.data_manager.get_fold_data(
            split,
            feature_mask=feature_mask,
            prefer_gpu=self.use_gpu,
        )

        discrete_params = config.get("discrete_params")
        continuous_params = config.get("continuous_params")

        model_params = dict(model_config.fixed_params)

        if discrete_params is not None:
            for i, (param_name, param_def) in enumerate(
                model_config.discrete_params.items()
            ):
                if i < len(discrete_params):
                    idx = int(discrete_params[i])
                    if "choices" in param_def:
                        model_params[param_name] = param_def["choices"][idx]
                    elif "range" in param_def:
                        low, high = param_def["range"]
                        model_params[param_name] = low + idx
                    else:
                        model_params[param_name] = idx

        if continuous_params is not None:
            for i, (param_name, param_def) in enumerate(
                model_config.continuous_params.items()
            ):
                if i < len(continuous_params):
                    model_params[param_name] = float(continuous_params[i])

        model_class = (
            model_config.cuml_class
            if self.use_gpu and model_config.cuml_class is not None
            else model_config.model_class
        )
        model = model_class(**model_params)
        model.fit(X_train, y_train)

        # Predict and score
        if self._needs_proba and hasattr(model, "predict_proba"):
            y_pred = model.predict_proba(X_test)
        else:
            y_pred = model.predict(X_test)

        y_test_np = ensure_numpy(y_test)
        y_pred_np = ensure_numpy(y_pred)

        return self._scorer(y_test_np, y_pred_np)



@dataclass
class NestedCVConfig:
    """Configuration for nested cross-validation.

    Attributes:
        outer_cv: Number of outer CV folds.
        inner_cv: Number of inner CV folds.
        stratified: Use stratified splitting.
        shuffle: Shuffle data before splitting.
        scoring: Scoring metric.
        use_gpu: Whether to use GPU.
        random_state: Random seed.
        n_jobs: Number of parallel jobs.
        verbose: Print progress.
        refit: Refit best model on full data after CV.
    """

    outer_cv: int = 5
    inner_cv: int = 3
    stratified: bool = True
    shuffle: bool = True
    scoring: str = "accuracy"
    use_gpu: bool = False
    random_state: int | None = None
    n_jobs: int = -1
    verbose: bool = True
    refit: bool = True


class NestedCVOptimizer:
    """High-level nested cross-validation optimizer.

    Combines the hybrid GA-PSO optimizer with nested cross-validation
    for unbiased performance estimation.
    """

    def __init__(
        self,
        config: NestedCVConfig | None = None,
    ) -> None:
        """Initialize nested CV optimizer.

        Args:
            config: Nested CV configuration.
        """
        self.config = config or NestedCVConfig()
        self._result: NestedCVResult | None = None
        self._final_model: Any = None

    def optimize(
        self,
        X: NDArray[np.floating],
        y: NDArray,
        model_config: ModelConfig,
        optimizer_config: Any | None = None,
    ) -> NestedCVResult:
        """Run nested CV optimization.

        Args:
            X: Feature matrix.
            y: Target array.
            model_config: Model configuration.
            optimizer_config: Optional optimizer configuration.

        Returns:
            NestedCVResult with scores and details.
        """
        outer_manager = OuterCVManager(
            X=X,
            y=y,
            outer_cv=self.config.outer_cv,
            inner_cv=self.config.inner_cv,
            stratified=self.config.stratified,
            shuffle=self.config.shuffle,
            scoring=self.config.scoring,
            use_gpu=self.config.use_gpu,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
        )

        def optimizer_factory(train_idx: NDArray[np.intp]) -> Any:
            """Create inner optimizer for given training indices."""
            X_train = X[train_idx]
            y_train = y[train_idx]

            # Create inner CV evaluator
            inner_data_manager = CVDataManager(
                X_train, y_train, use_gpu=self.config.use_gpu
            )
            inner_splitter = CVSplitter(
                n_splits=self.config.inner_cv,
                stratified=self.config.stratified,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
            )

            inner_evaluator = InnerCVEvaluator(
                data_manager=inner_data_manager,
                cv_splitter=inner_splitter,
                scoring=self.config.scoring,
                use_gpu=self.config.use_gpu,
                n_jobs=self.config.n_jobs,
            )

            return _InnerOptimizerWrapper(
                inner_evaluator=inner_evaluator,
                model_config=model_config,
                optimizer_config=optimizer_config,
            )

        result = outer_manager.run(optimizer_factory, model_config)
        self._result = result

        if self.config.refit and result.best_overall_config is not None:
            self._refit_on_full_data(X, y, model_config, result.best_overall_config)

        return result

    def _refit_on_full_data(
        self,
        X: NDArray[np.floating],
        y: NDArray,
        model_config: ModelConfig,
        config: dict[str, Any],
    ) -> None:
        """Refit best model on full dataset.

        Args:
            X: Full feature matrix.
            y: Full target array.
            model_config: Model configuration.
            config: Best configuration.
        """
        feature_mask = config.get("feature_mask")

        if feature_mask is not None:
            X_selected = X[:, feature_mask]
        else:
            X_selected = X

        model_params = dict(model_config.fixed_params)
        discrete_params = config.get("discrete_params")
        continuous_params = config.get("continuous_params")

        if discrete_params is not None:
            for i, (param_name, param_def) in enumerate(
                model_config.discrete_params.items()
            ):
                if i < len(discrete_params):
                    idx = int(discrete_params[i])
                    if "choices" in param_def:
                        model_params[param_name] = param_def["choices"][idx]
                    elif "range" in param_def:
                        low, high = param_def["range"]
                        model_params[param_name] = low + idx
                    else:
                        model_params[param_name] = idx

        if continuous_params is not None:
            for i, param_name in enumerate(model_config.continuous_params.keys()):
                if i < len(continuous_params):
                    model_params[param_name] = float(continuous_params[i])

        model_class = (
            model_config.cuml_class
            if self.config.use_gpu and model_config.cuml_class is not None
            else model_config.model_class
        )
        self._final_model = model_class(**model_params)
        self._final_model.fit(X_selected, y)

        if self.config.verbose:
            logger.info("Refitted model on full dataset (%d samples)", len(y))

    @property
    def result(self) -> NestedCVResult | None:
        """Get the nested CV result."""
        return self._result

    @property
    def final_model(self) -> Any | None:
        """Get the final refitted model."""
        return self._final_model


class _InnerOptimizerWrapper:
    """Wrapper for inner optimization loop using HybridGAPSOOptimizer.

    Bridges the nested CV framework with the hybrid GA-PSO optimizer,
    providing the optimize() interface expected by OuterCVManager.
    """

    def __init__(
        self,
        inner_evaluator: InnerCVEvaluator,
        model_config: ModelConfig,
        optimizer_config: HybridConfig | None = None,
    ) -> None:
        """Initialize inner optimizer wrapper.

        Args:
            inner_evaluator: Inner CV evaluator.
            model_config: Model configuration.
            optimizer_config: Optional HybridConfig for GA-PSO optimizer.
        """
        self.inner_evaluator = inner_evaluator
        self.model_config = model_config
        self.optimizer_config = optimizer_config or HybridConfig()

        self.best_fitness: float = float("-inf")
        self.best_feature_mask: NDArray[np.bool_] | None = None
        self.best_discrete_params: NDArray[np.int64] | None = None
        self.best_continuous_params: NDArray[np.float64] | None = None
        self.hybrid_result: HybridResult | None = None

    def _extract_bounds(
        self,
    ) -> tuple[list[tuple[int, int]], list[tuple[float, float]]]:
        """Extract parameter bounds from model config.

        Returns:
            Tuple of (discrete_bounds, continuous_bounds).
        """
        discrete_bounds: list[tuple[int, int]] = []
        for param_def in self.model_config.discrete_params.values():
            if "choices" in param_def:
                n_choices = len(param_def["choices"])
                discrete_bounds.append((0, n_choices - 1))
            elif "range" in param_def:
                low, high = param_def["range"]
                discrete_bounds.append((0, high - low))
            else:
                discrete_bounds.append((0, 0))

        continuous_bounds: list[tuple[float, float]] = []
        for param_def in self.model_config.continuous_params.values():
            if "range" in param_def:
                low, high = param_def["range"]
                continuous_bounds.append((float(low), float(high)))
            else:
                continuous_bounds.append((0.0, 1.0))

        return discrete_bounds, continuous_bounds

    def optimize(self) -> _InnerOptimizerWrapper:
        """Run inner optimization using HybridGAPSOOptimizer.

        Returns:
            Self with results populated.
        """
        n_features = self.inner_evaluator.data_manager.n_features

        discrete_bounds, continuous_bounds = self._extract_bounds()

        if not discrete_bounds and not continuous_bounds:
            logger.warning("No parameters to optimize, using default configuration")
            self.best_feature_mask = np.ones(n_features, dtype=bool)
            self.best_discrete_params = np.array([], dtype=np.int64)
            self.best_continuous_params = np.array([], dtype=np.float64)
            return self

        fitness_fn = create_cv_fitness_function(
            self.inner_evaluator,
            self.model_config,
        )

        config = self.optimizer_config
        if config.use_gpu != self.inner_evaluator.use_gpu:
            config = HybridConfig(
                ga_config=config.ga_config,
                pso_config=config.pso_config,
                use_chromosome_cache=config.use_chromosome_cache,
                chromosome_cache_size=config.chromosome_cache_size,
                use_full_config_cache=config.use_full_config_cache,
                full_config_cache_size=config.full_config_cache_size,
                use_warm_start=config.use_warm_start,
                warm_start_threshold=config.warm_start_threshold,
                use_gpu=self.inner_evaluator.use_gpu,
                verbose=config.verbose,
                random_seed=config.random_seed,
            )

        try:
            optimizer = HybridGAPSOOptimizer(
                n_features=n_features,
                discrete_bounds=discrete_bounds if discrete_bounds else [(0, 0)],
                continuous_bounds=continuous_bounds if continuous_bounds else [(0.0, 1.0)],
                config=config,
            )

            result = optimizer.optimize(fitness_fn)  # pyright: ignore[reportArgumentType]
            self.hybrid_result = result

            self.best_fitness = result.best_fitness
            self.best_feature_mask = result.best_feature_mask
            self.best_discrete_params = result.best_discrete_params
            self.best_continuous_params = result.best_continuous_params

            logger.info(
                "Inner optimization complete: fitness=%.6f, generations=%d",
                result.best_fitness,
                result.n_generations,
            )

        except Exception as e:
            logger.error("Hybrid optimization failed: %s", e)
            self.best_feature_mask = np.ones(n_features, dtype=bool)
            self.best_discrete_params = np.zeros(
                len(discrete_bounds) if discrete_bounds else 1, dtype=np.int64
            )
            self.best_continuous_params = np.array(
                [(b[0] + b[1]) / 2 for b in continuous_bounds]
                if continuous_bounds
                else [0.5],
                dtype=np.float64,
            )

        return self


def create_cv_fitness_function(
    inner_evaluator: InnerCVEvaluator,
    model_config: ModelConfig,
    train_idx: NDArray[np.intp] | None = None,
) -> Callable[
    [NDArray[np.bool_], NDArray[np.int64], NDArray[np.float64]], float
]:
    """Create a fitness function for the hybrid optimizer.

    This function wraps inner CV evaluation to provide the fitness function
    interface expected by the GA-PSO optimizer.

    Args:
        inner_evaluator: Inner CV evaluator.
        model_config: Model configuration.
        train_idx: Optional training indices (for nested CV).

    Returns:
        Fitness function compatible with HybridGAPSOOptimizer.
    """
    model_wrapper = ModelWrapper(
        model_config,
        use_gpu=inner_evaluator.use_gpu,
    )

    discrete_param_names = list(model_config.discrete_params.keys())  # noqa: F841
    continuous_param_names = list(model_config.continuous_params.keys())

    def fitness_fn(
        feature_mask: NDArray[np.bool_],
        discrete_params: NDArray[np.int64],
        continuous_params: NDArray[np.float64],
    ) -> float:
        """Compute fitness using inner CV.

        Args:
            feature_mask: Boolean feature mask.
            discrete_params: Discrete parameter indices.
            continuous_params: Continuous parameter values.

        Returns:
            Mean inner CV score.
        """
        discrete_values: dict[str, Any] = {}
        for i, (param_name, param_def) in enumerate(
            model_config.discrete_params.items()
        ):
            if i < len(discrete_params):
                idx = int(discrete_params[i])
                if "choices" in param_def:
                    choices = param_def["choices"]
                    if 0 <= idx < len(choices):
                        discrete_values[param_name] = choices[idx]
                    else:
                        logger.warning(
                            "Index %d out of bounds for %s choices (len=%d), using first",
                            idx, param_name, len(choices)
                        )
                        discrete_values[param_name] = choices[0]
                elif "range" in param_def:
                    low, _ = param_def["range"]
                    discrete_values[param_name] = low + idx
                else:
                    discrete_values[param_name] = idx

        continuous_values: dict[str, Any] = {}
        for i, param_name in enumerate(continuous_param_names):
            if i < len(continuous_params):
                continuous_values[param_name] = float(continuous_params[i])

        try:
            result = inner_evaluator.evaluate(
                model_wrapper=model_wrapper,
                feature_mask=feature_mask,
                discrete_params=discrete_values,
                continuous_params=continuous_values,
                train_idx=train_idx,
            )
            return result.mean_score
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Fitness evaluation failed with params=%s, continuous=%s: %s",
                discrete_values, continuous_values, e
            )
            return float("-inf")

    return fitness_fn


def create_batch_cv_fitness_function(
    inner_evaluator: InnerCVEvaluator,
    model_config: ModelConfig,
    train_idx: NDArray[np.intp] | None = None,
) -> Callable[
    [NDArray[np.bool_], NDArray[np.int64], NDArray[np.float64]],
    NDArray[np.float64],
]:
    """Create a batch fitness function for the hybrid optimizer.

    Args:
        inner_evaluator: Inner CV evaluator.
        model_config: Model configuration.
        train_idx: Optional training indices.

    Returns:
        Batch fitness function.
    """
    single_fn = create_cv_fitness_function(
        inner_evaluator, model_config, train_idx
    )

    def batch_fitness_fn(
        feature_masks: NDArray[np.bool_],
        discrete_params: NDArray[np.int64],
        continuous_params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Batch evaluate configurations.

        Args:
            feature_masks: Feature masks of shape (batch_size, n_features).
            discrete_params: Discrete params of shape (batch_size, n_discrete).
            continuous_params: Continuous params of shape (batch_size, n_continuous).

        Returns:
            Array of fitness values.
        """
        batch_size = len(continuous_params)
        scores = np.zeros(batch_size, dtype=np.float64)

        for i in range(batch_size):
            scores[i] = single_fn(
                feature_masks[i] if feature_masks.ndim > 1 else feature_masks,
                discrete_params[i] if discrete_params.ndim > 1 else discrete_params,
                continuous_params[i],
            )

        return scores

    return batch_fitness_fn
