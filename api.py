"""High-Level API for GPU-Accelerated Hybrid Optimization.

This module provides a simple, user-friendly API for the GPU-accelerated
hybrid GA-PSO optimization framework.

The module provides:
- Simple `optimize()` function for one-line optimization
- `optimize_with_cv()` for full nested cross-validation
- Quick optimization presets for common scenarios
- Factory functions for creating optimizers and evaluators
- Model registry for easy model selection

Example:
    >>> from api import optimize, quick_optimize
    >>> from sklearn.datasets import make_classification
    >>>
    >>> # Simple usage
    >>> X, y = make_classification(n_samples=1000, n_features=50)
    >>> result = optimize(X, y, model='random_forest', scoring='accuracy')
    >>> print(f"Best score: {result.mean_score:.4f}")
    >>>
    >>> # Quick optimization for common scenarios
    >>> result = quick_optimize(X, y, preset='fast')
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from .gpu_config import (
    GPUConfig,
    cpu_context,
    estimate_problem_memory,
    gpu_context,
    gpu_info,
    validate_gpu_setup,
)
from .optimization_config import (
    GASettings,
    PSOSettings,
    FeatureSelectionSettings,
    CustomParameterSpace,
    ParameterSpec,
    CachingSettings,
    CrossoverType,
    MutationType,
    SelectionType,
    InertiaStrategy,
    BoundaryHandling,
    InitializationMethod,
)
from .hybrid_optimizer import (
    HybridGAPSOOptimizer,
    HybridConfig,
    HybridResult,
)
from .nested_cv import (
    NestedCVOptimizer,
    NestedCVConfig,
    NestedCVResult,
    ModelConfig,
    CVSplitter,
    CVDataManager,
    InnerCVEvaluator,
    create_cv_fitness_function,
)
from .evaluation import (
    EvaluationPipeline,
    PipelineConfig,
    BatchEvaluator,
    GPUScorer,
    create_cv_splits,
)
from .utils.performance import (
    GPUMemoryManager,
    PerformanceProfiler,
    TransferOptimizer,
    create_memory_manager,
    performance_context,
)
from .utils.scoring import get_scorer, SCORING_FUNCTIONS, needs_probability_predictions
from .utils.common import TaskType as CommonTaskType, ScoringMetric
from .multi_model import (
    ModelRegistry as MultiModelRegistry,
    ModelDefinition,
    ParameterDefinition,
    ParameterType,
)
from .results import GenerationStats, OptimizationResults, FeatureImportance

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Configure logging
logger = logging.getLogger(__name__)


ModelType = Literal[
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "svm",
    "logistic_regression",
    "ridge",
    "lasso",
    "elastic_net",
    "knn",
    "mlp",
    "extra_trees",
]

TaskType = Literal["classification", "regression", "auto"]
ScoringType = Literal[
    "accuracy",
    "balanced_accuracy",
    "f1",
    "f1_macro",
    "f1_weighted",
    "precision",
    "recall",
    "roc_auc",
    "log_loss",
    "mse",
    "rmse",
    "mae",
    "r2",
    "mape",
]

PresetType = Literal["fast", "balanced", "thorough", "memory_efficient", "high_accuracy"]



@dataclass
class OptimizeResult:
    """Result from the optimize() function.

    Attributes:
        mean_score: Mean cross-validated score.
        std_score: Standard deviation of CV scores.
        ci_lower: Lower confidence interval bound.
        ci_upper: Upper confidence interval bound.
        best_params: Best hyperparameters found.
        selected_features: Indices of selected features.
        feature_mask: Boolean mask for feature selection.
        n_features_selected: Number of features selected.
        fold_scores: Scores for each outer CV fold.
        best_model: Best model fitted on all data (if refit=True).
        elapsed_time: Total optimization time in seconds.
        n_evaluations: Total number of evaluations performed.
        converged: Whether optimization converged.
        gpu_used: Whether GPU was used.
        detailed_results: Full nested CV results (if available).
        generation_stats: Structured generation statistics (from results.py).
    """

    mean_score: float
    std_score: float
    ci_lower: float
    ci_upper: float
    best_params: dict[str, Any]
    selected_features: NDArray[np.intp]
    feature_mask: NDArray[np.bool_]
    n_features_selected: int
    fold_scores: NDArray[np.float64]
    best_model: Any | None = None
    elapsed_time: float = 0.0
    n_evaluations: int = 0
    converged: bool = False
    gpu_used: bool = False
    detailed_results: Any | None = None
    generation_stats: list[GenerationStats] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a summary string of the results.

        Returns:
            Formatted summary string.
        """
        lines = [
            "=" * 60,
            "Optimization Results",
            "=" * 60,
            f"Score: {self.mean_score:.4f} Ãƒâ€šÃ‚Â± {self.std_score:.4f}",
            f"95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"Features: {self.n_features_selected} selected",
            f"Time: {self.elapsed_time:.1f}s",
            f"Evaluations: {self.n_evaluations}",
            f"Converged: {self.converged}",
            f"GPU: {'Yes' if self.gpu_used else 'No'}",
            "",
            "Best Parameters:",
        ]

        for key, value in self.best_params.items():
            lines.append(f"  {key}: {value}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OptimizeResult(score={self.mean_score:.4f}Ãƒâ€šÃ‚Â±{self.std_score:.4f}, "
            f"features={self.n_features_selected}, time={self.elapsed_time:.1f}s)"
        )




class ModelRegistry:
    """Registry for supported models and their default configurations.

    Provides easy access to model classes and default hyperparameter
    search spaces.
    """

    _classifiers: dict[str, dict[str, Any]] = {}
    _regressors: dict[str, dict[str, Any]] = {}
    _initialized: bool = False

    @classmethod
    def _initialize(cls) -> None:
        """Initialize model registry with default models."""
        if cls._initialized:
            return

        cls._classifiers = {
            "random_forest": {
                "class": "sklearn.ensemble.RandomForestClassifier",
                "discrete_params": {
                    "n_estimators": {"range": [10, 100], "type": "integer"},
                    "max_depth": {"range": [3, 15], "type": "integer"},
                    "min_samples_split": {"range": [2, 10], "type": "integer"},
                    "min_samples_leaf": {"range": [1, 5], "type": "integer"},
                    "criterion": {"choices": ["gini", "entropy"], "type": "categorical"},
                },
                "continuous_params": {
                    "max_features": {"range": [0.1, 1.0], "type": "continuous"},
                },
                "fixed_params": {"n_jobs": 1, "random_state": None},
            },
            "gradient_boosting": {
                "class": "sklearn.ensemble.GradientBoostingClassifier",
                "discrete_params": {
                    "n_estimators": {"range": [50, 500], "type": "integer"},
                    "max_depth": {"range": [2, 15], "type": "integer"},
                    "min_samples_split": {"range": [2, 20], "type": "integer"},
                    "min_samples_leaf": {"range": [1, 10], "type": "integer"},
                },
                "continuous_params": {
                    "learning_rate": {"range": [0.01, 0.3], "type": "continuous", "log_scale": True},
                    "subsample": {"range": [0.5, 1.0], "type": "continuous"},
                    "max_features": {"range": [0.1, 1.0], "type": "continuous"},
                },
                "fixed_params": {"random_state": None},
            },
            "extra_trees": {
                "class": "sklearn.ensemble.ExtraTreesClassifier",
                "discrete_params": {
                    "n_estimators": {"range": [50, 500], "type": "integer"},
                    "max_depth": {"range": [3, 30], "type": "integer"},
                    "min_samples_split": {"range": [2, 20], "type": "integer"},
                    "min_samples_leaf": {"range": [1, 10], "type": "integer"},
                    "criterion": {"choices": ["gini", "entropy"], "type": "categorical"},
                },
                "continuous_params": {
                    "max_features": {"range": [0.1, 1.0], "type": "continuous"},
                },
                "fixed_params": {"n_jobs": 1, "random_state": None},
            },
            "svm": {
                "class": "sklearn.svm.SVC",
                "discrete_params": {
                    "kernel": {"choices": ["rbf", "poly", "sigmoid"], "type": "categorical"},
                    "degree": {"range": [2, 5], "type": "integer"},
                },
                "continuous_params": {
                    "C": {"range": [0.01, 100.0], "type": "continuous", "log_scale": True},
                    "gamma": {"range": [0.001, 10.0], "type": "continuous", "log_scale": True},
                },
                "fixed_params": {"probability": True, "random_state": None},
            },
            "logistic_regression": {
                "class": "sklearn.linear_model.LogisticRegression",
                "discrete_params": {
                    "penalty": {"choices": ["l1", "l2", "elasticnet"], "type": "categorical"},
                    "solver": {"choices": ["saga"], "type": "categorical"},
                },
                "continuous_params": {
                    "C": {"range": [0.01, 100.0], "type": "continuous", "log_scale": True},
                    "l1_ratio": {"range": [0.0, 1.0], "type": "continuous"},
                },
                "fixed_params": {"max_iter": 1000, "random_state": None},
            },
            "knn": {
                "class": "sklearn.neighbors.KNeighborsClassifier",
                "discrete_params": {
                    "n_neighbors": {"range": [1, 30], "type": "integer"},
                    "weights": {"choices": ["uniform", "distance"], "type": "categorical"},
                    "p": {"choices": [1, 2], "type": "categorical"},
                },
                "continuous_params": {},
                "fixed_params": {"n_jobs": 1},
            },
            "mlp": {
                "class": "sklearn.neural_network.MLPClassifier",
                "discrete_params": {
                    "hidden_layer_sizes": {
                        "choices": [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                        "type": "categorical",
                    },
                    "activation": {"choices": ["relu", "tanh"], "type": "categorical"},
                },
                "continuous_params": {
                    "alpha": {"range": [0.0001, 0.1], "type": "continuous", "log_scale": True},
                    "learning_rate_init": {"range": [0.0001, 0.1], "type": "continuous", "log_scale": True},
                },
                "fixed_params": {"max_iter": 500, "early_stopping": True, "random_state": None},
            },
        }

        cls._regressors = {
            "random_forest": {
                "class": "sklearn.ensemble.RandomForestRegressor",
                "discrete_params": {
                    "n_estimators": {"range": [50, 500], "type": "integer"},
                    "max_depth": {"range": [3, 30], "type": "integer"},
                    "min_samples_split": {"range": [2, 20], "type": "integer"},
                    "min_samples_leaf": {"range": [1, 10], "type": "integer"},
                    "criterion": {"choices": ["squared_error", "absolute_error"], "type": "categorical"},
                },
                "continuous_params": {
                    "max_features": {"range": [0.1, 1.0], "type": "continuous"},
                },
                "fixed_params": {"n_jobs": 1, "random_state": None},
            },
            "gradient_boosting": {
                "class": "sklearn.ensemble.GradientBoostingRegressor",
                "discrete_params": {
                    "n_estimators": {"range": [50, 500], "type": "integer"},
                    "max_depth": {"range": [2, 15], "type": "integer"},
                    "min_samples_split": {"range": [2, 20], "type": "integer"},
                    "min_samples_leaf": {"range": [1, 10], "type": "integer"},
                },
                "continuous_params": {
                    "learning_rate": {"range": [0.01, 0.3], "type": "continuous", "log_scale": True},
                    "subsample": {"range": [0.5, 1.0], "type": "continuous"},
                    "max_features": {"range": [0.1, 1.0], "type": "continuous"},
                },
                "fixed_params": {"random_state": None},
            },
            "ridge": {
                "class": "sklearn.linear_model.Ridge",
                "discrete_params": {},
                "continuous_params": {
                    "alpha": {"range": [0.001, 100.0], "type": "continuous", "log_scale": True},
                },
                "fixed_params": {"random_state": None},
            },
            "lasso": {
                "class": "sklearn.linear_model.Lasso",
                "discrete_params": {},
                "continuous_params": {
                    "alpha": {"range": [0.001, 10.0], "type": "continuous", "log_scale": True},
                },
                "fixed_params": {"max_iter": 1000, "random_state": None},
            },
            "elastic_net": {
                "class": "sklearn.linear_model.ElasticNet",
                "discrete_params": {},
                "continuous_params": {
                    "alpha": {"range": [0.001, 10.0], "type": "continuous", "log_scale": True},
                    "l1_ratio": {"range": [0.0, 1.0], "type": "continuous"},
                },
                "fixed_params": {"max_iter": 1000, "random_state": None},
            },
            "svm": {
                "class": "sklearn.svm.SVR",
                "discrete_params": {
                    "kernel": {"choices": ["rbf", "poly", "sigmoid"], "type": "categorical"},
                    "degree": {"range": [2, 5], "type": "integer"},
                },
                "continuous_params": {
                    "C": {"range": [0.01, 100.0], "type": "continuous", "log_scale": True},
                    "gamma": {"range": [0.001, 10.0], "type": "continuous", "log_scale": True},
                    "epsilon": {"range": [0.01, 1.0], "type": "continuous"},
                },
                "fixed_params": {},
            },
            "knn": {
                "class": "sklearn.neighbors.KNeighborsRegressor",
                "discrete_params": {
                    "n_neighbors": {"range": [1, 30], "type": "integer"},
                    "weights": {"choices": ["uniform", "distance"], "type": "categorical"},
                    "p": {"choices": [1, 2], "type": "categorical"},
                },
                "continuous_params": {},
                "fixed_params": {"n_jobs": 1},
            },
            "mlp": {
                "class": "sklearn.neural_network.MLPRegressor",
                "discrete_params": {
                    "hidden_layer_sizes": {
                        "choices": [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                        "type": "categorical",
                    },
                    "activation": {"choices": ["relu", "tanh"], "type": "categorical"},
                },
                "continuous_params": {
                    "alpha": {"range": [0.0001, 0.1], "type": "continuous", "log_scale": True},
                    "learning_rate_init": {"range": [0.0001, 0.1], "type": "continuous", "log_scale": True},
                },
                "fixed_params": {"max_iter": 500, "early_stopping": True, "random_state": None},
            },
        }

        cls._initialized = True

    @classmethod
    def get_model_config(
        cls,
        model_name: str,
        task: TaskType = "classification",
    ) -> dict[str, Any]:
        """Get model configuration by name.

        Args:
            model_name: Name of the model.
            task: Task type ('classification' or 'regression').

        Returns:
            Model configuration dictionary.

        Raises:
            ValueError: If model not found.
        """
        cls._initialize()

        registry = cls._classifiers if task == "classification" else cls._regressors

        if model_name not in registry:
            available = list(registry.keys())
            msg = f"Model '{model_name}' not found. Available: {available}"
            raise ValueError(msg)

        return registry[model_name].copy()

    @classmethod
    def list_models(cls, task: TaskType = "classification") -> list[str]:
        """List available models.

        Args:
            task: Task type to list models for.

        Returns:
            List of model names.
        """
        cls._initialize()
        registry = cls._classifiers if task == "classification" else cls._regressors
        return list(registry.keys())

    @classmethod
    def get_model_class(cls, model_name: str, task: TaskType = "classification") -> type:
        """Get the model class by name.

        Args:
            model_name: Name of the model.
            task: Task type.

        Returns:
            Model class.
        """
        config = cls.get_model_config(model_name, task)
        class_path = config["class"]

        # Import the class
        module_path, class_name = class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    @classmethod
    def to_model_definition(
        cls,
        model_name: str,
        task: TaskType = "classification",
    ) -> ModelDefinition:
        """Convert a model config to a multi_model.ModelDefinition.

        This provides integration with the multi_model module for
        advanced multi-model optimization scenarios.

        Args:
            model_name: Name of the model.
            task: Task type.

        Returns:
            ModelDefinition for use with ModelRegistry from multi_model.
        """
        config = cls.get_model_config(model_name, task)
        model_class = cls.get_model_class(model_name, task)

        discrete_params: dict[str, ParameterDefinition] = {}
        for name, param_def in config.get("discrete_params", {}).items():
            param_type = param_def.get("type", "integer")
            if param_type == "integer":
                discrete_params[name] = ParameterDefinition(
                    name=name,
                    param_type=ParameterType.INTEGER,
                    lower=param_def["range"][0],
                    upper=param_def["range"][1],
                    log_scale=param_def.get("log_scale", False),
                )
            elif param_type == "categorical":
                discrete_params[name] = ParameterDefinition(
                    name=name,
                    param_type=ParameterType.CATEGORICAL,
                    choices=param_def["choices"],
                )

        continuous_params: dict[str, ParameterDefinition] = {}
        for name, param_def in config.get("continuous_params", {}).items():
            continuous_params[name] = ParameterDefinition(
                name=name,
                param_type=ParameterType.CONTINUOUS,
                lower=param_def["range"][0],
                upper=param_def["range"][1],
                log_scale=param_def.get("log_scale", False),
            )

        task_type_enum = (
            CommonTaskType.CLASSIFICATION
            if task == "classification"
            else CommonTaskType.REGRESSION
        )

        return ModelDefinition(
            name=model_name,
            model_class=model_class,
            discrete_params=discrete_params,
            continuous_params=continuous_params,
            fixed_params=config.get("fixed_params", {}),
            task_type=task_type_enum,
        )

    @classmethod
    def create_multi_model_registry(
        cls,
        model_names: list[str],
        task: TaskType = "classification",
    ) -> MultiModelRegistry:
        """Create a MultiModelRegistry from a list of model names.

        Args:
            model_names: List of model names to include.
            task: Task type.

        Returns:
            MultiModelRegistry populated with the specified models.
        """
        task_type_enum = (
            CommonTaskType.CLASSIFICATION
            if task == "classification"
            else CommonTaskType.REGRESSION
        )
        registry = MultiModelRegistry(task_type=task_type_enum)

        for name in model_names:
            model_def = cls.to_model_definition(name, task)
            registry.register(model_def)

        return registry



@dataclass
class OptimizationPreset:
    """Preset configuration for optimization."""

    name: str
    description: str
    population_size: int
    max_generations: int
    swarm_size: int
    pso_iterations: int
    outer_cv: int
    inner_cv: int
    early_stopping_generations: int
    use_gpu: bool


PRESETS: dict[str, OptimizationPreset] = {
    "fast": OptimizationPreset(
        name="fast",
        description="Quick optimization for rapid prototyping",
        population_size=20,
        max_generations=15,
        swarm_size=10,
        pso_iterations=10,
        outer_cv=3,
        inner_cv=2,
        early_stopping_generations=5,
        use_gpu=True,
    ),
    "balanced": OptimizationPreset(
        name="balanced",
        description="Balanced speed and accuracy",
        population_size=50,
        max_generations=50,
        swarm_size=30,
        pso_iterations=30,
        outer_cv=5,
        inner_cv=3,
        early_stopping_generations=15,
        use_gpu=True,
    ),
    "thorough": OptimizationPreset(
        name="thorough",
        description="Thorough search for best results",
        population_size=100,
        max_generations=100,
        swarm_size=50,
        pso_iterations=50,
        outer_cv=5,
        inner_cv=5,
        early_stopping_generations=25,
        use_gpu=True,
    ),
    "memory_efficient": OptimizationPreset(
        name="memory_efficient",
        description="Optimized for limited memory",
        population_size=30,
        max_generations=50,
        swarm_size=20,
        pso_iterations=30,
        outer_cv=3,
        inner_cv=3,
        early_stopping_generations=15,
        use_gpu=True,
    ),
    "high_accuracy": OptimizationPreset(
        name="high_accuracy",
        description="Maximum accuracy with extensive search",
        population_size=150,
        max_generations=150,
        swarm_size=75,
        pso_iterations=75,
        outer_cv=10,
        inner_cv=5,
        early_stopping_generations=40,
        use_gpu=True,
    ),
}


def get_preset(name: PresetType) -> OptimizationPreset:
    """Get optimization preset by name.

    Args:
        name: Preset name.

    Returns:
        OptimizationPreset configuration.

    Raises:
        ValueError: If preset not found.
    """
    if name not in PRESETS:
        available = list(PRESETS.keys())
        msg = f"Preset '{name}' not found. Available: {available}"
        raise ValueError(msg)
    return PRESETS[name]



def optimize(
    X: NDArray[np.floating],
    y: NDArray,
    model: ModelType | str | type = "random_forest",
    *,
    param_space: CustomParameterSpace | None = None,
    task: TaskType = "auto",
    scoring: ScoringType = "accuracy",
    cv: int = 5,
    # GA settings
    ga_settings: GASettings | None = None,
    n_generations: int | None = None,
    population_size: int | None = None,
    crossover_type: CrossoverType | str | None = None,
    crossover_rate: float | None = None,
    mutation_type: MutationType | str | None = None,
    mutation_rate: float | None = None,
    selection_type: SelectionType | str | None = None,
    tournament_size: int | None = None,
    elitism_count: int | None = None,
    # PSO settings
    pso_settings: PSOSettings | None = None,
    swarm_size: int | None = None,
    pso_iterations: int | None = None,
    inertia_strategy: InertiaStrategy | str | None = None,
    inertia_start: float | None = None,
    inertia_end: float | None = None,
    cognitive_coef: float | None = None,
    social_coef: float | None = None,
    velocity_clamp: float | None = None,
    boundary_handling: BoundaryHandling | str | None = None,
    # Feature selection
    feature_selection: bool | FeatureSelectionSettings = True,
    min_features: int | None = None,
    max_features: int | None = None,
    # Caching
    caching_settings: CachingSettings | None = None,
    use_caching: bool = True,
    # GPU and execution
    use_gpu: bool | None = None,
    gpu_config: GPUConfig | None = None,
    n_jobs: int = -1,
    random_state: int | None = None,
    verbose: int = 1,
    refit: bool = True,
    # Fixed model parameters
    **model_params: Any,
) -> OptimizeResult:
    """Optimize a machine learning pipeline with feature selection and hyperparameter tuning.

    This is the main entry point for the hybrid GA-PSO optimization framework.
    It provides a flexible interface supporting both simple and advanced usage.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target array of shape (n_samples,).
        model: Model type name, class, or instance. Supported names:
            'random_forest', 'gradient_boosting', 'svm', 'logistic_regression',
            'ridge', 'lasso', 'elastic_net', 'knn', 'mlp', 'extra_trees'.
        param_space: Custom parameter space for the model. If provided,
            overrides default parameter spaces.

        task: Task type ('classification', 'regression', or 'auto').
        scoring: Scoring metric. Classification: 'accuracy', 'f1', 'roc_auc', etc.
            Regression: 'mse', 'r2', 'mae', etc.
        cv: Number of outer cross-validation folds.

        ga_settings: Complete GA configuration object (overrides individual GA params).
        n_generations: Maximum generations for genetic algorithm.
        population_size: Population size for genetic algorithm.
        crossover_type: Crossover operator type ('uniform', 'blend', 'sbx', etc.).
        crossover_rate: Probability of crossover (0.0-1.0).
        mutation_type: Mutation operator type ('uniform', 'gaussian', 'adaptive', etc.).
        mutation_rate: Probability of mutation per gene (0.0-1.0).
        selection_type: Selection strategy ('tournament', 'roulette', 'rank', etc.).
        tournament_size: Tournament size for tournament selection.
        elitism_count: Number of best individuals preserved each generation.

        pso_settings: Complete PSO configuration object (overrides individual PSO params).
        swarm_size: Swarm size for particle swarm optimization.
        pso_iterations: Maximum PSO iterations per chromosome.
        inertia_strategy: Inertia weight strategy ('constant', 'linear_decay', 'adaptive', etc.).
        inertia_start: Initial inertia weight (0.0-1.0).
        inertia_end: Final inertia weight for decay strategies (0.0-1.0).
        cognitive_coef: Cognitive (personal best) coefficient (c1).
        social_coef: Social (global best) coefficient (c2).
        velocity_clamp: Velocity clamping factor (fraction of search range).
        boundary_handling: How to handle particles at boundaries ('clamp', 'reflect', 'wrap').

        feature_selection: Whether to perform feature selection, or FeatureSelectionSettings.
        min_features: Minimum number of features to select.
        max_features: Maximum number of features to select.

        caching_settings: Configuration for result caching.
        use_caching: Whether to cache evaluation results.

        use_gpu: Whether to use GPU acceleration. None for auto-detect.
        gpu_config: Detailed GPU configuration (overrides use_gpu).
        n_jobs: Number of parallel jobs (-1 for all cores).
        random_state: Random seed for reproducibility.
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
        refit: Refit best model on all data after optimization.
        **model_params: Additional fixed parameters for the model.

    Returns:
        OptimizeResult with optimization results.

    Example:
        >>> # Simple usage with defaults
        >>> result = optimize(X, y, model='random_forest', scoring='accuracy')
        >>>
        >>> # Custom model with custom parameter space
        >>> from sklearn.ensemble import GradientBoostingClassifier
        >>> space = CustomParameterSpace()
        >>> space.add_integer('n_estimators', 50, 500)
        >>> space.add_integer('max_depth', 2, 15)
        >>> space.add_continuous('learning_rate', 0.01, 0.3, log_scale=True)
        >>> space.add_continuous('subsample', 0.5, 1.0)
        >>> space.set_fixed(random_state=42)
        >>> 
        >>> result = optimize(
        ...     X, y,
        ...     model=GradientBoostingClassifier,
        ...     param_space=space,
        ...     scoring='roc_auc',
        ... )
        >>>
        >>> # Advanced usage with detailed configuration
        >>> ga = GASettings(
        ...     population_size=100,
        ...     crossover_type=CrossoverType.BLEND,
        ...     crossover_rate=0.85,
        ...     mutation_type=MutationType.ADAPTIVE,
        ...     selection_type=SelectionType.TOURNAMENT,
        ...     tournament_size=5,
        ... )
        >>> pso = PSOSettings(
        ...     swarm_size=50,
        ...     inertia_strategy=InertiaStrategy.LINEAR_DECAY,
        ...     inertia_start=0.9,
        ...     inertia_end=0.4,
        ...     cognitive_coef=2.0,
        ...     social_coef=2.0,
        ... )
        >>> result = optimize(
        ...     X, y,
        ...     model='gradient_boosting',
        ...     ga_settings=ga,
        ...     pso_settings=pso,
        ...     cv=5,
        ...     use_gpu=True,
        ... )
    """
    start_time = time.time()

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    if X.ndim != 2:
        msg = f"X must be 2-dimensional, got {X.ndim} dimensions"
        raise ValueError(msg)

    n_samples, n_features = X.shape

    if len(y) != n_samples:
        msg = f"X and y have different sample counts: {n_samples} vs {len(y)}"
        raise ValueError(msg)

    if task == "auto":
        unique_values = np.unique(y)
        if len(unique_values) <= 20 and np.all(unique_values == unique_values.astype(int)):
            task = "classification"
            if verbose > 0:
                logger.info("Auto-detected task: classification")
        else:
            task = "regression"
            if verbose > 0:
                logger.info("Auto-detected task: regression")

    if gpu_config is not None:
        effective_gpu_config = gpu_config.get_effective_config()
    elif use_gpu is not None:
        effective_gpu_config = GPUConfig(use_gpu=use_gpu).get_effective_config()
    else:
        effective_gpu_config = GPUConfig.auto().get_effective_config()

    gpu_used = effective_gpu_config.use_gpu and gpu_info.is_available()

    if verbose > 0:
        backend = "GPU" if gpu_used else "CPU"
        logger.info("Using %s backend", backend)

    # Build GA settings
    effective_ga = _build_ga_settings(
        ga_settings=ga_settings,
        n_generations=n_generations,
        population_size=population_size,
        crossover_type=crossover_type,
        crossover_rate=crossover_rate,
        mutation_type=mutation_type,
        mutation_rate=mutation_rate,
        selection_type=selection_type,
        tournament_size=tournament_size,
        elitism_count=elitism_count,
    )

    # Build PSO settings
    effective_pso = _build_pso_settings(
        pso_settings=pso_settings,
        swarm_size=swarm_size,
        pso_iterations=pso_iterations,
        inertia_strategy=inertia_strategy,
        inertia_start=inertia_start,
        inertia_end=inertia_end,
        cognitive_coef=cognitive_coef,
        social_coef=social_coef,
        velocity_clamp=velocity_clamp,
        boundary_handling=boundary_handling,
    )

    # Build feature selection settings
    effective_feature_selection = _build_feature_selection_settings(
        feature_selection=feature_selection,
        min_features=min_features,
        max_features=max_features,
        n_features=n_features,
    )

    # Build caching settings
    effective_caching = caching_settings if caching_settings is not None else CachingSettings(
        use_chromosome_cache=use_caching,
        use_full_config_cache=use_caching,
    )

    # Get model configuration
    if param_space is not None:
        # Use custom parameter space
        model_class = model if isinstance(model, type) else ModelRegistry.get_model_class(model, task)
        discrete_bounds = param_space.get_discrete_bounds()
        discrete_param_names = [p.name for p in param_space.discrete_params]
        continuous_bounds = param_space.get_continuous_bounds()
        continuous_param_names = [p.name for p in param_space.continuous_params]
        fixed_params = param_space.fixed_params.copy()
        fixed_params.update(model_params)
        model_config = {"discrete_params": {}, "continuous_params": {}}  # For compatibility
    elif isinstance(model, str):
        model_config = ModelRegistry.get_model_config(model, task)
        model_class = ModelRegistry.get_model_class(model, task)
        discrete_bounds, discrete_param_names = _extract_discrete_bounds(model_config)
        continuous_bounds, continuous_param_names = _extract_continuous_bounds(model_config)
        fixed_params = model_config.get("fixed_params", {}).copy()
        fixed_params.update(model_params)
    elif isinstance(model, type):
        model_class = model
        model_config = {"discrete_params": {}, "continuous_params": {}, "fixed_params": {}}
        discrete_bounds = []
        discrete_param_names = []
        continuous_bounds = []
        continuous_param_names = []
        fixed_params = model_params.copy()
    else:
        msg = f"model must be string, class, or provide param_space, got {type(model)}"
        raise TypeError(msg)

    # Apply random state
    if random_state is not None and "random_state" in fixed_params:
        fixed_params["random_state"] = random_state

    if verbose > 0:
        logger.info("Optimization configuration:")
        logger.info("  Samples: %d, Features: %d", n_samples, n_features)
        logger.info("  Model: %s", model_class.__name__)
        logger.info("  Scoring: %s", scoring)
        logger.info("  CV folds: %d", cv)
        logger.info("  GA: pop=%d, gen=%d, crossover=%s, mutation=%s",
                    effective_ga.population_size, effective_ga.max_generations,
                    effective_ga.crossover_type.value, effective_ga.mutation_type.value)
        logger.info("  PSO: swarm=%d, iter=%d, inertia=%s",
                    effective_pso.swarm_size, effective_pso.max_iterations,
                    effective_pso.inertia_strategy.value)
        logger.info("  Discrete params: %d, Continuous params: %d",
                    len(discrete_bounds), len(continuous_bounds))

    result = _run_optimization(
        X=X,
        y=y,
        model_class=model_class,
        model_config=model_config,
        fixed_params=fixed_params,
        discrete_bounds=discrete_bounds,
        discrete_param_names=discrete_param_names,
        continuous_bounds=continuous_bounds,
        continuous_param_names=continuous_param_names,
        n_features=n_features,
        feature_selection_settings=effective_feature_selection,
        scoring=scoring,
        cv=cv,
        ga_settings=effective_ga,
        pso_settings=effective_pso,
        caching_settings=effective_caching,
        gpu_config=effective_gpu_config,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        refit=refit,
        task=task,
        param_space=param_space,
    )

    elapsed_time = time.time() - start_time
    result.elapsed_time = elapsed_time
    result.gpu_used = gpu_used

    if verbose > 0:
        logger.info("Optimization completed in %.1fs", elapsed_time)
        logger.info("Best score: %.4f Ãƒâ€šÃ‚Â± %.4f", result.mean_score, result.std_score)

    return result


def _build_ga_settings(
    ga_settings: GASettings | None,
    n_generations: int | None,
    population_size: int | None,
    crossover_type: CrossoverType | str | None,
    crossover_rate: float | None,
    mutation_type: MutationType | str | None,
    mutation_rate: float | None,
    selection_type: SelectionType | str | None,
    tournament_size: int | None,
    elitism_count: int | None,
) -> GASettings:
    """Build effective GA settings from various inputs."""
    if ga_settings is not None:
        # Start with provided settings, override with individual params if set
        result = GASettings(
            population_size=population_size if population_size is not None else ga_settings.population_size,
            max_generations=n_generations if n_generations is not None else ga_settings.max_generations,
            crossover_type=CrossoverType.from_string(crossover_type) if isinstance(crossover_type, str) else (crossover_type if crossover_type is not None else ga_settings.crossover_type),
            crossover_rate=crossover_rate if crossover_rate is not None else ga_settings.crossover_rate,
            crossover_alpha=ga_settings.crossover_alpha,
            crossover_eta=ga_settings.crossover_eta,
            mutation_type=MutationType.from_string(mutation_type) if isinstance(mutation_type, str) else (mutation_type if mutation_type is not None else ga_settings.mutation_type),
            mutation_rate=mutation_rate if mutation_rate is not None else ga_settings.mutation_rate,
            mutation_sigma=ga_settings.mutation_sigma,
            mutation_eta=ga_settings.mutation_eta,
            mutation_adaptive=ga_settings.mutation_adaptive,
            selection_type=SelectionType.from_string(selection_type) if isinstance(selection_type, str) else (selection_type if selection_type is not None else ga_settings.selection_type),
            tournament_size=tournament_size if tournament_size is not None else ga_settings.tournament_size,
            truncation_fraction=ga_settings.truncation_fraction,
            elitism_count=elitism_count if elitism_count is not None else ga_settings.elitism_count,
            early_stopping_generations=ga_settings.early_stopping_generations,
            early_stopping_tolerance=ga_settings.early_stopping_tolerance,
            diversity_threshold=ga_settings.diversity_threshold,
            restart_on_stagnation=ga_settings.restart_on_stagnation,
        )
    else:
        # Build from individual params with defaults
        result = GASettings(
            population_size=population_size if population_size is not None else 50,
            max_generations=n_generations if n_generations is not None else 50,
            crossover_type=CrossoverType.from_string(crossover_type) if isinstance(crossover_type, str) else (crossover_type if crossover_type is not None else CrossoverType.BLEND),
            crossover_rate=crossover_rate if crossover_rate is not None else 0.8,
            mutation_type=MutationType.from_string(mutation_type) if isinstance(mutation_type, str) else (mutation_type if mutation_type is not None else MutationType.ADAPTIVE),
            mutation_rate=mutation_rate if mutation_rate is not None else 0.1,
            selection_type=SelectionType.from_string(selection_type) if isinstance(selection_type, str) else (selection_type if selection_type is not None else SelectionType.TOURNAMENT),
            tournament_size=tournament_size if tournament_size is not None else 3,
            elitism_count=elitism_count if elitism_count is not None else 2,
        )
    return result


def _build_pso_settings(
    pso_settings: PSOSettings | None,
    swarm_size: int | None,
    pso_iterations: int | None,
    inertia_strategy: InertiaStrategy | str | None,
    inertia_start: float | None,
    inertia_end: float | None,
    cognitive_coef: float | None,
    social_coef: float | None,
    velocity_clamp: float | None,
    boundary_handling: BoundaryHandling | str | None,
) -> PSOSettings:
    """Build effective PSO settings from various inputs."""
    if pso_settings is not None:
        # Start with provided settings, override with individual params if set
        result = PSOSettings(
            swarm_size=swarm_size if swarm_size is not None else pso_settings.swarm_size,
            max_iterations=pso_iterations if pso_iterations is not None else pso_settings.max_iterations,
            inertia_strategy=InertiaStrategy.from_string(inertia_strategy) if isinstance(inertia_strategy, str) else (inertia_strategy if inertia_strategy is not None else pso_settings.inertia_strategy),
            inertia_start=inertia_start if inertia_start is not None else pso_settings.inertia_start,
            inertia_end=inertia_end if inertia_end is not None else pso_settings.inertia_end,
            inertia_constant=pso_settings.inertia_constant,
            cognitive_coef=cognitive_coef if cognitive_coef is not None else pso_settings.cognitive_coef,
            social_coef=social_coef if social_coef is not None else pso_settings.social_coef,
            velocity_clamp=velocity_clamp if velocity_clamp is not None else pso_settings.velocity_clamp,
            max_velocity=pso_settings.max_velocity,
            boundary_handling=BoundaryHandling.from_string(boundary_handling) if isinstance(boundary_handling, str) else (boundary_handling if boundary_handling is not None else pso_settings.boundary_handling),
            use_constriction=pso_settings.use_constriction,
            local_search_probability=pso_settings.local_search_probability,
            early_stopping_iterations=pso_settings.early_stopping_iterations,
            early_stopping_tolerance=pso_settings.early_stopping_tolerance,
        )
    else:
        # Build from individual params with defaults
        result = PSOSettings(
            swarm_size=swarm_size if swarm_size is not None else 30,
            max_iterations=pso_iterations if pso_iterations is not None else 50,
            inertia_strategy=InertiaStrategy.from_string(inertia_strategy) if isinstance(inertia_strategy, str) else (inertia_strategy if inertia_strategy is not None else InertiaStrategy.LINEAR_DECAY),
            inertia_start=inertia_start if inertia_start is not None else 0.9,
            inertia_end=inertia_end if inertia_end is not None else 0.4,
            cognitive_coef=cognitive_coef if cognitive_coef is not None else 2.0,
            social_coef=social_coef if social_coef is not None else 2.0,
            velocity_clamp=velocity_clamp if velocity_clamp is not None else 0.5,
            boundary_handling=BoundaryHandling.from_string(boundary_handling) if isinstance(boundary_handling, str) else (boundary_handling if boundary_handling is not None else BoundaryHandling.CLAMP),
        )
    return result


def _build_feature_selection_settings(
    feature_selection: bool | FeatureSelectionSettings,
    min_features: int | None,
    max_features: int | None,
    n_features: int,
) -> FeatureSelectionSettings:
    """Build effective feature selection settings."""
    if isinstance(feature_selection, FeatureSelectionSettings):
        result = FeatureSelectionSettings(
            enabled=feature_selection.enabled,
            min_features=min_features if min_features is not None else feature_selection.min_features,
            max_features=max_features if max_features is not None else feature_selection.max_features,
            initial_selection_prob=feature_selection.initial_selection_prob,
            feature_mutation_rate=feature_selection.feature_mutation_rate,
            use_feature_importance=feature_selection.use_feature_importance,
            importance_threshold=feature_selection.importance_threshold,
            grouped_features=feature_selection.grouped_features,
            forbidden_combinations=feature_selection.forbidden_combinations,
        )
    else:
        result = FeatureSelectionSettings(
            enabled=bool(feature_selection),
            min_features=min_features,
            max_features=max_features,
        )
    return result


def _extract_discrete_bounds(
    model_config: dict[str, Any],
) -> tuple[list[tuple[int, int]], list[str]]:
    """Extract discrete parameter bounds from model config."""
    bounds = []
    names = []
    for param_name, param_spec in model_config.get("discrete_params", {}).items():
        if "range" in param_spec:
            low, high = param_spec["range"]
            bounds.append((0, high - low))
        elif "choices" in param_spec:
            bounds.append((0, len(param_spec["choices"]) - 1))
        names.append(param_name)
    return bounds, names


def _extract_continuous_bounds(
    model_config: dict[str, Any],
) -> tuple[list[tuple[float, float]], list[str]]:
    """Extract continuous parameter bounds from model config."""
    bounds = []
    names = []
    for param_name, param_spec in model_config.get("continuous_params", {}).items():
        low, high = param_spec["range"]
        bounds.append((low, high))
        names.append(param_name)
    return bounds, names


def _run_optimization(
    X: NDArray[np.floating],
    y: NDArray,
    model_class: type,
    model_config: dict[str, Any],
    fixed_params: dict[str, Any],
    discrete_bounds: list[tuple[int, int]],
    discrete_param_names: list[str],
    continuous_bounds: list[tuple[float, float]],
    continuous_param_names: list[str],
    n_features: int,
    feature_selection_settings: FeatureSelectionSettings,
    scoring: str,
    cv: int,
    ga_settings: GASettings,
    pso_settings: PSOSettings,
    caching_settings: CachingSettings,
    gpu_config: GPUConfig,
    n_jobs: int,
    random_state: int | None,
    verbose: int,
    refit: bool,
    task: str,
    param_space: CustomParameterSpace | None = None,
) -> OptimizeResult:
    """Run the actual optimization using HybridGAPSOOptimizer.

    Integrates the hybrid GA-PSO optimizer with nested cross-validation
    for robust hyperparameter optimization.
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    from scipy import stats as sp_stats

    start_time = time.time()

    use_gpu = gpu_config.use_gpu if gpu_config else False

    discrete_params_dict: dict[str, dict[str, Any]] = {}
    original_discrete_specs = model_config.get("discrete_params", {})
    for i, name in enumerate(discrete_param_names):
        if i < len(discrete_bounds):
            low, high = discrete_bounds[i]
            original_spec = original_discrete_specs.get(name, {})
            if "choices" in original_spec:
                discrete_params_dict[name] = {"choices": original_spec["choices"]}
            else:
                original_range = original_spec.get("range", (low, high))
                discrete_params_dict[name] = {"range": original_range}

    continuous_params_dict: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(continuous_param_names):
        if i < len(continuous_bounds):
            low, high = continuous_bounds[i]
            continuous_params_dict[name] = {"range": (low, high)}

    nested_model_config = ModelConfig(
        model_class=model_class,
        discrete_params=discrete_params_dict,
        continuous_params=continuous_params_dict,
        fixed_params=fixed_params,
    )

    hybrid_config = HybridConfig(
        ga_config=ga_settings,
        pso_config=pso_settings,
        use_chromosome_cache=caching_settings.use_chromosome_cache if caching_settings else True,
        chromosome_cache_size=caching_settings.chromosome_cache_size if caching_settings else 10000,
        use_full_config_cache=caching_settings.use_full_config_cache if caching_settings else True,
        full_config_cache_size=caching_settings.full_config_cache_size if caching_settings else 100000,
        use_gpu=use_gpu,
        verbose=verbose > 0,
        random_seed=random_state,
    )

    data_manager = CVDataManager(X, y, use_gpu=use_gpu)

    cv_splitter = CVSplitter(
        n_splits=cv,
        stratified=(task == "classification"),
        shuffle=True,
        random_state=random_state,
    )

    inner_evaluator = InnerCVEvaluator(
        data_manager=data_manager,
        cv_splitter=cv_splitter,
        scoring=scoring,
        use_gpu=use_gpu,
        n_jobs=n_jobs,
    )

    fitness_fn = create_cv_fitness_function(inner_evaluator, nested_model_config)

    effective_discrete_bounds = discrete_bounds if discrete_bounds else [(0, 0)]
    effective_continuous_bounds = continuous_bounds if continuous_bounds else [(0.0, 1.0)]

    generation_stats: list[GenerationStats] = []
    try:
        optimizer = HybridGAPSOOptimizer(
            n_features=n_features,
            discrete_bounds=effective_discrete_bounds,
            continuous_bounds=effective_continuous_bounds,
            config=hybrid_config,
        )

        hybrid_result = optimizer.optimize(fitness_fn)  # pyright: ignore[reportArgumentType]

        best_feature_mask = hybrid_result.best_feature_mask
        best_discrete_params = hybrid_result.best_discrete_params
        best_continuous_params = hybrid_result.best_continuous_params
        best_fitness = hybrid_result.best_fitness
        converged = hybrid_result.converged
        n_evaluations = hybrid_result.n_pso_evaluations
        generation_stats = hybrid_result.generation_stats

    except Exception as e:
        logger.warning("Hybrid optimization failed, falling back to baseline: %s", e)
        best_feature_mask = np.ones(n_features, dtype=bool)
        best_discrete_params = np.zeros(len(discrete_bounds) if discrete_bounds else 1, dtype=np.int64)
        best_continuous_params = np.array(
            [(b[0] + b[1]) / 2 for b in continuous_bounds] if continuous_bounds else [0.5],
            dtype=np.float64,
        )
        best_fitness = float("-inf")
        converged = False
        n_evaluations = 0

    best_params = dict(fixed_params)

    discrete_param_specs = model_config.get("discrete_params", {})
    
    for i, name in enumerate(discrete_param_names):
        if i < len(best_discrete_params):
            raw_value = int(best_discrete_params[i])
            param_spec = discrete_param_specs.get(name, {})
            
            if "choices" in param_spec:
                choices = param_spec["choices"]
                if 0 <= raw_value < len(choices):
                    best_params[name] = choices[raw_value]
                else:
                    best_params[name] = choices[0] if choices else raw_value
            elif "range" in param_spec:
                low, _ = param_spec["range"]
                best_params[name] = raw_value + low
            elif i < len(discrete_bounds):
                low, _ = discrete_bounds[i]
                best_params[name] = raw_value + low
            else:
                best_params[name] = raw_value

    for i, name in enumerate(continuous_param_names):
        if i < len(best_continuous_params):
            best_params[name] = float(best_continuous_params[i])

    selected_indices = np.where(best_feature_mask)[0]
    X_selected = X[:, best_feature_mask]

    if task == "classification":
        sklearn_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        sklearn_cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    sklearn_scoring_map = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
        "log_loss": "neg_log_loss",
        "mse": "neg_mean_squared_error",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
        "mape": "neg_mean_absolute_percentage_error",
    }
    sklearn_scorer = sklearn_scoring_map.get(scoring, scoring)

    try:
        final_model = model_class(**best_params)
        fold_scores = cross_val_score(
            final_model, X_selected, y,
            cv=sklearn_cv,
            scoring=sklearn_scorer,
            n_jobs=n_jobs,
        )

        if scoring in ["mse", "rmse", "mae", "log_loss", "mape"]:
            fold_scores = -fold_scores

    except Exception as e:
        logger.warning("Final cross-validation failed: %s", e)
        fold_scores = np.array([best_fitness] * cv)

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))

    alpha = 0.05
    t_stat = sp_stats.t.ppf(1 - alpha / 2, cv - 1)
    ci_half_width = t_stat * std_score / np.sqrt(cv)
    ci_lower = mean_score - ci_half_width
    ci_upper = mean_score + ci_half_width

    best_model = None
    if refit:
        try:
            best_model = model_class(**best_params)
            best_model.fit(X_selected, y)
        except Exception as e:
            logger.warning("Model refit failed: %s", e)

    elapsed_time = time.time() - start_time

    return OptimizeResult(
        mean_score=mean_score,
        std_score=std_score,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        best_params=best_params,
        selected_features=selected_indices,
        feature_mask=best_feature_mask,
        n_features_selected=int(np.sum(best_feature_mask)),
        fold_scores=fold_scores,
        best_model=best_model,
        elapsed_time=elapsed_time,
        n_evaluations=n_evaluations,
        converged=converged,
        gpu_used=use_gpu,
        generation_stats=generation_stats,
    )


def quick_optimize(
    X: NDArray[np.floating],
    y: NDArray,
    *,
    model: ModelType | str = "random_forest",
    preset: PresetType = "balanced",
    task: TaskType = "auto",
    scoring: ScoringType | None = None,
    use_gpu: bool | None = None,
    random_state: int | None = None,
    verbose: int = 1,
) -> OptimizeResult:
    """Quick optimization with preset configurations.

    Simplified interface using predefined optimization presets.

    Args:
        X: Feature matrix.
        y: Target array.
        model: Model type name.
        preset: Optimization preset ('fast', 'balanced', 'thorough',
            'memory_efficient', 'high_accuracy').
        task: Task type ('classification', 'regression', 'auto').
        scoring: Scoring metric (auto-selected if None).
        use_gpu: Use GPU acceleration.
        random_state: Random seed.
        verbose: Verbosity level.

    Returns:
        OptimizeResult with optimization results.

    Example:
        >>> # Fast prototyping
        >>> result = quick_optimize(X, y, preset='fast')
        >>>
        >>> # Thorough search
        >>> result = quick_optimize(X, y, preset='thorough', model='gradient_boosting')
    """
    preset_config = get_preset(preset)

    if scoring is None:
        if task == "auto":
            unique_values = np.unique(y)
            is_classification = len(unique_values) <= 20 and np.all(
                unique_values == unique_values.astype(int)
            )
            task = "classification" if is_classification else "regression"

        scoring = "accuracy" if task == "classification" else "r2"

    effective_use_gpu = use_gpu if use_gpu is not None else preset_config.use_gpu

    return optimize(
        X=X,
        y=y,
        model=model,
        task=task,
        scoring=scoring,
        cv=preset_config.outer_cv,
        n_generations=preset_config.max_generations,
        population_size=preset_config.population_size,
        swarm_size=preset_config.swarm_size,
        pso_iterations=preset_config.pso_iterations,
        elitism_count=max(1, preset_config.population_size // 10),
        use_gpu=effective_use_gpu,
        random_state=random_state,
        verbose=verbose,
    )




def optimize_classifier(
    X: NDArray[np.floating],
    y: NDArray,
    model: ModelType | str = "random_forest",
    scoring: ScoringType = "accuracy",
    **kwargs: Any,
) -> OptimizeResult:
    """Optimize a classification model.

    Convenience function with classification defaults.

    Args:
        X: Feature matrix.
        y: Target array.
        model: Model type name.
        scoring: Scoring metric.
        **kwargs: Additional arguments for optimize().

    Returns:
        OptimizeResult.

    Example:
        >>> result = optimize_classifier(X, y, model='random_forest', scoring='roc_auc')
    """
    return optimize(X, y, model=model, task="classification", scoring=scoring, **kwargs)


def optimize_regressor(
    X: NDArray[np.floating],
    y: NDArray,
    model: ModelType | str = "random_forest",
    scoring: ScoringType = "r2",
    **kwargs: Any,
) -> OptimizeResult:
    """Optimize a regression model.

    Convenience function with regression defaults.

    Args:
        X: Feature matrix.
        y: Target array.
        model: Model type name.
        scoring: Scoring metric.
        **kwargs: Additional arguments for optimize().

    Returns:
        OptimizeResult.

    Example:
        >>> result = optimize_regressor(X, y, model='gradient_boosting', scoring='mse')
    """
    return optimize(X, y, model=model, task="regression", scoring=scoring, **kwargs)


def list_available_models(task: TaskType = "classification") -> list[str]:
    """List available models for a task type.

    Args:
        task: Task type.

    Returns:
        List of model names.

    Example:
        >>> models = list_available_models('classification')
        >>> print(models)
    """
    return ModelRegistry.list_models(task)


def list_available_presets() -> list[str]:
    """List available optimization presets.

    Returns:
        List of preset names.

    Example:
        >>> presets = list_available_presets()
        >>> for name in presets:
        ...     preset = get_preset(name)
        ...     print(f"{name}: {preset.description}")
    """
    return list(PRESETS.keys())


def describe_preset(name: PresetType) -> str:
    """Get description of an optimization preset.

    Args:
        name: Preset name.

    Returns:
        Description string.
    """
    preset = get_preset(name)
    lines = [
        f"Preset: {preset.name}",
        f"Description: {preset.description}",
        f"Population Size: {preset.population_size}",
        f"Max Generations: {preset.max_generations}",
        f"Swarm Size: {preset.swarm_size}",
        f"PSO Iterations: {preset.pso_iterations}",
        f"Outer CV: {preset.outer_cv}",
        f"Inner CV: {preset.inner_cv}",
        f"Early Stopping: {preset.early_stopping_generations} generations",
        f"GPU Enabled: {preset.use_gpu}",
    ]
    return "\n".join(lines)




def optimize_with_cv(
    X: NDArray[np.floating],
    y: NDArray,
    model_class: type,
    discrete_params: dict[str, dict[str, Any]],
    continuous_params: dict[str, dict[str, Any]],
    fixed_params: dict[str, Any] | None = None,
    *,
    outer_cv: int = 5,
    inner_cv: int = 3,
    scoring: str = "accuracy",
    task: str = "classification",
    ga_settings: GASettings | None = None,
    pso_settings: PSOSettings | None = None,
    use_gpu: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
    verbose: bool = True,
    refit: bool = True,
) -> NestedCVResult:
    """Run full nested cross-validation optimization.

    This function provides direct access to the NestedCVOptimizer for
    maximum control over the optimization process.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target array of shape (n_samples,).
        model_class: Model class to optimize.
        discrete_params: Dictionary of discrete parameter specifications.
            Format: {'param_name': {'range': (min, max)} or {'choices': [...]}}
        continuous_params: Dictionary of continuous parameter specifications.
            Format: {'param_name': {'range': (min, max)}}
        fixed_params: Fixed parameters for the model.
        outer_cv: Number of outer CV folds.
        inner_cv: Number of inner CV folds.
        scoring: Scoring metric name.
        task: Task type ('classification' or 'regression').
        ga_settings: GA configuration (uses defaults if None).
        pso_settings: PSO configuration (uses defaults if None).
        use_gpu: Whether to use GPU acceleration.
        n_jobs: Number of parallel jobs (-1 for all cores).
        random_state: Random seed for reproducibility.
        verbose: Print progress information.
        refit: Refit best model on all data.

    Returns:
        NestedCVResult with detailed optimization results.

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> result = optimize_with_cv(
        ...     X, y,
        ...     model_class=RandomForestClassifier,
        ...     discrete_params={
        ...         'n_estimators': {'range': (50, 500)},
        ...         'max_depth': {'range': (3, 30)},
        ...     },
        ...     continuous_params={
        ...         'max_features': {'range': (0.1, 1.0)},
        ...     },
        ...     fixed_params={'n_jobs': -1},
        ...     outer_cv=5,
        ...     inner_cv=3,
        ... )
    """
    model_config = ModelConfig(
        model_class=model_class,
        discrete_params=discrete_params,
        continuous_params=continuous_params,
        fixed_params=fixed_params or {},
    )

    cv_config = NestedCVConfig(
        outer_cv=outer_cv,
        inner_cv=inner_cv,
        stratified=(task == "classification"),
        shuffle=True,
        scoring=scoring,
        use_gpu=use_gpu,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=refit,
    )

    hybrid_config = HybridConfig(
        ga_config=ga_settings or GASettings(),
        pso_config=pso_settings or PSOSettings(),
        use_gpu=use_gpu,
        verbose=verbose,
        random_seed=random_state,
    )

    optimizer = NestedCVOptimizer(config=cv_config)
    result = optimizer.optimize(X, y, model_config, hybrid_config)

    return result


def optimize_multi_model(
    X: NDArray[np.floating],
    y: NDArray,
    model_names: list[str] | None = None,
    *,
    task: TaskType = "classification",
    scoring: str = "accuracy",
    cv: int = 5,
    ga_settings: GASettings | None = None,
    pso_settings: PSOSettings | None = None,
    use_gpu: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
    verbose: bool = True,
) -> dict[str, OptimizeResult]:
    """Optimize multiple models and compare their performance.

    Uses MultiModelRegistry and MultiModelEvaluator to evaluate multiple
    model types simultaneously and find the best overall configuration.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target array of shape (n_samples,).
        model_names: List of model names to optimize. If None, uses defaults.
        task: Task type ('classification' or 'regression').
        scoring: Scoring metric name.
        cv: Number of cross-validation folds.
        ga_settings: GA configuration.
        pso_settings: PSO configuration.
        use_gpu: Whether to use GPU acceleration.
        n_jobs: Number of parallel jobs.
        random_state: Random seed for reproducibility.
        verbose: Print progress information.

    Returns:
        Dictionary mapping model names to their OptimizeResult.

    Example:
        >>> results = optimize_multi_model(
        ...     X, y,
        ...     model_names=['random_forest', 'gradient_boosting', 'svm'],
        ...     scoring='accuracy',
        ... )
        >>> best_model = max(results.items(), key=lambda x: x[1].mean_score)
        >>> print(f"Best model: {best_model[0]} with score {best_model[1].mean_score:.4f}")
    """
    from .multi_model import (
        ModelRegistry as MultiModelReg,
        MultiModelEvaluator,
        MultiModelPopulation,
        MultiModelChromosome,
    )
    from sklearn.model_selection import StratifiedKFold, KFold

    if model_names is None:
        if task == "classification":
            model_names = ["random_forest", "gradient_boosting", "logistic_regression"]
        else:
            model_names = ["random_forest", "gradient_boosting", "ridge"]

    task_type_enum = (
        CommonTaskType.CLASSIFICATION
        if task == "classification"
        else CommonTaskType.REGRESSION
    )
    registry = ModelRegistry.create_multi_model_registry(model_names, task)

    evaluator = MultiModelEvaluator(
        registry=registry,
        scoring=scoring,
        use_gpu=use_gpu,
    )

    if task == "classification":
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    cv_splits = list(cv_splitter.split(X, y))

    results: dict[str, OptimizeResult] = {}

    for model_idx, model_name in enumerate(model_names):
        if verbose:
            logger.info("Optimizing model: %s", model_name)

        try:
            model_def = registry.get_model(model_idx)
            
            n_features = X.shape[1]
            baseline_chromosome = MultiModelChromosome(  # pyright: ignore[reportCallIssue]
                model_index=model_idx,
                feature_mask=np.ones(n_features, dtype=bool),
                discrete_values=np.zeros(registry.total_discrete_params, dtype=np.int64),  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
                continuous_values=np.zeros(registry.total_continuous_params, dtype=np.float64),  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
            )
            
            baseline_score = evaluator.evaluate_individual(
                baseline_chromosome, X, y, cv_splits=cv_splits
            )
            
            if verbose:
                logger.info("  Baseline score for %s: %.4f", model_name, baseline_score)

            result = optimize(
                X, y,
                model=model_name,
                task=task,
                scoring=scoring,  # pyright: ignore[reportArgumentType]
                cv=cv,
                ga_settings=ga_settings,
                pso_settings=pso_settings,
                use_gpu=use_gpu,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose > 1 if isinstance(verbose, int) else False,
            )
            results[model_name] = result

            if verbose:
                improvement = result.mean_score - baseline_score if np.isfinite(baseline_score) else 0
                logger.info(
                    "  %s: score=%.4f Â± %.4f (improvement: %+.4f)",
                    model_name,
                    result.mean_score,
                    result.std_score,
                    improvement,
                )
        except Exception as e:
            logger.warning("Failed to optimize %s: %s", model_name, e)

    if results and verbose:
        best_name = max(results.keys(), key=lambda k: results[k].mean_score)
        best_result = results[best_name]
        logger.info(
            "Best model: %s (score=%.4f Â± %.4f)",
            best_name,
            best_result.mean_score,
            best_result.std_score,
        )

    return results


def compute_feature_importance(
    result: OptimizeResult,
    X: NDArray[np.floating],
    y: NDArray,
    method: str = "permutation",
    n_repeats: int = 10,
    random_state: int | None = None,
) -> FeatureImportance:
    """Compute feature importance from optimization result.

    Uses FeatureImportance from results.py to analyze which features
    are most important for the optimized model.

    Args:
        result: OptimizeResult from optimization.
        X: Feature matrix.
        y: Target array.
        method: Importance method ('permutation', 'model', 'shap').
        n_repeats: Number of repeats for permutation importance.
        random_state: Random seed.

    Returns:
        FeatureImportance with importance scores.

    Example:
        >>> result = optimize(X, y, model='random_forest')
        >>> importance = compute_feature_importance(result, X, y)
        >>> print(importance.top_features(10))
    """
    from .results import FeatureImportance

    n_features = X.shape[1]

    if result.best_model is None:
        raise ValueError("Result must have a fitted model (use refit=True)")

    model = result.best_model
    X_selected = X[:, result.feature_mask]

    if method == "model" and hasattr(model, "feature_importances_"):
        raw_importance = model.feature_importances_
        importance_scores = np.zeros(n_features)
        importance_scores[result.feature_mask] = raw_importance
        std_scores = np.zeros(n_features)

    elif method == "permutation":
        from sklearn.inspection import permutation_importance

        perm_result = permutation_importance(
            model, X_selected, y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )
        importance_scores = np.zeros(n_features)
        std_scores = np.zeros(n_features)
        importance_scores[result.feature_mask] = perm_result.importances_mean  # pyright: ignore[reportAttributeAccessIssue]
        std_scores[result.feature_mask] = perm_result.importances_std  # pyright: ignore[reportAttributeAccessIssue]

    else:
        importance_scores = result.feature_mask.astype(float)
        std_scores = np.zeros(n_features)

    return FeatureImportance(  # pyright: ignore[reportCallIssue]
        feature_indices=np.arange(n_features),
        importance_scores=importance_scores,  # pyright: ignore[reportCallIssue]
        std_scores=std_scores,  # pyright: ignore[reportCallIssue]
        method=method,  # pyright: ignore[reportCallIssue]
    )


def create_optimizer(
    n_features: int,
    discrete_bounds: list[tuple[int, int]],
    continuous_bounds: list[tuple[float, float]],
    *,
    ga_settings: GASettings | None = None,
    pso_settings: PSOSettings | None = None,
    use_gpu: bool = False,
    use_caching: bool = True,
    random_state: int | None = None,
    verbose: bool = True,
) -> HybridGAPSOOptimizer:
    """Create a configured HybridGAPSOOptimizer for custom pipelines.

    This factory function provides an easy way to create an optimizer
    with custom configuration for use in custom optimization pipelines.

    Args:
        n_features: Number of features for feature selection.
        discrete_bounds: Bounds for discrete parameters [(min, max), ...].
        continuous_bounds: Bounds for continuous parameters [(min, max), ...].
        ga_settings: GA configuration (uses defaults if None).
        pso_settings: PSO configuration (uses defaults if None).
        use_gpu: Whether to use GPU acceleration.
        use_caching: Whether to enable result caching.
        random_state: Random seed for reproducibility.
        verbose: Print progress information.

    Returns:
        Configured HybridGAPSOOptimizer instance.

    Example:
        >>> optimizer = create_optimizer(
        ...     n_features=50,
        ...     discrete_bounds=[(0, 10), (0, 5)],
        ...     continuous_bounds=[(0.0, 1.0), (0.001, 0.1)],
        ...     use_gpu=True,
        ... )
        >>> result = optimizer.optimize(my_fitness_function)
    """
    config = HybridConfig(
        ga_config=ga_settings or GASettings(),
        pso_config=pso_settings or PSOSettings(),
        use_chromosome_cache=use_caching,
        use_full_config_cache=use_caching,
        use_gpu=use_gpu,
        verbose=verbose,
        random_seed=random_state,
    )

    return HybridGAPSOOptimizer(
        n_features=n_features,
        discrete_bounds=discrete_bounds if discrete_bounds else [(0, 0)],
        continuous_bounds=continuous_bounds if continuous_bounds else [(0.0, 1.0)],
        config=config,
    )


def create_evaluator(
    scoring: str = "accuracy",
    use_gpu: bool = False,
    min_samples_for_gpu: int = 1000,
    cache_predictions: bool = True,
) -> EvaluationPipeline:
    """Create an EvaluationPipeline for custom model evaluation.

    Args:
        scoring: Scoring metric name.
        use_gpu: Whether to use GPU acceleration.
        min_samples_for_gpu: Minimum samples to use GPU scoring.
        cache_predictions: Whether to cache predictions.

    Returns:
        Configured EvaluationPipeline instance.

    Example:
        >>> evaluator = create_evaluator(scoring='roc_auc', use_gpu=True)
        >>> result = evaluator.evaluate(model, X_train, y_train, X_val, y_val)
    """
    config = PipelineConfig(
        use_gpu=use_gpu,
        scoring=scoring,
        min_samples_for_gpu=min_samples_for_gpu,
        cache_predictions=cache_predictions,
    )

    return EvaluationPipeline(config=config)


def get_gpu_memory_manager(
    memory_limit: float = 0.8,
    pool_size_mb: int = 512,
) -> GPUMemoryManager:
    """Get a GPUMemoryManager for manual GPU memory control.

    Args:
        memory_limit: Fraction of GPU memory to use (0.0-1.0).
        pool_size_mb: Memory pool size in MB.

    Returns:
        GPUMemoryManager instance.

    Example:
        >>> manager = get_gpu_memory_manager(memory_limit=0.7)
        >>> with manager.managed_allocation(size_bytes=1000000):
        ...     # Do GPU work
        ...     pass
    """
    return create_memory_manager(
        memory_limit=memory_limit,
        pool_size_mb=pool_size_mb,
    )


def profile_optimization(
    X: NDArray[np.floating],
    y: NDArray,
    model: ModelType | str = "random_forest",
    *,
    preset: PresetType = "fast",
    **kwargs: Any,
) -> tuple[OptimizeResult, dict[str, Any]]:
    """Run optimization with performance profiling.

    Runs the optimization while collecting detailed performance metrics
    including timing, memory usage, and GPU utilization.

    Args:
        X: Feature matrix.
        y: Target array.
        model: Model type name.
        preset: Optimization preset to use.
        **kwargs: Additional arguments for optimize().

    Returns:
        Tuple of (OptimizeResult, profiling_stats).

    Example:
        >>> result, stats = profile_optimization(X, y, model='random_forest')
        >>> print(f"Total time: {stats['total_time']:.2f}s")
        >>> print(f"Peak memory: {stats['peak_memory_mb']:.1f} MB")
    """
    profiler = PerformanceProfiler()

    profiler.start("total_optimization")

    get_preset(preset)

    with performance_context(name="quick_optimize", enable_gpu_sync=True) as ctx:
        result = quick_optimize(
            X, y,
            model=model,
            preset=preset,
            **kwargs,
        )

    profiler.stop("total_optimization")

    timing_stats = profiler.get_stats()
    ctx_stats = ctx.get_stats()
    stats = {
        "total_time": timing_stats.get("total_optimization", {}).get("total", 0.0),
        "quick_optimize_time": ctx_stats.get("quick_optimize", {}).get("total", 0.0),
        "gpu_used": result.gpu_used,
        "n_evaluations": result.n_evaluations,
        "converged": result.converged,
        "timing_breakdown": timing_stats,
    }

    return result, stats


def create_batch_evaluator(
    X: NDArray[np.floating],
    y: NDArray,
    scoring: str = "accuracy",
    cv: int = 5,
    use_gpu: bool = False,
    n_jobs: int = -1,
    task: str = "classification",
    random_state: int | None = None,
) -> tuple[BatchEvaluator, list[tuple[NDArray[np.intp], NDArray[np.intp]]]]:
    """Create a BatchEvaluator for efficient parallel model evaluation.

    Args:
        X: Feature matrix.
        y: Target array.
        scoring: Scoring metric name.
        cv: Number of CV folds.
        use_gpu: Whether to use GPU acceleration.
        n_jobs: Number of parallel jobs.
        task: Task type ('classification' or 'regression').
        random_state: Random seed.

    Returns:
        Tuple of (BatchEvaluator instance, cv_splits) for use in evaluation.

    Example:
        >>> evaluator, cv_splits = create_batch_evaluator(X, y, scoring='accuracy')
        >>> results = evaluator.evaluate_batch(model_builders, X, y, cv_splits)
    """
    from .evaluation import PipelineConfig, BatchEvaluator, create_cv_splits

    config = PipelineConfig(
        use_gpu=use_gpu,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    stratified = task == "classification"
    cv_splits = create_cv_splits(
        n_samples=len(y),
        n_splits=cv,
        y=y if stratified else None,
        stratified=stratified,
        shuffle=True,
        random_state=random_state,
    )

    evaluator = BatchEvaluator(config=config)

    return evaluator, cv_splits




__all__ = [
    "optimize",
    "quick_optimize",
    "optimize_classifier",
    "optimize_regressor",
    "optimize_with_cv",
    "optimize_multi_model",
    "compute_feature_importance",
    "create_optimizer",
    "create_evaluator",
    "create_batch_evaluator",
    "get_gpu_memory_manager",
    "profile_optimization",
    "OptimizeResult",
    "HybridResult",
    "NestedCVResult",
    "GenerationStats",
    "FeatureImportance",
    "HybridGAPSOOptimizer",
    "HybridConfig",
    "NestedCVOptimizer",
    "NestedCVConfig",
    "EvaluationPipeline",
    "PipelineConfig",
    "BatchEvaluator",
    "GPUScorer",
    "create_cv_splits",
    "GPUMemoryManager",
    "PerformanceProfiler",
    "TransferOptimizer",
    "performance_context",
    "GPUConfig",
    "gpu_info",
    "gpu_context",
    "cpu_context",
    "GASettings",
    "CrossoverType",
    "MutationType",
    "SelectionType",
    "PSOSettings",
    "InertiaStrategy",
    "BoundaryHandling",
    "FeatureSelectionSettings",
    "CustomParameterSpace",
    "ParameterSpec",
    "CachingSettings",
    "InitializationMethod",
    "get_scorer",
    "SCORING_FUNCTIONS",
    "needs_probability_predictions",
    "list_available_models",
    "list_available_presets",
    "describe_preset",
    "get_preset",
    "estimate_problem_memory",
    "validate_gpu_setup",
    "ModelRegistry",
]
