# AI use disclaimer

Large language models have been used in the development of this application, for fast-prototyping and error-checking. The readme file has been generated using LLMs, and manually reviewed and edited.

# Hybrid GA-PSO Hyperparameter Optimization Framework

A GPU-accelerated hybrid optimization framework combining Genetic Algorithms (GA) and Particle Swarm Optimization (PSO) for simultaneous hyperparameter tuning and feature selection in machine learning models.

## Overview

This framework provides an approach to automated machine learning by jointly optimizing model hyperparameters and feature subsets. The hybrid algorithm leverages GA for discrete/combinatorial optimization (feature selection, categorical parameters) and PSO for continuous parameter optimization, with optional GPU acceleration for large-scale problems.

## Features

- **Hybrid GA-PSO Optimization**: Combines the global exploration of genetic algorithms with the local refinement of particle swarm optimization
- **GPU Acceleration**: Optional CUDA support via CuPy and cuML for speedups on large datasets
- **Nested Cross-Validation**:  Model evaluation with inner/outer CV loops to prevent overfitting
- **Automatic Feature Selection**: Simultaneous optimization of feature subsets alongside hyperparameters
- **Extensive Model Support**: Built-in support for scikit-learn classifiers and regressors, with custom model compatibility
- **Flexible Configuration**:  Control over GA operators, PSO dynamics, and optimization behavior
- **Performance Profiling**: Built-in tools for memory management and execution profiling


## Quick Start

### Basic Usage

```python
from api import optimize
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=50, random_state=42)

result = optimize(
    X, y,
    model='random_forest',
    scoring='accuracy',
    cv=5
)

print(f"Score: {result.mean_score:.4f} +/- {result.std_score:.4f}")
print(f"Features selected: {result.n_features_selected}")
print(f"Best parameters: {result.best_params}")
```

### Using Presets

```python
from api import quick_optimize

# Available presets: 'fast', 'balanced', 'thorough', 'memory_efficient', 'high_accuracy'
result = quick_optimize(X, y, model='gradient_boosting', preset='balanced')
```

### Custom Configuration

```python
from api import (
    optimize,
    GASettings,
    PSOSettings,
    FeatureSelectionSettings,
    CustomParameterSpace,
    CrossoverType,
    MutationType,
    InertiaStrategy,
)

# Define custom parameter space
space = CustomParameterSpace()
space.add_integer('n_estimators', 50, 500)
space.add_integer('max_depth', 2, 20)
space.add_continuous('learning_rate', 0.01, 0.3, log_scale=True)
space.add_categorical('loss', ['log_loss', 'exponential'])
space.set_fixed(random_state=42)

# Configure GA
ga_settings = GASettings(
    population_size=100,
    max_generations=50,
    crossover_type=CrossoverType.SIMULATED_BINARY,
    crossover_rate=0.9,
    mutation_type=MutationType.POLYNOMIAL,
    elitism_count=3,
)

# Configure PSO
pso_settings = PSOSettings(
    swarm_size=50,
    max_iterations=75,
    inertia_strategy=InertiaStrategy.LINEAR_DECAY,
    inertia_start=0.9,
    inertia_end=0.4,
    cognitive_coef=2.5,
    social_coef=1.8,
)

# Configure feature selection
fs_settings = FeatureSelectionSettings(
    enabled=True,
    min_features=5,
    max_features=30,
)

result = optimize(
    X, y,
    model=GradientBoostingClassifier,
    param_space=space,
    ga_settings=ga_settings,
    pso_settings=pso_settings,
    feature_selection=fs_settings,
    scoring='roc_auc',
    cv=5,
)
```

## Supported Models

### Classification

- `random_forest` - Random Forest Classifier
- `gradient_boosting` - Gradient Boosting Classifier
- `xgboost` - XGBoost Classifier
- `lightgbm` - LightGBM Classifier
- `svm` - Support Vector Classifier
- `logistic_regression` - Logistic Regression
- `knn` - K-Nearest Neighbors
- `mlp` - Multi-Layer Perceptron
- `extra_trees` - Extra Trees Classifier

### Regression

- `random_forest` - Random Forest Regressor
- `gradient_boosting` - Gradient Boosting Regressor
- `xgboost` - XGBoost Regressor
- `lightgbm` - LightGBM Regressor
- `svm` - Support Vector Regressor
- `ridge` - Ridge Regression
- `lasso` - Lasso Regression
- `elastic_net` - Elastic Net
- `knn` - K-Nearest Neighbors Regressor
- `mlp` - Multi-Layer Perceptron Regressor

## Scoring Metrics

### Classification

| Metric | Description |
|--------|-------------|
| `accuracy` | Classification accuracy |
| `balanced_accuracy` | Balanced accuracy for imbalanced datasets |
| `f1` | F1 score (binary) |
| `f1_macro` | Macro-averaged F1 |
| `f1_weighted` | Weighted F1 |
| `precision` | Precision score |
| `recall` | Recall score |
| `roc_auc` | Area under ROC curve |
| `log_loss` | Logarithmic loss |

### Regression

| Metric | Description |
|--------|-------------|
| `r2` | R-squared coefficient |
| `mse` | Mean squared error |
| `rmse` | Root mean squared error |
| `mae` | Mean absolute error |
| `mape` | Mean absolute percentage error |

## GPU Configuration

```python
from api import GPUConfig, optimize

# Automatic GPU detection
config = GPUConfig.auto()

# CPU only
config = GPUConfig.cpu_only()

# High performance GPU settings
config = GPUConfig.high_performance()

# Memory efficient GPU settings
config = GPUConfig.memory_efficient()

# Custom configuration
config = GPUConfig(
    use_gpu=True,
    device_id=0,
    memory_limit=0.8,
    fallback_to_cpu=True,
)

result = optimize(X, y, model='random_forest', gpu_config=config)
```

## API Reference

### Main Functions

| Function | Description |
|----------|-------------|
| `optimize()` | Full optimization with all configuration options |
| `quick_optimize()` | Preset-based optimization for common scenarios |
| `optimize_classifier()` | Optimized for classification tasks |
| `optimize_regressor()` | Optimized for regression tasks |
| `optimize_with_cv()` | Full nested cross-validation optimization |
| `optimize_multi_model()` | Compare multiple model types |

### Configuration Classes

| Class | Description |
|-------|-------------|
| `GASettings` | Genetic algorithm configuration |
| `PSOSettings` | Particle swarm optimization configuration |
| `FeatureSelectionSettings` | Feature selection constraints |
| `CustomParameterSpace` | Custom hyperparameter search space |
| `GPUConfig` | GPU acceleration settings |

### Result Classes

| Class | Description |
|-------|-------------|
| `OptimizeResult` | Results from optimize() |
| `HybridResult` | Results from hybrid optimizer |
| `NestedCVResult` | Results from nested cross-validation |

## Project Structure

```
.
├── api.py                 # High-level API
├── hybrid_optimizer.py    # Hybrid GA-PSO implementation
├── genetoc_algorithm.py         # GPU-accelerated genetic algorithm
├── particle_swarm.py      # GPU-accelerated PSO
├── nested_cv.py           # Nested cross-validation
├── evaluation.py          # Model evaluation pipeline
├── optimization_config.py # Configuration classes
├── gpu_config.py          # GPU configuration
├── performance.py         # Performance profiling
├── model_space.py         # Model parameter spaces
├── parameter_space.py     # Parameter space definitions
├── multi_model.py         # Multi-model optimization
├── results.py             # Result data structures
├── scoring.py             # Scoring metrics
├── backend.py             # NumPy/CuPy backend abstraction
├── data.py                # Data handling utilities
├── config.py              # Global configuration
├── common.py              # Common utilities
├── utils.py               # Helper functions
└── examples.py            # Usage examples
```

## Algorithm Details

### Hybrid Optimization Strategy

1. **Initialization**: Population initialized using Latin Hypercube Sampling or Sobol sequences for better coverage
2. **GA Phase**: Evolves discrete parameters and feature masks using selection, crossover, and mutation
3. **PSO Phase**: Refines continuous parameters using particle swarm dynamics
4. **Warm-Starting**: PSO is warm-started with elite solutions from GA
5. **Two-Level Caching**: Reduces redundant evaluations across both optimization levels

### Genetic Algorithm Operators

- **Selection**: Tournament, Roulette, Rank, Truncation, Elitist, Stochastic Universal Sampling
- **Crossover**: Uniform, Single-Point, Two-Point, Blend, Simulated Binary (SBX), Arithmetic
- **Mutation**: Uniform, Gaussian, Polynomial, Adaptive, Boundary
