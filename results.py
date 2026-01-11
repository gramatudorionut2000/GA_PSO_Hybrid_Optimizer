from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .utils.common import (
    ensure_numpy,

)

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)




@dataclass
class GenerationStats:
    """Statistics for a single generation.

    Attributes:
        generation: Generation number.
        best_fitness: Best fitness in generation.
        mean_fitness: Mean fitness of population.
        std_fitness: Standard deviation of fitness.
        median_fitness: Median fitness.
        min_fitness: Minimum fitness.
        diversity: Population diversity metric.
        n_evaluated: Number of individuals evaluated.
        elapsed_time: Time for this generation.
    """

    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    median_fitness: float
    min_fitness: float
    diversity: float = 0.0
    n_evaluated: int = 0
    elapsed_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "std_fitness": self.std_fitness,
            "median_fitness": self.median_fitness,
            "min_fitness": self.min_fitness,
            "diversity": self.diversity,
            "n_evaluated": self.n_evaluated,
            "elapsed_time": self.elapsed_time,
        }


@dataclass
class FeatureImportance:
    """Feature importance analysis results.

    Attributes:
        feature_indices: Indices of features.
        selection_frequency: Selection frequency for each feature.
        final_selection: Whether selected in final solution.
        coselection_scores: Mean co-selection with other features.
        importance_rank: Rank by importance (1 = most important).
    """

    feature_indices: NDArray[np.intp]
    selection_frequency: NDArray[np.float64]
    final_selection: NDArray[np.bool_]
    coselection_scores: NDArray[np.float64]
    importance_rank: NDArray[np.intp]

    def get_top_features(self, n: int = 10) -> NDArray[np.intp]:
        """Get indices of top N features by importance."""
        sorted_idx = np.argsort(self.importance_rank)
        return self.feature_indices[sorted_idx[:n]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_indices": self.feature_indices.tolist(),
            "selection_frequency": self.selection_frequency.tolist(),
            "final_selection": self.final_selection.tolist(),
            "coselection_scores": self.coselection_scores.tolist(),
            "importance_rank": self.importance_rank.tolist(),
        }


@dataclass
class OptimizationResults:
    """Comprehensive optimization results.

    Attributes:
        best_feature_mask: Boolean mask for selected features.
        best_discrete_params: Best discrete parameter values.
        best_continuous_params: Best continuous parameter values.
        best_fitness: Best fitness achieved.
        n_generations: Number of generations run.
        n_evaluations: Total fitness evaluations.
        converged: Whether optimization converged.
        elapsed_time: Total elapsed time.
        generation_history: Stats for each generation.
        feature_importance: Feature importance analysis.
        parameter_names: Names for discrete/continuous parameters.
        cache_stats: Cache statistics.
    """

    best_feature_mask: NDArray[np.bool_]
    best_discrete_params: NDArray[np.int64]
    best_continuous_params: NDArray[np.float64]
    best_fitness: float
    n_generations: int
    n_evaluations: int
    converged: bool
    elapsed_time: float
    generation_history: list[GenerationStats] = field(default_factory=list)
    feature_importance: FeatureImportance | None = None
    parameter_names: dict[str, list[str]] = field(default_factory=dict)
    cache_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features_selected(self) -> int:
        """Number of features selected."""
        return int(np.sum(self.best_feature_mask))

    @property
    def selected_feature_indices(self) -> NDArray[np.intp]:
        """Indices of selected features."""
        return np.where(self.best_feature_mask)[0]

    def get_final_fitness_improvement(self) -> float:
        """Get fitness improvement from first to last generation."""
        if not self.generation_history:
            return 0.0
        first = self.generation_history[0].best_fitness
        last = self.generation_history[-1].best_fitness
        return last - first

    def get_convergence_generation(self, threshold: float = 0.001) -> int | None:
        """Find generation where convergence occurred."""
        if len(self.generation_history) < 2:
            return None

        best_so_far = self.generation_history[0].best_fitness
        for gen_stats in self.generation_history[1:]:
            improvement = gen_stats.best_fitness - best_so_far
            if improvement > threshold:
                best_so_far = gen_stats.best_fitness
            else:
                return gen_stats.generation
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "best_feature_mask": self.best_feature_mask.tolist(),
            "best_discrete_params": self.best_discrete_params.tolist(),
            "best_continuous_params": self.best_continuous_params.tolist(),
            "best_fitness": self.best_fitness,
            "n_generations": self.n_generations,
            "n_evaluations": self.n_evaluations,
            "converged": self.converged,
            "elapsed_time": self.elapsed_time,
            "n_features_selected": self.n_features_selected,
            "selected_feature_indices": self.selected_feature_indices.tolist(),
            "generation_history": [g.to_dict() for g in self.generation_history],
            "feature_importance": (
                self.feature_importance.to_dict() if self.feature_importance else None
            ),
            "parameter_names": self.parameter_names,
            "cache_stats": self.cache_stats,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationResults":
        """Create from dictionary."""
        generation_history = [
            GenerationStats(**g) for g in data.get("generation_history", [])
        ]

        feature_importance = None
        if data.get("feature_importance"):
            fi_data = data["feature_importance"]
            feature_importance = FeatureImportance(
                feature_indices=np.array(fi_data["feature_indices"], dtype=np.intp),
                selection_frequency=np.array(
                    fi_data["selection_frequency"], dtype=np.float64
                ),
                final_selection=np.array(fi_data["final_selection"], dtype=np.bool_),
                coselection_scores=np.array(
                    fi_data["coselection_scores"], dtype=np.float64
                ),
                importance_rank=np.array(fi_data["importance_rank"], dtype=np.intp),
            )

        return cls(
            best_feature_mask=np.array(data["best_feature_mask"], dtype=np.bool_),
            best_discrete_params=np.array(data["best_discrete_params"], dtype=np.int64),
            best_continuous_params=np.array(
                data["best_continuous_params"], dtype=np.float64
            ),
            best_fitness=data["best_fitness"],
            n_generations=data["n_generations"],
            n_evaluations=data["n_evaluations"],
            converged=data["converged"],
            elapsed_time=data["elapsed_time"],
            generation_history=generation_history,
            feature_importance=feature_importance,
            parameter_names=data.get("parameter_names", {}),
            cache_stats=data.get("cache_stats", {}),
        )


@dataclass
class FoldResult:
    """Results for a single CV fold."""

    fold_idx: int
    inner_result: OptimizationResults
    outer_score: float
    n_train_samples: int
    n_test_samples: int
    elapsed_time: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fold_idx": self.fold_idx,
            "inner_result": self.inner_result.to_dict(),
            "outer_score": self.outer_score,
            "n_train_samples": self.n_train_samples,
            "n_test_samples": self.n_test_samples,
            "elapsed_time": self.elapsed_time,
        }


@dataclass
class NestedCVResults:
    """Results from nested cross-validation."""

    fold_results: list[FoldResult]
    mean_score: float
    std_score: float
    ci_lower: float
    ci_upper: float
    feature_stability: NDArray[np.float64]
    total_elapsed_time: float
    scoring_metric: str

    @property
    def n_folds(self) -> int:
        """Number of folds."""
        return len(self.fold_results)

    @property
    def outer_scores(self) -> NDArray[np.float64]:
        """Array of outer fold scores."""
        return np.array([f.outer_score for f in self.fold_results])

    def get_best_fold(self) -> FoldResult:
        """Get fold with best outer score."""
        scores = self.outer_scores
        best_idx = int(np.argmax(scores))
        return self.fold_results[best_idx]

    def get_consistent_features(self, threshold: float = 0.8) -> NDArray[np.intp]:
        """Get features selected in at least threshold fraction of folds."""
        return np.where(self.feature_stability >= threshold)[0]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fold_results": [f.to_dict() for f in self.fold_results],
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "feature_stability": self.feature_stability.tolist(),
            "total_elapsed_time": self.total_elapsed_time,
            "scoring_metric": self.scoring_metric,
            "n_folds": self.n_folds,
            "outer_scores": self.outer_scores.tolist(),
        }



class FeatureAnalyzer:
    """GPU-assisted feature selection analysis."""

    def __init__(self, use_gpu: bool = False) -> None:
        """Initialize feature analyzer."""
        self.use_gpu = use_gpu and self._check_gpu()

    @staticmethod
    def _check_gpu() -> bool:
        """Check GPU availability."""
        try:
            import cupy as cp
            _ = cp.array([1, 2, 3])
            return True
        except (ImportError, Exception):
            return False

    def _get_array_module(self) -> Any:
        """Get appropriate array module."""
        if self.use_gpu:
            try:
                import cupy as cp
                return cp
            except ImportError:
                pass
        return np

    def compute_selection_frequency(
        self,
        feature_history: NDArray[np.bool_] | list[NDArray[np.bool_]],
        weights: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute feature selection frequency across generations/individuals."""
        if isinstance(feature_history, list):
            all_genes = np.vstack([ensure_numpy(fg) for fg in feature_history])
        else:
            all_genes = ensure_numpy(feature_history)

        xp = self._get_array_module()

        if self.use_gpu:
            all_genes = xp.asarray(all_genes.astype(np.float64))
            if weights is not None:
                weights = xp.asarray(weights)

        if weights is not None:
            weights = weights / xp.sum(weights)
            selection_counts = xp.sum(all_genes * weights[:, None], axis=0)  # pyright: ignore[reportOptionalSubscript]
        else:
            selection_counts = xp.mean(all_genes, axis=0)

        return ensure_numpy(selection_counts).astype(np.float64)

    def compute_coselection_matrix(
        self,
        feature_history: NDArray[np.bool_] | list[NDArray[np.bool_]],
        normalize: bool = True,
    ) -> NDArray[np.float64]:
        """Compute feature co-selection matrix."""
        if isinstance(feature_history, list):
            all_genes = np.vstack([ensure_numpy(fg) for fg in feature_history])
        else:
            all_genes = ensure_numpy(feature_history)

        xp = self._get_array_module()

        if self.use_gpu:
            all_genes = xp.asarray(all_genes.astype(np.float64))

        coselection = xp.dot(all_genes.T, all_genes)

        if normalize:
            n_individuals = all_genes.shape[0]
            coselection = coselection / n_individuals

        return ensure_numpy(coselection).astype(np.float64)

    def compute_conditional_selection(
        self,
        feature_history: NDArray[np.bool_] | list[NDArray[np.bool_]],
        condition_feature: int,
    ) -> NDArray[np.float64]:
        """Compute selection probability conditioned on another feature."""
        if isinstance(feature_history, list):
            all_genes = np.vstack([ensure_numpy(fg) for fg in feature_history])
        else:
            all_genes = ensure_numpy(feature_history)

        condition_mask = all_genes[:, condition_feature].astype(bool)

        if not np.any(condition_mask):
            return np.zeros(all_genes.shape[1], dtype=np.float64)

        conditional_freq = np.mean(all_genes[condition_mask], axis=0)
        return conditional_freq.astype(np.float64)

    def analyze_features(
        self,
        feature_history: NDArray[np.bool_] | list[NDArray[np.bool_]],
        final_selection: NDArray[np.bool_],
        fitness_history: NDArray[np.float64] | None = None,
    ) -> FeatureImportance:
        """Comprehensive feature analysis."""
        freq = self.compute_selection_frequency(feature_history, weights=fitness_history)
        cosel_matrix = self.compute_coselection_matrix(feature_history)

        n_features = len(freq)
        cosel_scores = np.zeros(n_features, dtype=np.float64)
        for i in range(n_features):
            mask = np.ones(n_features, dtype=bool)
            mask[i] = False
            cosel_scores[i] = np.mean(cosel_matrix[i, mask])

        combined_score = freq + 0.5 * cosel_scores
        importance_rank = np.argsort(-combined_score).argsort() + 1

        return FeatureImportance(
            feature_indices=np.arange(n_features, dtype=np.intp),
            selection_frequency=freq,
            final_selection=np.asarray(final_selection, dtype=np.bool_),
            coselection_scores=cosel_scores,
            importance_rank=importance_rank.astype(np.intp),
        )




class ConvergenceAnalyzer:
    """Analyzes optimization convergence patterns."""

    def __init__(
        self,
        window_size: int = 5,
        improvement_threshold: float = 1e-4,
    ) -> None:
        """Initialize convergence analyzer."""
        self.window_size = window_size
        self.improvement_threshold = improvement_threshold

    def compute_convergence_metrics(
        self,
        generation_history: list[GenerationStats],
    ) -> dict[str, Any]:
        """Compute convergence metrics from generation history."""
        if not generation_history:
            return {}

        best_fitness = np.array([g.best_fitness for g in generation_history])
        mean_fitness = np.array([g.mean_fitness for g in generation_history])
        diversity = np.array([g.diversity for g in generation_history])

        n_gens = len(best_fitness)
        improvements = np.diff(best_fitness)

        converged_gen = None
        for i in range(self.window_size, len(improvements)):
            window = improvements[i - self.window_size : i]
            if np.max(np.abs(window)) < self.improvement_threshold:
                converged_gen = i
                break

        if n_gens > self.window_size:
            moving_avg = np.convolve(
                improvements, np.ones(self.window_size) / self.window_size, mode="valid"
            )
        else:
            moving_avg = np.array([])

        total_improvement = best_fitness[-1] - best_fitness[0]
        mean_improvement_rate = total_improvement / max(n_gens - 1, 1)

        if len(diversity) > 1 and diversity[0] > 0:
            diversity_loss = (diversity[0] - diversity[-1]) / diversity[0]
        else:
            diversity_loss = 0.0

        return {
            "n_generations": n_gens,
            "total_improvement": float(total_improvement),
            "mean_improvement_rate": float(mean_improvement_rate),
            "converged_generation": converged_gen,
            "final_best_fitness": float(best_fitness[-1]),
            "final_mean_fitness": float(mean_fitness[-1]),
            "final_diversity": float(diversity[-1]) if len(diversity) > 0 else 0.0,
            "diversity_loss": float(diversity_loss),
            "improvement_history": improvements.tolist(),
            "moving_avg_improvement": moving_avg.tolist() if len(moving_avg) > 0 else [],
        }

    def detect_premature_convergence(
        self,
        generation_history: list[GenerationStats],
        expected_generations: int,
    ) -> dict[str, Any]:
        """Detect if optimization converged prematurely."""
        metrics = self.compute_convergence_metrics(generation_history)

        actual_gens = metrics["n_generations"]
        converged_gen = metrics.get("converged_generation")

        is_premature = False
        premature_ratio = 0.0

        if converged_gen is not None:
            premature_ratio = converged_gen / expected_generations
            is_premature = premature_ratio < 0.5

        return {
            "is_premature": is_premature,
            "converged_generation": converged_gen,
            "expected_generations": expected_generations,
            "actual_generations": actual_gens,
            "premature_ratio": premature_ratio,
            "recommendation": self._get_recommendation(
                is_premature, metrics.get("diversity_loss", 0)
            ),
        }

    def _get_recommendation(
        self,
        is_premature: bool,
        diversity_loss: float,
    ) -> str:
        """Get recommendation based on convergence analysis."""
        if not is_premature:
            return "Convergence appears normal."

        recommendations = []
        if diversity_loss > 0.8:
            recommendations.append("Consider increasing mutation rate to maintain diversity.")
            recommendations.append("Try larger population size.")

        if is_premature:
            recommendations.append("Consider reducing selection pressure.")
            recommendations.append("Try different initialization method.")

        return " ".join(recommendations) if recommendations else "No specific recommendations."




class ResultsAggregator:
    """Aggregates results from multiple optimization runs or CV folds."""

    def __init__(self, confidence_level: float = 0.95) -> None:
        """Initialize results aggregator."""
        self.confidence_level = confidence_level

    def aggregate_fold_results(
        self,
        fold_results: list[FoldResult],
        scoring_metric: str = "accuracy",
    ) -> NestedCVResults:
        """Aggregate results from nested CV folds."""
        if not fold_results:
            msg = "No fold results to aggregate"
            raise ValueError(msg)

        outer_scores = np.array([f.outer_score for f in fold_results])
        mean_score = float(np.mean(outer_scores))
        std_score = float(np.std(outer_scores, ddof=1)) if len(outer_scores) > 1 else 0.0

        ci_lower, ci_upper = self._compute_confidence_interval(outer_scores)
        feature_stability = self._compute_feature_stability(fold_results)
        total_time = sum(f.elapsed_time for f in fold_results)

        return NestedCVResults(
            fold_results=fold_results,
            mean_score=mean_score,
            std_score=std_score,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            feature_stability=feature_stability,
            total_elapsed_time=total_time,
            scoring_metric=scoring_metric,
        )

    def _compute_confidence_interval(
        self,
        scores: NDArray[np.float64],
    ) -> tuple[float, float]:
        """Compute confidence interval for scores."""
        n = len(scores)
        if n < 2:
            mean = float(np.mean(scores))
            return mean, mean

        mean = float(np.mean(scores))
        sem = float(stats.sem(scores))
        t_value = stats.t.ppf((1 + self.confidence_level) / 2, df=n - 1)
        margin = t_value * sem

        return mean - margin, mean + margin  # pyright: ignore[reportReturnType]

    def _compute_feature_stability(
        self,
        fold_results: list[FoldResult],
    ) -> NDArray[np.float64]:
        """Compute feature selection stability across folds."""
        if not fold_results:
            return np.array([], dtype=np.float64)

        feature_masks = np.array(
            [f.inner_result.best_feature_mask for f in fold_results]
        )
        return np.mean(feature_masks, axis=0).astype(np.float64)

    def aggregate_multiple_runs(
        self,
        run_results: list[OptimizationResults],
    ) -> dict[str, Any]:
        """Aggregate results from multiple independent runs."""
        if not run_results:
            msg = "No run results to aggregate"
            raise ValueError(msg)

        best_fitness = np.array([r.best_fitness for r in run_results])
        n_features = np.array([r.n_features_selected for r in run_results])
        elapsed_times = np.array([r.elapsed_time for r in run_results])

        feature_masks = np.array([r.best_feature_mask for r in run_results])
        feature_stability = np.mean(feature_masks, axis=0)
        consensus_features = np.where(feature_stability > 0.5)[0]
        best_run_idx = int(np.argmax(best_fitness))

        return {
            "n_runs": len(run_results),
            "fitness": {
                "mean": float(np.mean(best_fitness)),
                "std": float(np.std(best_fitness)),
                "min": float(np.min(best_fitness)),
                "max": float(np.max(best_fitness)),
                "median": float(np.median(best_fitness)),
            },
            "n_features_selected": {
                "mean": float(np.mean(n_features)),
                "std": float(np.std(n_features)),
                "min": int(np.min(n_features)),
                "max": int(np.max(n_features)),
            },
            "elapsed_time": {
                "mean": float(np.mean(elapsed_times)),
                "total": float(np.sum(elapsed_times)),
            },
            "feature_stability": feature_stability.tolist(),
            "consensus_features": consensus_features.tolist(),
            "best_run_idx": best_run_idx,
            "best_result": run_results[best_run_idx].to_dict(),
        }




class StatisticalAnalyzer:
    """Statistical analysis utilities for optimization results."""

    @staticmethod
    def paired_ttest(
        scores_a: NDArray[np.float64],
        scores_b: NDArray[np.float64],
    ) -> dict[str, Any]:
        """Perform paired t-test between two sets of scores."""
        if len(scores_a) != len(scores_b):
            msg = "Score arrays must have same length"
            raise ValueError(msg)

        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        diff = scores_a - scores_b
        cohens_d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff) > 0 else 0.0

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": cohens_d,
            "mean_diff": float(np.mean(diff)),
            "significant_0.05": p_value < 0.05,
            "significant_0.01": p_value < 0.01,
        }

    @staticmethod
    def wilcoxon_test(
        scores_a: NDArray[np.float64],
        scores_b: NDArray[np.float64],
    ) -> dict[str, Any]:
        """Perform Wilcoxon signed-rank test (non-parametric)."""
        try:
            stat, p_value = stats.wilcoxon(scores_a, scores_b)
            return {
                "statistic": float(stat),  # pyright: ignore[reportArgumentType]
                "p_value": float(p_value),  # pyright: ignore[reportArgumentType]
                "significant_0.05": p_value < 0.05,  # pyright: ignore[reportOperatorIssue]
            }
        except ValueError as e:
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant_0.05": False,
                "error": str(e),
            }

    @staticmethod
    def bootstrap_confidence_interval(
        scores: NDArray[np.float64],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: int | None = None,
    ) -> dict[str, Any]:
        """Compute bootstrap confidence interval."""
        rng = np.random.default_rng(random_state)
        n = len(scores)

        bootstrap_means = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sample_idx = rng.integers(0, n, size=n)
            bootstrap_means[i] = np.mean(scores[sample_idx])

        alpha = 1 - confidence_level
        ci_lower = float(np.percentile(bootstrap_means, alpha / 2 * 100))
        ci_upper = float(np.percentile(bootstrap_means, (1 - alpha / 2) * 100))

        return {
            "mean": float(np.mean(scores)),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_width": ci_upper - ci_lower,
            "confidence_level": confidence_level,
            "n_bootstrap": n_bootstrap,
            "bootstrap_std": float(np.std(bootstrap_means)),
        }

    @staticmethod
    def compute_effect_size(
        mean_diff: float,
        std_pooled: float,
    ) -> dict[str, Any]:
        """Compute and interpret effect size."""
        if std_pooled == 0:
            cohens_d = 0.0
        else:
            cohens_d = mean_diff / std_pooled

        if abs(cohens_d) < 0.2:
            interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            interpretation = "small"
        elif abs(cohens_d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return {
            "cohens_d": cohens_d,
            "interpretation": interpretation,
            "direction": "positive" if cohens_d > 0 else "negative",
        }




class ResultsExporter:
    """Export optimization results to various formats."""

    @staticmethod
    def to_json(
        results: OptimizationResults | NestedCVResults,
        filepath: str | Path,
        indent: int = 2,
    ) -> None:
        """Export results to JSON file."""
        filepath = Path(filepath)
        with filepath.open("w") as f:
            json.dump(results.to_dict(), f, indent=indent)

    @staticmethod
    def to_csv_summary(
        results: NestedCVResults,
        filepath: str | Path,
    ) -> None:
        """Export nested CV summary to CSV."""
        filepath = Path(filepath)

        lines = [
            "fold,outer_score,n_features,best_fitness,elapsed_time",
        ]

        for fold in results.fold_results:
            line = (
                f"{fold.fold_idx},"
                f"{fold.outer_score:.6f},"
                f"{fold.inner_result.n_features_selected},"
                f"{fold.inner_result.best_fitness:.6f},"
                f"{fold.elapsed_time:.2f}"
            )
            lines.append(line)

        lines.append("")
        lines.append(f"mean,{results.mean_score:.6f},,,,")
        lines.append(f"std,{results.std_score:.6f},,,,")
        lines.append(f"ci_lower,{results.ci_lower:.6f},,,,")
        lines.append(f"ci_upper,{results.ci_upper:.6f},,,,")

        with filepath.open("w") as f:
            f.write("\n".join(lines))

    @staticmethod
    def to_feature_report(
        feature_importance: FeatureImportance,
        feature_names: list[str] | None = None,
        filepath: str | Path | None = None,
    ) -> str:
        """Generate feature importance report."""
        lines = [
            "Feature Importance Report",
            "=" * 50,
            "",
            "Rank | Feature | Frequency | Co-selection | Selected",
            "-" * 50,
        ]

        sorted_idx = np.argsort(feature_importance.importance_rank)

        for idx in sorted_idx:
            rank = feature_importance.importance_rank[idx]
            if feature_names is not None:
                name = feature_names[idx]
            else:
                name = f"Feature_{idx}"
            freq = feature_importance.selection_frequency[idx]
            cosel = feature_importance.coselection_scores[idx]
            selected = "Yes" if feature_importance.final_selection[idx] else "No"

            lines.append(f"{rank:4d} | {name:20s} | {freq:9.4f} | {cosel:12.4f} | {selected}")

        lines.append("")
        lines.append("-" * 50)
        lines.append(f"Total selected: {np.sum(feature_importance.final_selection)}")

        report = "\n".join(lines)

        if filepath is not None:
            with Path(filepath).open("w") as f:
                f.write(report)

        return report




def collect_from_hybrid_result(
    hybrid_result: Any,
    feature_history: list[NDArray[np.bool_]] | None = None,
    fitness_history: NDArray[np.float64] | None = None,
) -> OptimizationResults:
    """Collect OptimizationResults from HybridResult."""
    best_feature_mask = ensure_numpy(hybrid_result.best_feature_mask)
    best_discrete_params = ensure_numpy(hybrid_result.best_discrete_params)
    best_continuous_params = ensure_numpy(hybrid_result.best_continuous_params)

    generation_stats = []
    for gen_data in hybrid_result.generation_history:
        if isinstance(gen_data, dict):
            stats_obj = GenerationStats(
                generation=gen_data.get("generation", 0),
                best_fitness=gen_data.get("best", gen_data.get("best_fitness", 0.0)),
                mean_fitness=gen_data.get("mean", gen_data.get("mean_fitness", 0.0)),
                std_fitness=gen_data.get("std", gen_data.get("std_fitness", 0.0)),
                median_fitness=gen_data.get("median", gen_data.get("median_fitness", 0.0)),
                min_fitness=gen_data.get("min", gen_data.get("min_fitness", 0.0)),
                diversity=gen_data.get("diversity", 0.0),
                n_evaluated=gen_data.get("n_evaluated", 0),
                elapsed_time=gen_data.get("elapsed_time", 0.0),
            )
            generation_stats.append(stats_obj)

    feature_importance = None
    if feature_history is not None:
        analyzer = FeatureAnalyzer(use_gpu=False)
        feature_importance = analyzer.analyze_features(
            feature_history=feature_history,
            final_selection=best_feature_mask,
            fitness_history=fitness_history,
        )

    return OptimizationResults(
        best_feature_mask=best_feature_mask.astype(np.bool_),
        best_discrete_params=best_discrete_params.astype(np.int64),
        best_continuous_params=best_continuous_params.astype(np.float64),
        best_fitness=float(hybrid_result.best_fitness),
        n_generations=int(hybrid_result.n_generations),
        n_evaluations=int(hybrid_result.n_pso_evaluations),
        converged=bool(hybrid_result.converged),
        elapsed_time=float(hybrid_result.elapsed_time),
        generation_history=generation_stats,
        feature_importance=feature_importance,
        cache_stats=hybrid_result.cache_stats,
    )


def compute_diversity(
    population_genes: NDArray[np.bool_] | NDArray[np.float64],
    use_gpu: bool = False,
) -> float:
    """Compute population diversity metric."""
    genes = ensure_numpy(population_genes)
    pop_size = genes.shape[0]

    if pop_size < 2:
        return 0.0

    xp = np
    if use_gpu:
        try:
            import cupy as cp
            xp = cp
            genes = cp.asarray(genes)
        except ImportError:
            pass

    is_binary = genes.dtype == np.bool_ or np.issubdtype(genes.dtype, np.integer)
    genes_float = genes.astype(xp.float64)

    diff = genes_float[:, None, :] - genes_float[None, :, :]

    if is_binary:
        distances = xp.mean(xp.abs(diff) > 0, axis=2)
    else:
        distances = xp.sqrt(xp.sum(diff**2, axis=2))

    n = pop_size
    triu_indices = xp.triu_indices(n, k=1)
    pairwise_distances = distances[triu_indices]

    diversity = float(ensure_numpy(xp.mean(pairwise_distances)))

    return diversity