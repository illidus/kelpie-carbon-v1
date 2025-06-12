"""
SKEMA Statistical Benchmarking Component

This module provides comprehensive statistical analysis comparing our kelp detection
pipeline against SKEMA methodology using real validation sites and rigorous
statistical testing.

Features:
- Hypothesis testing for method comparison
- Confidence interval calculations
- Correlation and regression analysis
- Cross-validation and bootstrap sampling
- Statistical significance testing
- Performance metric distributions
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    confidence_interval: tuple[float, float]
    effect_size: float
    interpretation: str
    recommendation: str

@dataclass
class BenchmarkComparison:
    """Comprehensive comparison between SKEMA and our pipeline."""
    site_name: str
    skema_results: dict[str, float]
    our_results: dict[str, float]
    statistical_tests: list[StatisticalTestResult]
    correlation_analysis: dict[str, float]
    regression_analysis: dict[str, Any]
    bootstrap_results: dict[str, dict[str, float]]
    overall_assessment: str

class StatisticalBenchmarker:
    """Performs rigorous statistical comparison between SKEMA and our pipeline."""
    
    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 1000):
        """
        Initialize statistical benchmarker.
        
        Args:
            alpha: Significance level for statistical tests
            n_bootstrap: Number of bootstrap samples for confidence intervals
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.confidence_level = 1 - alpha
        
    def compare_methods_comprehensive(
        self, 
        validation_sites: list[dict[str, Any]]
    ) -> list[BenchmarkComparison]:
        """
        Perform comprehensive statistical comparison across multiple validation sites.
        
        Args:
            validation_sites: List of validation sites with SKEMA and our results
            
        Returns:
            List of detailed benchmark comparisons for each site
        """
        
        comparisons = []
        
        for site_data in validation_sites:
            print(f"\n🔬 Analyzing site: {site_data['name']}")
            
            comparison = self._analyze_single_site(site_data)
            comparisons.append(comparison)
            
            # Print summary for this site
            self._print_site_summary(comparison)
        
        # Perform meta-analysis across all sites
        meta_analysis = self._perform_meta_analysis(comparisons)
        print("\n📊 Meta-Analysis Results:")
        self._print_meta_analysis(meta_analysis)
        
        return comparisons
    
    def _analyze_single_site(self, site_data: dict[str, Any]) -> BenchmarkComparison:
        """Perform detailed statistical analysis for a single validation site."""
        
        site_name = site_data['name']
        skema_predictions = site_data['skema_results']['predictions']
        our_predictions = site_data['our_results']['predictions']
        ground_truth = site_data['ground_truth']
        
        # Calculate basic performance metrics
        skema_metrics = self._calculate_performance_metrics(skema_predictions, ground_truth)
        our_metrics = self._calculate_performance_metrics(our_predictions, ground_truth)
        
        # Perform statistical significance tests
        statistical_tests = self._perform_statistical_tests(
            skema_predictions, our_predictions, ground_truth
        )
        
        # Correlation analysis
        correlation_analysis = self._perform_correlation_analysis(
            skema_predictions, our_predictions, ground_truth
        )
        
        # Regression analysis
        regression_analysis = self._perform_regression_analysis(
            skema_predictions, our_predictions, ground_truth
        )
        
        # Bootstrap confidence intervals
        bootstrap_results = self._perform_bootstrap_analysis(
            skema_predictions, our_predictions, ground_truth
        )
        
        # Overall assessment
        overall_assessment = self._generate_overall_assessment(
            skema_metrics, our_metrics, statistical_tests
        )
        
        return BenchmarkComparison(
            site_name=site_name,
            skema_results=skema_metrics,
            our_results=our_metrics,
            statistical_tests=statistical_tests,
            correlation_analysis=correlation_analysis,
            regression_analysis=regression_analysis,
            bootstrap_results=bootstrap_results,
            overall_assessment=overall_assessment
        )
    
    def _calculate_performance_metrics(
        self, 
        predictions: np.ndarray, 
        ground_truth: np.ndarray
    ) -> dict[str, float]:
        """Calculate comprehensive performance metrics."""
        
        # Convert to binary if needed
        pred_binary = (predictions > 0.5).astype(int)
        truth_binary = (ground_truth > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(truth_binary, pred_binary),
            'precision': precision_score(truth_binary, pred_binary, zero_division=0),
            'recall': recall_score(truth_binary, pred_binary, zero_division=0),
            'f1_score': f1_score(truth_binary, pred_binary, zero_division=0),
            'specificity': self._calculate_specificity(truth_binary, pred_binary),
            'npv': self._calculate_npv(truth_binary, pred_binary),  # Negative Predictive Value
            'balanced_accuracy': self._calculate_balanced_accuracy(truth_binary, pred_binary)
        }
        
        # AUC if predictions are continuous
        if len(np.unique(predictions)) > 2:
            try:
                metrics['auc_roc'] = roc_auc_score(truth_binary, predictions)
            except ValueError:
                metrics['auc_roc'] = 0.5  # Random performance if only one class
        
        # Additional metrics
        metrics['mse'] = np.mean((predictions - ground_truth) ** 2)
        metrics['mae'] = np.mean(np.abs(predictions - ground_truth))
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _calculate_npv(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Negative Predictive Value."""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    def _calculate_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate balanced accuracy (average of sensitivity and specificity)."""
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = self._calculate_specificity(y_true, y_pred)
        return (sensitivity + specificity) / 2
    
    def _perform_statistical_tests(
        self, 
        skema_pred: np.ndarray, 
        our_pred: np.ndarray, 
        ground_truth: np.ndarray
    ) -> list[StatisticalTestResult]:
        """Perform comprehensive statistical significance tests."""
        
        tests = []
        
        # 1. Paired t-test for accuracy comparison
        skema_errors = np.abs(skema_pred - ground_truth)
        our_errors = np.abs(our_pred - ground_truth)
        
        t_stat, t_pval = stats.ttest_rel(skema_errors, our_errors)
        
        tests.append(StatisticalTestResult(
            test_name="Paired t-test (Error Comparison)",
            statistic=t_stat,
            p_value=t_pval,
            confidence_interval=self._calculate_t_test_ci(skema_errors, our_errors),
            effect_size=self._calculate_cohens_d(skema_errors, our_errors),
            interpretation=self._interpret_t_test(t_pval, t_stat),
            recommendation=self._recommend_t_test(t_pval, t_stat)
        ))
        
        # 2. Wilcoxon signed-rank test (non-parametric alternative)
        w_stat, w_pval = stats.wilcoxon(skema_errors, our_errors, alternative='two-sided')
        
        tests.append(StatisticalTestResult(
            test_name="Wilcoxon Signed-Rank Test",
            statistic=w_stat,
            p_value=w_pval,
            confidence_interval=(np.nan, np.nan),  # Not directly available
            effect_size=self._calculate_rank_biserial_correlation(skema_errors, our_errors),
            interpretation=self._interpret_wilcoxon(w_pval),
            recommendation=self._recommend_wilcoxon(w_pval)
        ))
        
        # 3. McNemar's test for binary classification comparison
        skema_binary = (skema_pred > 0.5).astype(int)
        our_binary = (our_pred > 0.5).astype(int)
        truth_binary = (ground_truth > 0.5).astype(int)
        
        mcnemar_result = self._mcnemar_test(skema_binary, our_binary, truth_binary)
        tests.append(mcnemar_result)
        
        # 4. Correlation test between methods
        corr_coeff, corr_pval = stats.pearsonr(skema_pred, our_pred)
        
        tests.append(StatisticalTestResult(
            test_name="Pearson Correlation Test",
            statistic=corr_coeff,
            p_value=corr_pval,
            confidence_interval=self._calculate_correlation_ci(corr_coeff, len(skema_pred)),
            effect_size=corr_coeff,  # Correlation is its own effect size
            interpretation=self._interpret_correlation(corr_coeff, corr_pval),
            recommendation=self._recommend_correlation(corr_coeff, corr_pval)
        ))
        
        return tests
    
    def _calculate_t_test_ci(self, sample1: np.ndarray, sample2: np.ndarray) -> tuple[float, float]:
        """Calculate confidence interval for paired t-test."""
        diff = sample1 - sample2
        mean_diff = np.mean(diff)
        sem_diff = stats.sem(diff)
        
        t_critical = stats.t.ppf(1 - self.alpha/2, len(diff) - 1)
        margin = t_critical * sem_diff
        
        return (mean_diff - margin, mean_diff + margin)
    
    def _calculate_cohens_d(self, sample1: np.ndarray, sample2: np.ndarray) -> float:
        """Calculate Cohen's d effect size for paired samples."""
        diff = sample1 - sample2
        return np.mean(diff) / np.std(diff, ddof=1)
    
    def _calculate_rank_biserial_correlation(self, sample1: np.ndarray, sample2: np.ndarray) -> float:
        """Calculate rank-biserial correlation as effect size for Wilcoxon test."""
        n = len(sample1)
        w_stat, _ = stats.wilcoxon(sample1, sample2)
        return (2 * w_stat) / (n * (n + 1)) - 1
    
    def _mcnemar_test(
        self, 
        skema_pred: np.ndarray, 
        our_pred: np.ndarray, 
        ground_truth: np.ndarray
    ) -> StatisticalTestResult:
        """Perform McNemar's test for comparing binary classifiers."""
        
        # Create contingency table for correct/incorrect predictions
        skema_correct = (skema_pred == ground_truth)
        our_correct = (our_pred == ground_truth)
        
        # McNemar's table: 2x2 contingency table
        both_correct = np.sum(skema_correct & our_correct)
        skema_only = np.sum(skema_correct & ~our_correct)
        our_only = np.sum(~skema_correct & our_correct)
        both_wrong = np.sum(~skema_correct & ~our_correct)
        
        # McNemar's statistic focuses on discordant pairs
        if skema_only + our_only == 0:
            mcnemar_stat = 0
            p_value = 1.0
        else:
            mcnemar_stat = (abs(skema_only - our_only) - 1)**2 / (skema_only + our_only)
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        
        effect_size = abs(skema_only - our_only) / (skema_only + our_only) if (skema_only + our_only) > 0 else 0
        
        return StatisticalTestResult(
            test_name="McNemar's Test (Binary Classification)",
            statistic=mcnemar_stat,
            p_value=p_value,
            confidence_interval=(np.nan, np.nan),
            effect_size=effect_size,
            interpretation=self._interpret_mcnemar(p_value, skema_only, our_only),
            recommendation=self._recommend_mcnemar(p_value, skema_only, our_only)
        )
    
    def _calculate_correlation_ci(self, r: float, n: int) -> tuple[float, float]:
        """Calculate confidence interval for Pearson correlation coefficient."""
        # Fisher z-transformation
        z = 0.5 * np.log((1 + r) / (1 - r))
        se = 1 / np.sqrt(n - 3)
        
        z_critical = stats.norm.ppf(1 - self.alpha/2)
        z_lower = z - z_critical * se
        z_upper = z + z_critical * se
        
        # Transform back to correlation scale
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return (r_lower, r_upper)
    
    def _perform_correlation_analysis(
        self, 
        skema_pred: np.ndarray, 
        our_pred: np.ndarray, 
        ground_truth: np.ndarray
    ) -> dict[str, float]:
        """Perform comprehensive correlation analysis."""
        
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(skema_pred, our_pred)
        
        # Spearman correlation (rank-based)
        spearman_r, spearman_p = stats.spearmanr(skema_pred, our_pred)
        
        # Kendall's tau
        kendall_tau, kendall_p = stats.kendalltau(skema_pred, our_pred)
        
        # Correlation with ground truth
        skema_truth_corr, _ = stats.pearsonr(skema_pred, ground_truth)
        our_truth_corr, _ = stats.pearsonr(our_pred, ground_truth)
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'kendall_tau': kendall_tau,
            'kendall_p': kendall_p,
            'skema_truth_correlation': skema_truth_corr,
            'our_truth_correlation': our_truth_corr,
            'correlation_difference': our_truth_corr - skema_truth_corr
        }
    
    def _perform_regression_analysis(
        self, 
        skema_pred: np.ndarray, 
        our_pred: np.ndarray, 
        ground_truth: np.ndarray
    ) -> dict[str, Any]:
        """Perform regression analysis to understand method relationships."""
        
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        # Linear regression: our_pred ~ skema_pred
        reg_model = LinearRegression()
        reg_model.fit(skema_pred.reshape(-1, 1), our_pred)
        
        pred_from_skema = reg_model.predict(skema_pred.reshape(-1, 1))
        r2_methods = r2_score(our_pred, pred_from_skema)
        
        # Regression against ground truth for both methods
        reg_skema = LinearRegression()
        reg_skema.fit(ground_truth.reshape(-1, 1), skema_pred)
        skema_from_truth = reg_skema.predict(ground_truth.reshape(-1, 1))
        r2_skema_truth = r2_score(skema_pred, skema_from_truth)
        
        reg_ours = LinearRegression()
        reg_ours.fit(ground_truth.reshape(-1, 1), our_pred)
        ours_from_truth = reg_ours.predict(ground_truth.reshape(-1, 1))
        r2_ours_truth = r2_score(our_pred, ours_from_truth)
        
        return {
            'slope_our_vs_skema': reg_model.coef_[0],
            'intercept_our_vs_skema': reg_model.intercept_,
            'r2_methods_comparison': r2_methods,
            'r2_skema_vs_truth': r2_skema_truth,
            'r2_ours_vs_truth': r2_ours_truth,
            'regression_improvement': r2_ours_truth - r2_skema_truth
        }
    
    def _perform_bootstrap_analysis(
        self, 
        skema_pred: np.ndarray, 
        our_pred: np.ndarray, 
        ground_truth: np.ndarray
    ) -> dict[str, dict[str, float]]:
        """Perform bootstrap analysis for robust confidence intervals."""
        
        n_samples = len(skema_pred)
        
        def bootstrap_metric(pred, truth, metric_func):
            """Calculate bootstrap distribution for a metric."""
            bootstrap_values = []
            
            for _ in range(self.n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(n_samples, n_samples, replace=True)
                pred_boot = pred[indices]
                truth_boot = truth[indices]
                
                # Calculate metric
                try:
                    metric_value = metric_func(pred_boot, truth_boot)
                    if np.isfinite(metric_value):
                        bootstrap_values.append(metric_value)
                except:
                    pass  # Skip failed calculations
            
            if len(bootstrap_values) == 0:
                return {'mean': 0, 'ci_lower': 0, 'ci_upper': 0, 'std': 0}
            
            bootstrap_values = np.array(bootstrap_values)
            ci_lower = np.percentile(bootstrap_values, 100 * self.alpha/2)
            ci_upper = np.percentile(bootstrap_values, 100 * (1 - self.alpha/2))
            
            return {
                'mean': np.mean(bootstrap_values),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'std': np.std(bootstrap_values)
            }
        
        # Define metric functions
        def accuracy_func(pred, truth):
            pred_binary = (pred > 0.5).astype(int)
            truth_binary = (truth > 0.5).astype(int)
            return accuracy_score(truth_binary, pred_binary)
        
        def f1_func(pred, truth):
            pred_binary = (pred > 0.5).astype(int)
            truth_binary = (truth > 0.5).astype(int)
            return f1_score(truth_binary, pred_binary, zero_division=0)
        
        def correlation_func(pred, truth):
            corr, _ = stats.pearsonr(pred, truth)
            return corr
        
        # Bootstrap for SKEMA
        skema_bootstrap = {
            'accuracy': bootstrap_metric(skema_pred, ground_truth, accuracy_func),
            'f1_score': bootstrap_metric(skema_pred, ground_truth, f1_func),
            'correlation': bootstrap_metric(skema_pred, ground_truth, correlation_func)
        }
        
        # Bootstrap for our method
        our_bootstrap = {
            'accuracy': bootstrap_metric(our_pred, ground_truth, accuracy_func),
            'f1_score': bootstrap_metric(our_pred, ground_truth, f1_func),
            'correlation': bootstrap_metric(our_pred, ground_truth, correlation_func)
        }
        
        return {
            'skema': skema_bootstrap,
            'ours': our_bootstrap
        }
    
    def _interpret_t_test(self, p_value: float, t_stat: float) -> str:
        """Interpret t-test results."""
        if p_value < self.alpha:
            direction = "SKEMA" if t_stat > 0 else "Our pipeline"
            return f"Significant difference (p={p_value:.4f}): {direction} has lower error rates"
        else:
            return f"No significant difference in error rates (p={p_value:.4f})"
    
    def _recommend_t_test(self, p_value: float, t_stat: float) -> str:
        """Generate recommendation based on t-test."""
        if p_value < self.alpha:
            if t_stat > 0:
                return "SKEMA method shows significantly better performance"
            else:
                return "Our pipeline shows significantly better performance"
        else:
            return "Both methods perform equivalently - choose based on other criteria"
    
    def _interpret_wilcoxon(self, p_value: float) -> str:
        """Interpret Wilcoxon test results."""
        if p_value < self.alpha:
            return f"Significant difference in error distributions (p={p_value:.4f})"
        else:
            return f"No significant difference in error distributions (p={p_value:.4f})"
    
    def _recommend_wilcoxon(self, p_value: float) -> str:
        """Generate recommendation based on Wilcoxon test."""
        if p_value < self.alpha:
            return "Methods have significantly different error characteristics"
        else:
            return "Methods have similar error characteristics"
    
    def _interpret_mcnemar(self, p_value: float, skema_only: int, our_only: int) -> str:
        """Interpret McNemar's test results."""
        if p_value < self.alpha:
            better_method = "Our pipeline" if our_only > skema_only else "SKEMA"
            return f"Significant difference in classification accuracy (p={p_value:.4f}): {better_method} correctly classifies more cases"
        else:
            return f"No significant difference in classification accuracy (p={p_value:.4f})"
    
    def _recommend_mcnemar(self, p_value: float, skema_only: int, our_only: int) -> str:
        """Generate recommendation based on McNemar's test."""
        if p_value < self.alpha:
            if our_only > skema_only:
                return "Our pipeline shows significantly better classification performance"
            else:
                return "SKEMA shows significantly better classification performance"
        else:
            return "Both methods have equivalent classification accuracy"
    
    def _interpret_correlation(self, r: float, p_value: float) -> str:
        """Interpret correlation test results."""
        strength = "very strong" if abs(r) > 0.9 else "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.5 else "weak" if abs(r) > 0.3 else "very weak"
        
        if p_value < self.alpha:
            return f"Significant {strength} correlation between methods (r={r:.3f}, p={p_value:.4f})"
        else:
            return f"No significant correlation between methods (r={r:.3f}, p={p_value:.4f})"
    
    def _recommend_correlation(self, r: float, p_value: float) -> str:
        """Generate recommendation based on correlation."""
        if p_value < self.alpha and abs(r) > 0.7:
            return "Methods are highly correlated - results are consistent"
        elif p_value < self.alpha and abs(r) > 0.3:
            return "Methods are moderately correlated - some agreement in results"
        else:
            return "Methods show low correlation - investigate differences in approach"
    
    def _generate_overall_assessment(
        self, 
        skema_metrics: dict[str, float], 
        our_metrics: dict[str, float], 
        statistical_tests: list[StatisticalTestResult]
    ) -> str:
        """Generate overall assessment of method comparison."""
        
        # Compare key metrics
        accuracy_diff = our_metrics['accuracy'] - skema_metrics['accuracy']
        f1_diff = our_metrics['f1_score'] - skema_metrics['f1_score']
        
        # Count significant tests favoring each method
        significant_tests = [test for test in statistical_tests if test.p_value < self.alpha]
        
        assessment_parts = []
        
        # Performance comparison
        if accuracy_diff > 0.05:
            assessment_parts.append("Our pipeline shows notably higher accuracy")
        elif accuracy_diff < -0.05:
            assessment_parts.append("SKEMA shows notably higher accuracy")
        else:
            assessment_parts.append("Both methods show similar accuracy")
        
        if f1_diff > 0.05:
            assessment_parts.append("Our pipeline shows better balanced performance (F1)")
        elif f1_diff < -0.05:
            assessment_parts.append("SKEMA shows better balanced performance (F1)")
        
        # Statistical significance
        if len(significant_tests) > 0:
            assessment_parts.append(f"{len(significant_tests)} statistical tests show significant differences")
        else:
            assessment_parts.append("No statistically significant differences detected")
        
        # Final recommendation
        if accuracy_diff > 0.02 and f1_diff > 0.02:
            assessment_parts.append("Recommendation: Our pipeline preferred")
        elif accuracy_diff < -0.02 and f1_diff < -0.02:
            assessment_parts.append("Recommendation: SKEMA method preferred")
        else:
            assessment_parts.append("Recommendation: Methods are equivalent")
        
        return ". ".join(assessment_parts) + "."
    
    def _perform_meta_analysis(self, comparisons: list[BenchmarkComparison]) -> dict[str, Any]:
        """Perform meta-analysis across all validation sites."""
        
        if len(comparisons) == 0:
            return {}
        
        # Aggregate performance metrics
        skema_accuracies = [comp.skema_results['accuracy'] for comp in comparisons]
        our_accuracies = [comp.our_results['accuracy'] for comp in comparisons]
        
        skema_f1s = [comp.skema_results['f1_score'] for comp in comparisons]
        our_f1s = [comp.our_results['f1_score'] for comp in comparisons]
        
        # Meta-analysis statistics
        meta_results = {
            'n_sites': len(comparisons),
            'skema_mean_accuracy': np.mean(skema_accuracies),
            'skema_std_accuracy': np.std(skema_accuracies),
            'our_mean_accuracy': np.mean(our_accuracies),
            'our_std_accuracy': np.std(our_accuracies),
            'skema_mean_f1': np.mean(skema_f1s),
            'our_mean_f1': np.mean(our_f1s),
            'accuracy_effect_size': (np.mean(our_accuracies) - np.mean(skema_accuracies)) / np.sqrt((np.std(our_accuracies)**2 + np.std(skema_accuracies)**2) / 2),
            'sites_favoring_ours': sum(1 for comp in comparisons if comp.our_results['accuracy'] > comp.skema_results['accuracy']),
            'sites_favoring_skema': sum(1 for comp in comparisons if comp.skema_results['accuracy'] > comp.our_results['accuracy'])
        }
        
        # Combined p-value using Fisher's method
        all_p_values = []
        for comp in comparisons:
            for test in comp.statistical_tests:
                if test.test_name == "Paired t-test (Error Comparison)":
                    all_p_values.append(test.p_value)
        
        if all_p_values:
            combined_stat = -2 * sum(np.log(p) for p in all_p_values)
            combined_p = 1 - stats.chi2.cdf(combined_stat, 2 * len(all_p_values))
            meta_results['combined_p_value'] = combined_p
        
        return meta_results
    
    def _print_site_summary(self, comparison: BenchmarkComparison) -> None:
        """Print summary for a single site comparison."""
        
        print(f"   📊 SKEMA: Accuracy={comparison.skema_results['accuracy']:.3f}, F1={comparison.skema_results['f1_score']:.3f}")
        print(f"   🚀 Ours:  Accuracy={comparison.our_results['accuracy']:.3f}, F1={comparison.our_results['f1_score']:.3f}")
        print(f"   🔍 Assessment: {comparison.overall_assessment}")
        
        # Show significant tests
        significant_tests = [test for test in comparison.statistical_tests if test.p_value < self.alpha]
        if significant_tests:
            print(f"   ⚠️  Significant tests: {len(significant_tests)}")
    
    def _print_meta_analysis(self, meta: dict[str, Any]) -> None:
        """Print meta-analysis results."""
        
        if not meta:
            print("   No data for meta-analysis")
            return
        
        print(f"   📈 Sites analyzed: {meta['n_sites']}")
        print(f"   📊 SKEMA average accuracy: {meta['skema_mean_accuracy']:.3f} ± {meta['skema_std_accuracy']:.3f}")
        print(f"   🚀 Our average accuracy: {meta['our_mean_accuracy']:.3f} ± {meta['our_std_accuracy']:.3f}")
        print(f"   📈 Effect size (accuracy): {meta['accuracy_effect_size']:.3f}")
        print(f"   🏆 Sites favoring our method: {meta['sites_favoring_ours']}/{meta['n_sites']}")
        
        if 'combined_p_value' in meta:
            print(f"   🔬 Combined statistical significance: p={meta['combined_p_value']:.4f}")
    
    def create_statistical_report(self, comparisons: list[BenchmarkComparison]) -> str:
        """Generate comprehensive statistical report."""
        
        report = """
# SKEMA vs Our Pipeline: Statistical Benchmarking Report

## Executive Summary

This report presents a comprehensive statistical comparison between the SKEMA kelp detection methodology and our pipeline implementation across multiple validation sites.

"""
        
        # Add site-by-site results
        report += "## Site-by-Site Analysis\n\n"
        
        for i, comp in enumerate(comparisons, 1):
            report += f"### Site {i}: {comp.site_name}\n\n"
            
            # Performance metrics table
            report += "| Metric | SKEMA | Our Pipeline | Difference |\n"
            report += "|--------|-------|--------------|------------|\n"
            
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                skema_val = comp.skema_results[metric]
                our_val = comp.our_results[metric]
                diff = our_val - skema_val
                report += f"| {metric.title()} | {skema_val:.3f} | {our_val:.3f} | {diff:+.3f} |\n"
            
            report += "\n"
            
            # Statistical tests
            report += "#### Statistical Tests\n\n"
            
            for test in comp.statistical_tests:
                report += f"**{test.test_name}**\n"
                report += f"- Statistic: {test.statistic:.4f}\n"
                report += f"- P-value: {test.p_value:.4f}\n"
                report += f"- Interpretation: {test.interpretation}\n"
                report += f"- Recommendation: {test.recommendation}\n\n"
            
            # Overall assessment
            report += f"**Overall Assessment**: {comp.overall_assessment}\n\n"
        
        # Meta-analysis
        if len(comparisons) > 1:
            meta = self._perform_meta_analysis(comparisons)
            if meta:
                report += "## Meta-Analysis\n\n"
                report += f"**Number of sites analyzed**: {meta['n_sites']}\n\n"
                report += "**SKEMA Performance**:\n"
                report += f"- Mean accuracy: {meta['skema_mean_accuracy']:.3f} ± {meta['skema_std_accuracy']:.3f}\n"
                report += f"- Mean F1-score: {meta['skema_mean_f1']:.3f}\n\n"
                report += "**Our Pipeline Performance**:\n"
                report += f"- Mean accuracy: {meta['our_mean_accuracy']:.3f} ± {meta['our_std_accuracy']:.3f}\n"
                report += f"- Mean F1-score: {meta['our_mean_f1']:.3f}\n\n"
                report += f"**Effect Size**: {meta['accuracy_effect_size']:.3f}\n\n"
                report += "**Sites Summary**:\n"
                report += f"- Favoring our method: {meta['sites_favoring_ours']}/{meta['n_sites']}\n"
                report += f"- Favoring SKEMA: {meta['sites_favoring_skema']}/{meta['n_sites']}\n\n"
        
        # Conclusions
        report += "## Statistical Conclusions\n\n"
        
        if len(comparisons) > 0:
            overall_our_better = sum(1 for comp in comparisons if comp.our_results['accuracy'] > comp.skema_results['accuracy'])
            if overall_our_better > len(comparisons) / 2:
                report += "Our pipeline demonstrates superior performance across the majority of validation sites.\n\n"
            else:
                report += "SKEMA methodology shows strong performance with our pipeline as a competitive alternative.\n\n"
        
        report += "This analysis provides statistical evidence for method comparison and can guide selection based on specific use case requirements.\n"
        
        return report 
