"""
Statistical engine for computing ground truth results
"""
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Tuple
import config


class StatisticalEngine:
    """Compute ground truth statistical test results"""
    
    @staticmethod
    def compute_one_sample_t_test(sample: np.ndarray, 
                                  population_mean: float,
                                  alternative: str = "two-sided") -> Dict[str, Any]:
        """Compute one-sample t-test"""
        n = len(sample)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        
        # Test statistic
        t_stat = (sample_mean - population_mean) / (sample_std / np.sqrt(n))
        
        # p-value
        if alternative == "two-sided":
            p_value = 2 * stats.t.sf(np.abs(t_stat), df=n-1)
        elif alternative == "greater":
            p_value = stats.t.sf(t_stat, df=n-1)
        else:  # less
            p_value = stats.t.cdf(t_stat, df=n-1)
        
        # Critical value (two-sided)
        critical_value = stats.t.ppf(1 - config.EVALUATION["significance_level"]/2, df=n-1)
        
        # Confidence interval
        margin = critical_value * (sample_std / np.sqrt(n))
        ci = (sample_mean - margin, sample_mean + margin)
        
        # Decision
        decision = "reject_H0" if p_value < config.EVALUATION["significance_level"] else "fail_to_reject_H0"
        
        return {
            "test_method": "one_sample_t_test",
            "test_statistic": float(t_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": n - 1,
            "critical_value": float(critical_value),
            "confidence_interval": ci,
            "decision": decision,
            "sample_mean": float(sample_mean),
            "sample_std": float(sample_std),
            "hypotheses": {
                "H0": f"μ = {population_mean}",
                "H1": f"μ ≠ {population_mean}" if alternative == "two-sided" else 
                      f"μ > {population_mean}" if alternative == "greater" else f"μ < {population_mean}"
            }
        }
    
    @staticmethod
    def compute_two_sample_t_test(sample1: np.ndarray, sample2: np.ndarray,
                                  equal_var: bool = True,
                                  alternative: str = "two-sided") -> Dict[str, Any]:
        """Compute independent two-sample t-test"""
        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
        
        if equal_var:
            # Pooled variance
            pooled_var = ((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2)
            se = np.sqrt(pooled_var * (1/n1 + 1/n2))
            df = n1 + n2 - 2
        else:
            # Welch's t-test
            se = np.sqrt(std1**2/n1 + std2**2/n2)
            df = (std1**2/n1 + std2**2/n2)**2 / (
                (std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1)
            )
        
        # Test statistic
        t_stat = (mean1 - mean2) / se
        
        # p-value
        if alternative == "two-sided":
            p_value = 2 * stats.t.sf(np.abs(t_stat), df=df)
        elif alternative == "greater":
            p_value = stats.t.sf(t_stat, df=df)
        else:
            p_value = stats.t.cdf(t_stat, df=df)
        
        # Critical value
        critical_value = stats.t.ppf(1 - config.EVALUATION["significance_level"]/2, df=df)
        
        # Confidence interval for difference
        margin = critical_value * se
        ci = (mean1 - mean2 - margin, mean1 - mean2 + margin)
        
        decision = "reject_H0" if p_value < config.EVALUATION["significance_level"] else "fail_to_reject_H0"
        
        return {
            "test_method": "two_sample_t_test" + ("_equal_var" if equal_var else "_welch"),
            "test_statistic": float(t_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": float(df),
            "critical_value": float(critical_value),
            "confidence_interval": ci,
            "decision": decision,
            "mean_difference": float(mean1 - mean2),
            "sample1_mean": float(mean1),
            "sample2_mean": float(mean2),
            "sample1_std": float(std1),
            "sample2_std": float(std2),
            "hypotheses": {
                "H0": "μ1 = μ2",
                "H1": "μ1 ≠ μ2" if alternative == "two-sided" else 
                      "μ1 > μ2" if alternative == "greater" else "μ1 < μ2"
            }
        }
    
    @staticmethod
    def compute_paired_t_test(sample1: np.ndarray, sample2: np.ndarray,
                             alternative: str = "two-sided") -> Dict[str, Any]:
        """Compute paired t-test"""
        differences = sample1 - sample2
        n = len(differences)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        # Test statistic
        t_stat = mean_diff / (std_diff / np.sqrt(n))
        
        # p-value
        if alternative == "two-sided":
            p_value = 2 * stats.t.sf(np.abs(t_stat), df=n-1)
        elif alternative == "greater":
            p_value = stats.t.sf(t_stat, df=n-1)
        else:
            p_value = stats.t.cdf(t_stat, df=n-1)
        
        # Critical value
        critical_value = stats.t.ppf(1 - config.EVALUATION["significance_level"]/2, df=n-1)
        
        # Confidence interval
        margin = critical_value * (std_diff / np.sqrt(n))
        ci = (mean_diff - margin, mean_diff + margin)
        
        decision = "reject_H0" if p_value < config.EVALUATION["significance_level"] else "fail_to_reject_H0"
        
        return {
            "test_method": "paired_t_test",
            "test_statistic": float(t_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": n - 1,
            "critical_value": float(critical_value),
            "confidence_interval": ci,
            "decision": decision,
            "mean_difference": float(mean_diff),
            "std_difference": float(std_diff),
            "hypotheses": {
                "H0": "μd = 0 (no difference)",
                "H1": "μd ≠ 0" if alternative == "two-sided" else 
                      "μd > 0" if alternative == "greater" else "μd < 0"
            }
        }
    
    @staticmethod
    def compute_one_sample_z_test(sample: np.ndarray, 
                                  population_mean: float,
                                  population_std: float,
                                  alternative: str = "two-sided") -> Dict[str, Any]:
        """Compute one-sample z-test"""
        n = len(sample)
        sample_mean = np.mean(sample)
        
        # Test statistic
        z_stat = (sample_mean - population_mean) / (population_std / np.sqrt(n))
        
        # p-value
        if alternative == "two-sided":
            p_value = 2 * stats.norm.sf(np.abs(z_stat))
        elif alternative == "greater":
            p_value = stats.norm.sf(z_stat)
        else:
            p_value = stats.norm.cdf(z_stat)
        
        # Critical value
        critical_value = stats.norm.ppf(1 - config.EVALUATION["significance_level"]/2)
        
        # Confidence interval
        margin = critical_value * (population_std / np.sqrt(n))
        ci = (sample_mean - margin, sample_mean + margin)
        
        decision = "reject_H0" if p_value < config.EVALUATION["significance_level"] else "fail_to_reject_H0"
        
        return {
            "test_method": "one_sample_z_test",
            "test_statistic": float(z_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": None,
            "critical_value": float(critical_value),
            "confidence_interval": ci,
            "decision": decision,
            "sample_mean": float(sample_mean),
            "hypotheses": {
                "H0": f"μ = {population_mean}",
                "H1": f"μ ≠ {population_mean}" if alternative == "two-sided" else 
                      f"μ > {population_mean}" if alternative == "greater" else f"μ < {population_mean}"
            }
        }
    
    @staticmethod
    def compute_two_sample_z_test(sample1: np.ndarray, sample2: np.ndarray,
                                  std1: float, std2: float,
                                  alternative: str = "two-sided") -> Dict[str, Any]:
        """Compute two-sample z-test (known population stds)"""
        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        
        # Test statistic
        se = np.sqrt(std1**2/n1 + std2**2/n2)
        z_stat = (mean1 - mean2) / se
        
        # p-value
        if alternative == "two-sided":
            p_value = 2 * stats.norm.sf(np.abs(z_stat))
        elif alternative == "greater":
            p_value = stats.norm.sf(z_stat)
        else:
            p_value = stats.norm.cdf(z_stat)
        
        # Critical value
        critical_value = stats.norm.ppf(1 - config.EVALUATION["significance_level"]/2)
        
        # Confidence interval
        margin = critical_value * se
        ci = (mean1 - mean2 - margin, mean1 - mean2 + margin)
        
        decision = "reject_H0" if p_value < config.EVALUATION["significance_level"] else "fail_to_reject_H0"
        
        return {
            "test_method": "two_sample_z_test",
            "test_statistic": float(z_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": None,
            "critical_value": float(critical_value),
            "confidence_interval": ci,
            "decision": decision,
            "mean_difference": float(mean1 - mean2),
            "hypotheses": {
                "H0": "μ1 = μ2",
                "H1": "μ1 ≠ μ2" if alternative == "two-sided" else 
                      "μ1 > μ2" if alternative == "greater" else "μ1 < μ2"
            }
        }
    
    @staticmethod
    def compute_anova(groups: list) -> Dict[str, Any]:
        """Compute one-way ANOVA"""
        # Convert to numpy arrays
        groups = [np.array(g) for g in groups]
        
        k = len(groups)  # number of groups
        n = sum(len(g) for g in groups)  # total sample size
        
        # Grand mean
        grand_mean = np.mean(np.concatenate(groups))
        
        # Between-group sum of squares
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        df_between = k - 1
        ms_between = ss_between / df_between
        
        # Within-group sum of squares
        ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)
        df_within = n - k
        ms_within = ss_within / df_within
        
        # F-statistic
        f_stat = ms_between / ms_within
        
        # p-value
        p_value = stats.f.sf(f_stat, df_between, df_within)
        
        # Critical value
        critical_value = stats.f.ppf(1 - config.EVALUATION["significance_level"], 
                                     df_between, df_within)
        
        decision = "reject_H0" if p_value < config.EVALUATION["significance_level"] else "fail_to_reject_H0"
        
        return {
            "test_method": "one_way_anova",
            "test_statistic": float(f_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": (df_between, df_within),
            "critical_value": float(critical_value),
            "confidence_interval": None,
            "decision": decision,
            "ss_between": float(ss_between),
            "ss_within": float(ss_within),
            "ms_between": float(ms_between),
            "ms_within": float(ms_within),
            "group_means": [float(np.mean(g)) for g in groups],
            "hypotheses": {
                "H0": "All group means are equal",
                "H1": "At least one group mean differs"
            }
        }
    
    @staticmethod
    def compute_chi_square_goodness_of_fit(observed: np.ndarray,
                                          expected: np.ndarray) -> Dict[str, Any]:
        """Compute chi-square goodness of fit test"""
        observed = np.array(observed)
        expected = np.array(expected)
        
        # Chi-square statistic
        chi_stat = np.sum((observed - expected)**2 / expected)
        
        # Degrees of freedom
        df = len(observed) - 1
        
        # p-value
        p_value = stats.chi2.sf(chi_stat, df)
        
        # Critical value
        critical_value = stats.chi2.ppf(1 - config.EVALUATION["significance_level"], df)
        
        decision = "reject_H0" if p_value < config.EVALUATION["significance_level"] else "fail_to_reject_H0"
        
        return {
            "test_method": "chi_square_goodness_of_fit",
            "test_statistic": float(chi_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": df,
            "critical_value": float(critical_value),
            "confidence_interval": None,
            "decision": decision,
            "observed": observed.tolist(),
            "expected": expected.tolist(),
            "hypotheses": {
                "H0": "Observed frequencies match expected frequencies",
                "H1": "Observed frequencies do not match expected frequencies"
            }
        }
    
    @staticmethod
    def compute_chi_square_independence(contingency_table: np.ndarray) -> Dict[str, Any]:
        """Compute chi-square test of independence"""
        contingency = np.array(contingency_table)
        
        # Compute chi-square using scipy
        chi_stat, p_value, df, expected = stats.chi2_contingency(contingency)
        
        # Critical value
        critical_value = stats.chi2.ppf(1 - config.EVALUATION["significance_level"], df)
        
        decision = "reject_H0" if p_value < config.EVALUATION["significance_level"] else "fail_to_reject_H0"
        
        return {
            "test_method": "chi_square_independence",
            "test_statistic": float(chi_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": int(df),
            "critical_value": float(critical_value),
            "confidence_interval": None,
            "decision": decision,
            "observed": contingency.tolist(),
            "expected": expected.tolist(),
            "hypotheses": {
                "H0": "Variables are independent",
                "H1": "Variables are not independent"
            }
        }
    
    @staticmethod
    def compute_ground_truth(data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute ground truth for any test type"""
        test_type = data["test_type"]
        
        if test_type == "one_sample_t_test":
            return StatisticalEngine.compute_one_sample_t_test(
                data["sample1"], data["population_mean"]
            )
        elif test_type == "two_sample_t_test":
            return StatisticalEngine.compute_two_sample_t_test(
                data["sample1"], data["sample2"]
            )
        elif test_type == "paired_t_test":
            return StatisticalEngine.compute_paired_t_test(
                data["sample1"], data["sample2"]
            )
        elif test_type == "one_sample_z_test":
            return StatisticalEngine.compute_one_sample_z_test(
                data["sample1"], data["population_mean"], data["population_std"]
            )
        elif test_type == "two_sample_z_test":
            return StatisticalEngine.compute_two_sample_z_test(
                data["sample1"], data["sample2"],
                data["population_std1"], data["population_std2"]
            )
        elif test_type == "anova":
            return StatisticalEngine.compute_anova(data["groups"])
        elif test_type == "chi_square_goodness_of_fit":
            return StatisticalEngine.compute_chi_square_goodness_of_fit(
                data["observed"], data["expected"]
            )
        elif test_type == "chi_square_independence":
            return StatisticalEngine.compute_chi_square_independence(
                data["contingency_table"]
            )
        else:
            raise ValueError(f"Unknown test type: {test_type}")
