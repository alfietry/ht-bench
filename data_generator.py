"""
Data generator for various statistical distributions and hypothesis testing scenarios
"""
import numpy as np
from typing import Dict, Any, List, Tuple
import config


class DataGenerator:
    """Generate synthetic data for hypothesis testing"""
    
    def __init__(self, seed: int = config.RANDOM_SEED):
        self.rng = np.random.default_rng(seed)
    
    def generate_normal(self, mean: float, std: float, size: int) -> np.ndarray:
        """Generate normal distribution data"""
        return self.rng.normal(mean, std, size)
    
    def generate_t_distribution(self, df: int, size: int) -> np.ndarray:
        """Generate t-distribution data"""
        return self.rng.standard_t(df, size)
    
    def generate_exponential(self, scale: float, size: int) -> np.ndarray:
        """Generate exponential distribution data"""
        return self.rng.exponential(scale, size)
    
    def generate_uniform(self, low: float, high: float, size: int) -> np.ndarray:
        """Generate uniform distribution data"""
        return self.rng.uniform(low, high, size)
    
    def generate_chi_square(self, df: int, size: int) -> np.ndarray:
        """Generate chi-square distribution data"""
        return self.rng.chisquare(df, size)
    
    def generate_one_sample_t_test(self, sample_size: int, 
                                   true_mean: float = 10,
                                   std: float = 2,
                                   null_mean: float = 10) -> Dict[str, Any]:
        """Generate data for one-sample t-test"""
        sample = self.generate_normal(true_mean, std, sample_size)
        
        return {
            "test_type": "one_sample_t_test",
            "sample1": sample,
            "population_mean": null_mean,
            "true_effect": true_mean - null_mean,
            "metadata": {
                "sample_size": sample_size,
                "true_mean": true_mean,
                "std": std,
                "null_mean": null_mean
            }
        }
    
    def generate_two_sample_t_test(self, sample_size1: int, sample_size2: int,
                                   mean1: float = 10, mean2: float = 12,
                                   std1: float = 2, std2: float = 2,
                                   paired: bool = False) -> Dict[str, Any]:
        """Generate data for two-sample t-test"""
        sample1 = self.generate_normal(mean1, std1, sample_size1)
        
        if paired:
            # Generate paired data with correlation
            sample2 = sample1 + self.generate_normal(mean2 - mean1, std2 * 0.5, sample_size1)
        else:
            sample2 = self.generate_normal(mean2, std2, sample_size2)
        
        return {
            "test_type": "paired_t_test" if paired else "two_sample_t_test",
            "sample1": sample1,
            "sample2": sample2,
            "true_effect": mean2 - mean1,
            "paired": paired,
            "metadata": {
                "sample_size1": sample_size1,
                "sample_size2": sample_size2,
                "mean1": mean1,
                "mean2": mean2,
                "std1": std1,
                "std2": std2
            }
        }
    
    def generate_one_sample_z_test(self, sample_size: int,
                                   true_mean: float = 100,
                                   population_std: float = 15,
                                   null_mean: float = 100) -> Dict[str, Any]:
        """Generate data for one-sample z-test"""
        sample = self.generate_normal(true_mean, population_std, sample_size)
        
        return {
            "test_type": "one_sample_z_test",
            "sample1": sample,
            "population_mean": null_mean,
            "population_std": population_std,
            "true_effect": true_mean - null_mean,
            "metadata": {
                "sample_size": sample_size,
                "true_mean": true_mean,
                "population_std": population_std,
                "null_mean": null_mean
            }
        }
    
    def generate_two_sample_z_test(self, sample_size1: int, sample_size2: int,
                                   mean1: float = 100, mean2: float = 105,
                                   std1: float = 15, std2: float = 15) -> Dict[str, Any]:
        """Generate data for two-sample z-test (known variances)"""
        sample1 = self.generate_normal(mean1, std1, sample_size1)
        sample2 = self.generate_normal(mean2, std2, sample_size2)
        
        return {
            "test_type": "two_sample_z_test",
            "sample1": sample1,
            "sample2": sample2,
            "population_std1": std1,
            "population_std2": std2,
            "true_effect": mean2 - mean1,
            "metadata": {
                "sample_size1": sample_size1,
                "sample_size2": sample_size2,
                "mean1": mean1,
                "mean2": mean2,
                "std1": std1,
                "std2": std2
            }
        }
    
    def generate_anova(self, num_groups: int = 3, 
                      samples_per_group: int = 20,
                      group_means: List[float] = None,
                      common_std: float = 2) -> Dict[str, Any]:
        """Generate data for one-way ANOVA"""
        if group_means is None:
            group_means = [10, 12, 11]
        
        if len(group_means) != num_groups:
            raise ValueError("Number of group means must match num_groups")
        
        groups = []
        for mean in group_means:
            group = self.generate_normal(mean, common_std, samples_per_group)
            groups.append(group)
        
        return {
            "test_type": "anova",
            "groups": groups,
            "metadata": {
                "num_groups": num_groups,
                "samples_per_group": samples_per_group,
                "group_means": group_means,
                "common_std": common_std
            }
        }
    
    def generate_chi_square_goodness_of_fit(self, n_samples: int = 100,
                                           expected_proportions: List[float] = None) -> Dict[str, Any]:
        """Generate data for chi-square goodness of fit test"""
        if expected_proportions is None:
            expected_proportions = [0.25, 0.25, 0.25, 0.25]
        
        # Generate actual data with slight deviation
        actual_proportions = np.array(expected_proportions) + self.rng.normal(0, 0.02, len(expected_proportions))
        actual_proportions = np.abs(actual_proportions)
        actual_proportions = actual_proportions / actual_proportions.sum()
        
        # Generate observed counts
        observed = self.rng.multinomial(n_samples, actual_proportions)
        expected = np.array(expected_proportions) * n_samples
        
        return {
            "test_type": "chi_square_goodness_of_fit",
            "observed": observed.tolist(),
            "expected": expected.tolist(),
            "metadata": {
                "n_samples": n_samples,
                "expected_proportions": expected_proportions,
                "actual_proportions": actual_proportions.tolist()
            }
        }
    
    def generate_chi_square_independence(self, contingency_shape: Tuple[int, int] = (2, 2),
                                        n_samples: int = 200,
                                        association_strength: float = 0.0) -> Dict[str, Any]:
        """Generate data for chi-square test of independence
        
        Args:
            contingency_shape: Shape of contingency table (rows, cols)
            n_samples: Total number of observations
            association_strength: Strength of association (0=independent, 1=strong association)
        """
        rows, cols = contingency_shape
        
        if association_strength == 0:
            # Generate independent data
            row_probs = self.rng.dirichlet(np.ones(rows))
            col_probs = self.rng.dirichlet(np.ones(cols))
            cell_probs = np.outer(row_probs, col_probs)
        else:
            # Generate associated data
            cell_probs = self.rng.dirichlet(np.ones(rows * cols)).reshape(rows, cols)
            # Adjust to create association
            for i in range(min(rows, cols)):
                cell_probs[i, i] *= (1 + association_strength)
            cell_probs = cell_probs / cell_probs.sum()
        
        # Generate contingency table
        contingency = self.rng.multinomial(n_samples, cell_probs.flatten()).reshape(rows, cols)
        
        return {
            "test_type": "chi_square_independence",
            "contingency_table": contingency.tolist(),
            "metadata": {
                "shape": contingency_shape,
                "n_samples": n_samples,
                "association_strength": association_strength
            }
        }
    
    def generate_scenario(self, test_type: str, **kwargs) -> Dict[str, Any]:
        """Generate data for any test type"""
        generators = {
            "one_sample_t_test": self.generate_one_sample_t_test,
            "two_sample_t_test": lambda **kw: self.generate_two_sample_t_test(paired=False, **kw),
            "paired_t_test": lambda **kw: self.generate_two_sample_t_test(paired=True, **kw),
            "one_sample_z_test": self.generate_one_sample_z_test,
            "two_sample_z_test": self.generate_two_sample_z_test,
            "anova": self.generate_anova,
            "chi_square_goodness_of_fit": self.generate_chi_square_goodness_of_fit,
            "chi_square_independence": self.generate_chi_square_independence,
        }
        
        if test_type not in generators:
            raise ValueError(f"Unknown test type: {test_type}")
        
        return generators[test_type](**kwargs)
    
    def generate_batch(self, test_types: List[str], 
                      sample_sizes: List[int] = None) -> List[Dict[str, Any]]:
        """Generate multiple scenarios"""
        if sample_sizes is None:
            sample_sizes = config.SAMPLE_SIZES
        
        scenarios = []
        for test_type in test_types:
            for sample_size in sample_sizes:
                try:
                    if test_type in ["one_sample_t_test", "one_sample_z_test"]:
                        scenario = self.generate_scenario(test_type, sample_size=sample_size)
                    elif test_type in ["two_sample_t_test", "two_sample_z_test", "paired_t_test"]:
                        scenario = self.generate_scenario(
                            test_type, 
                            sample_size1=sample_size,
                            sample_size2=sample_size
                        )
                    elif test_type == "anova":
                        scenario = self.generate_scenario(
                            test_type,
                            samples_per_group=sample_size
                        )
                    elif test_type in ["chi_square_goodness_of_fit", "chi_square_independence"]:
                        scenario = self.generate_scenario(
                            test_type,
                            n_samples=sample_size
                        )
                    else:
                        continue
                    
                    scenarios.append(scenario)
                except Exception as e:
                    print(f"Error generating {test_type} with size {sample_size}: {e}")
        
        return scenarios


def create_test_context(test_type: str) -> str:
    """Create contextual description for test type"""
    contexts = {
        "one_sample_t_test": "A researcher wants to test if a sample mean differs from a known population value.",
        "two_sample_t_test": "A researcher wants to compare the means of two independent groups.",
        "paired_t_test": "A researcher wants to compare measurements from the same subjects under two conditions.",
        "one_sample_z_test": "A researcher wants to test if a sample mean differs from a population mean (population std known).",
        "two_sample_z_test": "A researcher wants to compare two population means (population stds known).",
        "anova": "A researcher wants to compare means across multiple groups.",
        "chi_square_goodness_of_fit": "A researcher wants to test if observed frequencies match expected frequencies.",
        "chi_square_independence": "A researcher wants to test if two categorical variables are independent.",
    }
    return contexts.get(test_type, "")
