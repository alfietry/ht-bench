"""
Evaluation metrics for comparing LLM responses to ground truth
"""
import numpy as np
from typing import Dict, Any, List, Optional
from .response_parser import ParsedResponse
from . import config


class EvaluationMetrics:
    """Compute evaluation metrics with hierarchical dependencies"""
    
    @staticmethod
    def evaluate_response(parsed: Dict[str, Any], 
                         ground_truth: Dict[str, Any],
                         strict_mode: bool = True) -> Dict[str, Any]:
        """
        Evaluate with hierarchical dependencies:
        - Test selection must be correct for downstream metrics to count
        - P-value accuracy only valid if test statistic is correct
        
        Args:
            strict_mode: If True, downstream metrics require upstream correctness
        """
        metrics = {
            "test_correct": False,
            "test_statistic_correct": False,
            "p_value_correct": False,
            "decision_correct": False,
            "complete_pipeline_correct": False
        }
        
        # Level 1: Test Selection
        if parsed.get("test_type") == ground_truth.get("test_type"):
            metrics["test_correct"] = True
        else:
            if strict_mode:
                # Wrong test invalidates all downstream metrics
                return metrics
        
        # Level 2: Test Statistic (only if test correct in strict mode)
        if strict_mode and not metrics["test_correct"]:
            return metrics
            
        t_stat_parsed = parsed.get("test_statistic")
        t_stat_truth = ground_truth.get("test_statistic")
        
        if t_stat_parsed is not None and t_stat_truth is not None:
            if abs(t_stat_parsed - t_stat_truth) < config.EVALUATION["test_stat_tolerance"]:
                metrics["test_statistic_correct"] = True
        
        # Level 3: P-Value (only if test statistic correct in strict mode)
        if strict_mode and not metrics["test_statistic_correct"]:
            return metrics
            
        p_val_parsed = parsed.get("p_value")
        p_val_truth = ground_truth.get("p_value")
        
        if p_val_parsed is not None and p_val_truth is not None:
            if abs(p_val_parsed - p_val_truth) < config.EVALUATION["p_value_tolerance"]:
                metrics["p_value_correct"] = True
        
        # Level 4: Decision (requires correct p-value in strict mode)
        if strict_mode and not metrics["p_value_correct"]:
            return metrics
            
        decision_parsed = parsed.get("decision")
        decision_truth = ground_truth.get("decision")
        
        if decision_parsed == decision_truth:
            metrics["decision_correct"] = True
        
        # Complete pipeline correctness
        metrics["complete_pipeline_correct"] = all([
            metrics["test_correct"],
            metrics["test_statistic_correct"],
            metrics["p_value_correct"],
            metrics["decision_correct"]
        ])
        
        return metrics

    @staticmethod
    def test_method_accuracy(predicted: Optional[str], 
                            ground_truth: str) -> float:
        """Check if test method matches exactly"""
        if predicted is None:
            return 0.0
        
        # Normalize strings
        pred_norm = predicted.lower().replace("-", "_").replace(" ", "_")
        gt_norm = ground_truth.lower().replace("-", "_").replace(" ", "_")
        
        # Exact match
        if pred_norm == gt_norm:
            return 1.0
        
        # Partial matches (e.g., "t_test" in "two_sample_t_test")
        key_terms = ["t_test", "z_test", "anova", "chi_square", "paired", "one_sample", "two_sample"]
        pred_terms = [term for term in key_terms if term in pred_norm]
        gt_terms = [term for term in key_terms if term in gt_norm]
        
        if pred_terms and gt_terms:
            overlap = len(set(pred_terms) & set(gt_terms))
            return overlap / len(gt_terms)
        
        return 0.0
    
    @staticmethod
    def numerical_accuracy(predicted: Optional[float], 
                          ground_truth: float,
                          tolerance: float) -> Dict[str, Any]:
        """Check numerical accuracy within tolerance"""
        if predicted is None:
            return {
                "exact_match": False,
                "within_tolerance": False,
                "error": None,
                "relative_error": None
            }
        
        error = abs(predicted - ground_truth)
        relative_error = error / abs(ground_truth) if ground_truth != 0 else error
        
        return {
            "exact_match": predicted == ground_truth,
            "within_tolerance": error <= tolerance,
            "error": float(error),
            "relative_error": float(relative_error),
            "predicted": float(predicted),
            "ground_truth": float(ground_truth)
        }
    
    @staticmethod
    def decision_accuracy(predicted: Optional[str], 
                         ground_truth: str) -> Dict[str, Any]:
        """Check if hypothesis test decision is correct"""
        if predicted is None:
            return {
                "correct": False,
                "predicted": None,
                "ground_truth": ground_truth
            }
        
        # Normalize decision strings
        pred_norm = predicted.lower().replace(" ", "_")
        gt_norm = ground_truth.lower().replace(" ", "_")
        
        correct = pred_norm == gt_norm
        
        return {
            "correct": correct,
            "predicted": predicted,
            "ground_truth": ground_truth
        }
    
    @staticmethod
    def p_value_accuracy(predicted: Optional[float],
                        ground_truth: float) -> Dict[str, Any]:
        """Evaluate p-value accuracy"""
        result = EvaluationMetrics.numerical_accuracy(
            predicted, ground_truth, 
            config.EVALUATION["p_value_tolerance"]
        )
        
        # Additional p-value specific checks
        if predicted is not None:
            result["valid_range"] = 0 <= predicted <= 1
            result["correct_significance"] = (
                (predicted < 0.05 and ground_truth < 0.05) or
                (predicted >= 0.05 and ground_truth >= 0.05)
            )
        else:
            result["valid_range"] = False
            result["correct_significance"] = False
        
        return result
    
    @staticmethod
    def test_statistic_accuracy(predicted: Optional[float],
                               ground_truth: float) -> Dict[str, Any]:
        """Evaluate test statistic accuracy"""
        return EvaluationMetrics.numerical_accuracy(
            predicted, ground_truth,
            config.EVALUATION["test_statistic_tolerance"]
        )
    
    @staticmethod
    def reasoning_quality_score(parsed: ParsedResponse,
                               ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Score reasoning quality based on rubric"""
        rubric = config.EVALUATION["reasoning_rubric"]
        scores = {}
        
        # Hypothesis clarity (1 point)
        if parsed.hypotheses is not None:
            scores["hypothesis_clarity"] = rubric["hypothesis_clarity"]
        else:
            scores["hypothesis_clarity"] = 0
        
        # Test justification (1 point)
        if parsed.test_method is not None:
            correct = EvaluationMetrics.test_method_accuracy(
                parsed.test_method, ground_truth["test_method"]
            )
            scores["test_justification"] = rubric["test_justification"] * correct
        else:
            scores["test_justification"] = 0
        
        # Assumption checking (1 point)
        if parsed.assumptions is not None:
            scores["assumption_checking"] = rubric["assumption_checking"]
        else:
            scores["assumption_checking"] = 0
        
        # Correct interpretation (1 point)
        if parsed.decision is not None:
            decision_result = EvaluationMetrics.decision_accuracy(
                parsed.decision, ground_truth["decision"]
            )
            scores["correct_interpretation"] = rubric["correct_interpretation"] if decision_result["correct"] else 0
        else:
            scores["correct_interpretation"] = 0
        
        # Statistical rigor (1 point) - based on presence of key components
        components = [
            parsed.test_statistic is not None,
            parsed.p_value is not None,
            parsed.decision is not None
        ]
        rigor_score = sum(components) / len(components)
        scores["statistical_rigor"] = rubric["statistical_rigor"] * rigor_score
        
        total_score = sum(scores.values())
        max_score = sum(rubric.values())
        
        return {
            "scores": scores,
            "total": total_score,
            "max": max_score,
            "percentage": (total_score / max_score * 100) if max_score > 0 else 0
        }
    
    @staticmethod
    def detect_hallucinations(parsed: ParsedResponse,
                             ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential hallucinations in response"""
        hallucinations = []
        
        # Check p-value range
        if parsed.p_value is not None and (parsed.p_value < 0 or parsed.p_value > 1):
            hallucinations.append({
                "type": "invalid_p_value",
                "value": parsed.p_value,
                "message": "p-value outside valid range [0, 1]"
            })
        
        # Check test statistic magnitude
        if parsed.test_statistic is not None and ground_truth.get("test_statistic"):
            gt_stat = ground_truth["test_statistic"]
            if abs(parsed.test_statistic - gt_stat) > abs(gt_stat) * 100:
                hallucinations.append({
                    "type": "unrealistic_test_statistic",
                    "predicted": parsed.test_statistic,
                    "ground_truth": gt_stat,
                    "message": f"Test statistic differs by >100x from ground truth"
                })
        
        # Check decision consistency with p-value
        if parsed.p_value is not None and parsed.decision is not None:
            should_reject = parsed.p_value < config.EVALUATION["significance_level"]
            does_reject = "reject" in parsed.decision.lower() and "fail" not in parsed.decision.lower()
            
            if should_reject != does_reject:
                hallucinations.append({
                    "type": "inconsistent_decision",
                    "p_value": parsed.p_value,
                    "decision": parsed.decision,
                    "message": f"Decision inconsistent with p-value"
                })
        
        return {
            "has_hallucinations": len(hallucinations) > 0,
            "count": len(hallucinations),
            "details": hallucinations
        }
    
    @staticmethod
    def comprehensive_evaluation(parsed: ParsedResponse,
                                ground_truth: Dict[str, Any],
                                raw_response: str = "") -> Dict[str, Any]:
        """Perform comprehensive evaluation"""
        evaluation = {
            "test_method": EvaluationMetrics.test_method_accuracy(
                parsed.test_method, ground_truth["test_method"]
            ),
            "p_value": EvaluationMetrics.p_value_accuracy(
                parsed.p_value, ground_truth["p_value"]
            ),
            "test_statistic": EvaluationMetrics.test_statistic_accuracy(
                parsed.test_statistic, ground_truth["test_statistic"]
            ),
            "decision": EvaluationMetrics.decision_accuracy(
                parsed.decision, ground_truth["decision"]
            ),
            "reasoning_quality": EvaluationMetrics.reasoning_quality_score(
                parsed, ground_truth
            ),
            "hallucinations": EvaluationMetrics.detect_hallucinations(
                parsed, ground_truth
            ),
            "completeness": {
                "has_hypotheses": parsed.hypotheses is not None,
                "has_test_method": parsed.test_method is not None,
                "has_test_statistic": parsed.test_statistic is not None,
                "has_p_value": parsed.p_value is not None,
                "has_decision": parsed.decision is not None,
            }
        }
        
        # Calculate overall score
        key_metrics = [
            evaluation["test_method"],  # 0-1
            1.0 if evaluation["p_value"]["within_tolerance"] else 0.0,
            1.0 if evaluation["test_statistic"]["within_tolerance"] else 0.0,
            1.0 if evaluation["decision"]["correct"] else 0.0,
        ]
        evaluation["overall_accuracy"] = np.mean(key_metrics)
        
        return evaluation


class BenchmarkAggregator:
    """Aggregate results across multiple evaluations"""
    
    @staticmethod
    def aggregate_by_model(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Aggregate results by model"""
        model_results = {}
        
        for result in results:
            model = result["model"]
            if model not in model_results:
                model_results[model] = []
            model_results[model].append(result)
        
        aggregated = {}
        for model, model_res in model_results.items():
            aggregated[model] = BenchmarkAggregator._compute_statistics(model_res)
        
        return aggregated
    
    @staticmethod
    def aggregate_by_prompt_type(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Aggregate results by prompt type"""
        prompt_results = {}
        
        for result in results:
            prompt_type = result["prompt_type"]
            if prompt_type not in prompt_results:
                prompt_results[prompt_type] = []
            prompt_results[prompt_type].append(result)
        
        aggregated = {}
        for prompt_type, prompt_res in prompt_results.items():
            aggregated[prompt_type] = BenchmarkAggregator._compute_statistics(prompt_res)
        
        return aggregated
    
    @staticmethod
    def aggregate_by_test_type(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Aggregate results by test type"""
        test_results = {}
        
        for result in results:
            test_type = result.get("test_type", "unknown")
            if test_type not in test_results:
                test_results[test_type] = []
            test_results[test_type].append(result)
        
        aggregated = {}
        for test_type, test_res in test_results.items():
            aggregated[test_type] = BenchmarkAggregator._compute_statistics(test_res)
        
        return aggregated
    
    @staticmethod
    def _compute_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics for a group of results"""
        n = len(results)
        
        if n == 0:
            return {}
        
        # Extract metrics
        overall_accuracies = [r.get("evaluation", {}).get("overall_accuracy", 0) for r in results]
        test_method_accuracies = [r.get("evaluation", {}).get("test_method", 0) for r in results]
        decision_accuracies = [
            1.0 if r.get("evaluation", {}).get("decision", {}).get("correct", False) else 0.0 
            for r in results
        ]
        p_value_accuracies = [
            1.0 if r.get("evaluation", {}).get("p_value", {}).get("within_tolerance", False) else 0.0
            for r in results
        ]
        
        reasoning_scores = [
            r.get("evaluation", {}).get("reasoning_quality", {}).get("percentage", 0)
            for r in results
        ]
        
        hallucination_rates = [
            1.0 if r.get("evaluation", {}).get("hallucinations", {}).get("has_hallucinations", False) else 0.0
            for r in results
        ]

        latencies = [
            r.get("latency_seconds") for r in results
            if isinstance(r.get("latency_seconds"), (int, float))
        ]
        
        return {
            "n_samples": n,
            "overall_accuracy": {
                "mean": np.mean(overall_accuracies),
                "std": np.std(overall_accuracies),
                "min": np.min(overall_accuracies),
                "max": np.max(overall_accuracies)
            },
            "test_method_accuracy": {
                "mean": np.mean(test_method_accuracies),
                "exact_match_rate": sum(1 for x in test_method_accuracies if x == 1.0) / n
            },
            "decision_accuracy": {
                "mean": np.mean(decision_accuracies),
                "correct_rate": sum(decision_accuracies) / n
            },
            "p_value_accuracy": {
                "mean": np.mean(p_value_accuracies),
                "within_tolerance_rate": sum(p_value_accuracies) / n
            },
            "reasoning_quality": {
                "mean": np.mean(reasoning_scores),
                "std": np.std(reasoning_scores)
            },
            "hallucination_rate": np.mean(hallucination_rates),
            "latency": {
                "mean": float(np.mean(latencies)) if latencies else None,
                "median": float(np.median(latencies)) if latencies else None,
                "min": float(np.min(latencies)) if latencies else None,
                "max": float(np.max(latencies)) if latencies else None,
            }
        }
    
    @staticmethod
    def create_comparison_matrix(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comparison matrix across models and prompt types"""
        models = list(set(r["model"] for r in results))
        prompt_types = list(set(r["prompt_type"] for r in results))
        
        matrix = {}
        for model in models:
            matrix[model] = {}
            for prompt_type in prompt_types:
                filtered = [r for r in results 
                          if r["model"] == model and r["prompt_type"] == prompt_type]
                if filtered:
                    matrix[model][prompt_type] = BenchmarkAggregator._compute_statistics(filtered)
        
        return matrix
    
def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate comprehensive metrics including F1 score for conclusion correctness."""
    total = len(results)
    if total == 0:
        return {
            "accuracy": 0.0,
            "p_value_accuracy": 0.0,
            "test_selection_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "total_scenarios": 0
        }
    
    # Extract metrics from nested evaluation structure
    correct = 0
    p_value_correct = 0
    test_correct = 0
    
    for r in results:
        # Handle nested evaluation structure
        eval_data = r.get("evaluation", {})
        if isinstance(eval_data, dict):
            # Overall accuracy
            if eval_data.get("overall_accuracy", 0) >= 0.75:  # Consider >=75% as correct
                correct += 1
            # P-value accuracy
            if eval_data.get("p_value", {}).get("within_tolerance", False):
                p_value_correct += 1
            # Test selection
            if eval_data.get("test_method", 0) == 1.0:
                test_correct += 1
    
    # F1 Score Calculation - for conclusion correctness (reject H0 or not)
    # Extract decision from nested structure
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    
    for r in results:
        ground_truth = r.get("ground_truth", {})
        eval_data = r.get("evaluation", {})
        decision_data = eval_data.get("decision", {})
        
        # Get ground truth decision (handle None)
        gt_decision = ground_truth.get("decision", "")
        gt_decision = gt_decision.lower() if gt_decision else ""
        
        # Get LLM decision (handle None)
        llm_decision = decision_data.get("predicted", "") if isinstance(decision_data, dict) else ""
        llm_decision = llm_decision.lower() if llm_decision else ""
        
        # Normalize decisions
        gt_rejects = "reject" in gt_decision and "fail" not in gt_decision
        llm_rejects = "reject" in llm_decision and "fail" not in llm_decision
        
        # Calculate confusion matrix
        if gt_rejects and llm_rejects:
            tp += 1
        elif not gt_rejects and llm_rejects:
            fp += 1
        elif gt_rejects and not llm_rejects:
            fn += 1
        else:
            tn += 1
    
    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1: Harmonic mean of precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": correct / total,
        "p_value_accuracy": p_value_correct / total,
        "test_selection_accuracy": test_correct / total,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "total_scenarios": total,
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn
        }
    }
