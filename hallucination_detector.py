from typing import Dict, List, Any
import re
import numpy as np

class HallucinationDetector:
    """Detect multiple types of hallucinations in LLM statistical reasoning"""
    
    HALLUCINATION_TYPES = {
        "structural": [
            "missing_required_field",
            "invalid_json_format",
            "unparseable_output"
        ],
        "numerical": [
            "p_value_out_of_range",
            "negative_test_statistic_for_positive_test",
            "test_statistic_magnitude_implausible",
            "contradictory_p_value_and_decision"
        ],
        "logical": [
            "decision_contradicts_p_value",
            "wrong_tail_test_for_hypothesis",
            "incorrect_degrees_of_freedom",
            "wrong_statistical_test_cited"
        ],
        "reasoning": [
            "contradictory_explanation",
            "fabricated_formula",
            "misinterpreted_null_hypothesis",
            "confidence_interval_direction_error"
        ]
    }
    
    @staticmethod
    def detect_all(parsed: Dict[str, Any], 
                   raw_output: str,
                   ground_truth: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Comprehensive hallucination detection across all tiers
        
        Returns:
            Dict mapping hallucination_type to list of detected issues
        """
        hallucinations = {k: [] for k in HallucinationDetector.HALLUCINATION_TYPES.keys()}
        
        # Structural hallucinations
        hallucinations["structural"].extend(
            HallucinationDetector._detect_structural(parsed, raw_output)
        )
        
        # Numerical hallucinations
        hallucinations["numerical"].extend(
            HallucinationDetector._detect_numerical(parsed, ground_truth)
        )
        
        # Logical hallucinations
        hallucinations["logical"].extend(
            HallucinationDetector._detect_logical(parsed, raw_output, ground_truth)
        )
        
        # Reasoning hallucinations
        hallucinations["reasoning"].extend(
            HallucinationDetector._detect_reasoning(parsed, raw_output, ground_truth)
        )
        
        return hallucinations
    
    @staticmethod
    def _detect_structural(parsed: Dict[str, Any], raw_output: str) -> List[str]:
        """Detect structural formatting issues"""
        issues = []
        
        required_fields = ["test_type", "test_statistic", "p_value", "decision"]
        for field in required_fields:
            if parsed.get(field) is None:
                issues.append(f"missing_required_field:{field}")
        
        if not parsed and len(raw_output) > 0:
            issues.append("unparseable_output")
        
        return issues
    
    @staticmethod
    def _detect_numerical(parsed: Dict[str, Any], ground_truth: Dict[str, Any]) -> List[str]:
        """Detect numerical impossibilities"""
        issues = []
        
        p_val = parsed.get("p_value")
        if p_val is not None:
            if not (0 <= p_val <= 1):
                issues.append(f"p_value_out_of_range:{p_val}")
        
        t_stat = parsed.get("test_statistic")
        test_type = parsed.get("test_type")
        
        # Chi-square and F-statistics must be non-negative
        if test_type in ["chi_square", "anova"] and t_stat is not None:
            if t_stat < 0:
                issues.append(f"negative_test_statistic_for_positive_test:{t_stat}")
        
        # Implausibly large test statistics (>100 suggests numerical error)
        if t_stat is not None and abs(t_stat) > 100:
            issues.append(f"test_statistic_magnitude_implausible:{t_stat}")
        
        # Contradictory p-value and decision
        if p_val is not None and parsed.get("decision"):
            alpha = ground_truth.get("significance_level", 0.05)
            expected_decision = "reject" if p_val < alpha else "fail_to_reject"
            if parsed["decision"].lower().replace(" ", "_") != expected_decision:
                issues.append(f"contradictory_p_value_and_decision:p={p_val},decision={parsed['decision']}")
        
        return issues
    
    @staticmethod
    def _detect_logical(parsed: Dict[str, Any], raw_output: str, ground_truth: Dict[str, Any]) -> List[str]:
        """Detect logical inconsistencies"""
        issues = []
        
        # Check if model cited wrong test type
        cited_tests = re.findall(r'(one[- ]sample|two[- ]sample|paired|independent|dependent)\s+t[- ]test', 
                                 raw_output.lower())
        if cited_tests and parsed.get("test_type"):
            correct_type = ground_truth.get("test_type", "").replace("_", " ")
            if correct_type not in cited_tests[0]:
                issues.append(f"wrong_statistical_test_cited:{cited_tests[0]}")
        
        # Check for wrong tail specification
        if "one-tailed" in raw_output.lower() and ground_truth.get("alternative") == "two-sided":
            issues.append("wrong_tail_test_for_hypothesis:claimed_one_tailed_for_two_sided_test")
        
        # Check degrees of freedom if mentioned
        df_matches = re.findall(r'(?:df|degrees of freedom)\s*=\s*(\d+)', raw_output.lower())
        if df_matches and ground_truth.get("degrees_of_freedom"):
            claimed_df = int(df_matches[0])
            true_df = ground_truth["degrees_of_freedom"]
            if claimed_df != true_df:
                issues.append(f"incorrect_degrees_of_freedom:claimed={claimed_df},actual={true_df}")
        
        return issues
    
    @staticmethod
    def _detect_reasoning(parsed: Dict[str, Any], raw_output: str, ground_truth: Dict[str, Any]) -> List[str]:
        """Detect reasoning errors in explanations"""
        issues = []
        
        # Check for contradictory statements about p-value
        p_val = parsed.get("p_value")
        if p_val is not None:
            # Find comparison statements
            greater_than_alpha = re.search(r'p[- ]?value.*?>\s*0\.05', raw_output.lower())
            less_than_alpha = re.search(r'p[- ]?value.*?<\s*0\.05', raw_output.lower())
            
            if greater_than_alpha and less_than_alpha:
                issues.append("contradictory_explanation:both_greater_and_less_than_alpha")
            elif greater_than_alpha and p_val < 0.05:
                issues.append(f"contradictory_explanation:claimed_p>0.05_but_p={p_val}")
            elif less_than_alpha and p_val >= 0.05:
                issues.append(f"contradictory_explanation:claimed_p<0.05_but_p={p_val}")
        
        # Check for fabricated formulas (common hallucination)
        suspicious_formulas = [
            r't\s*=\s*\(\s*x̄\s*-\s*μ\s*\)\s*/\s*\(\s*σ\s*/\s*√n\s*\)',  # z-test formula claimed as t-test
            r'chi[- ]?square\s*=\s*\(n[- ]1\)\s*s²\s*/\s*σ²'  # wrong chi-square formula
        ]
        
        for pattern in suspicious_formulas:
            if re.search(pattern, raw_output.lower()):
                issues.append(f"fabricated_formula:{pattern}")
        
        # Check for null hypothesis misinterpretation
        null_hypothesis = ground_truth.get("null_hypothesis", "")
        if null_hypothesis and "null hypothesis" in raw_output.lower():
            # Extract what the model claims the null hypothesis is
            null_claim = re.search(r'null hypothesis.*?:(.*?)(?:\.|$)', raw_output.lower())
            if null_claim and null_hypothesis.lower() not in null_claim.group(1):
                issues.append(f"misinterpreted_null_hypothesis")
        
        return issues
    
    @staticmethod
    def compute_hallucination_rate(hallucinations: Dict[str, List[str]]) -> Dict[str, float]:
        """Compute hallucination rates by category"""
        total_checks = sum(len(HallucinationDetector.HALLUCINATION_TYPES[k]) 
                          for k in hallucinations.keys())
        total_detected = sum(len(v) for v in hallucinations.values())
        
        return {
            "overall_rate": total_detected / total_checks if total_checks > 0 else 0,
            **{f"{k}_rate": len(v) / len(HallucinationDetector.HALLUCINATION_TYPES[k])
               for k, v in hallucinations.items()}
        }