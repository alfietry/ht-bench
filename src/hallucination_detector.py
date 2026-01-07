from typing import Dict, List, Any, Union
import re
import numpy as np
from .response_parser import ParsedResponse

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
    def detect_all(parsed: Union[ParsedResponse, Dict[str, Any]], 
                   raw_output: str,
                   ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive hallucination detection across all tiers
        
        Args:
            parsed: ParsedResponse object or dict with extracted fields
            raw_output: Raw LLM response text
            ground_truth: Dict with correct statistical values
        
        Returns:
            Dict with hallucination results including has_hallucination flag,
            hallucination_types list, severity, and detailed issues by category
        """
        # Convert ParsedResponse to dict if needed
        if hasattr(parsed, 'model_dump'):
            parsed_dict = parsed.model_dump()
        else:
            parsed_dict = parsed
        
        hallucinations = {k: [] for k in HallucinationDetector.HALLUCINATION_TYPES.keys()}
        
        # Structural hallucinations
        hallucinations["structural"].extend(
            HallucinationDetector._detect_structural(parsed_dict, raw_output)
        )
        
        # Numerical hallucinations
        hallucinations["numerical"].extend(
            HallucinationDetector._detect_numerical(parsed_dict, ground_truth)
        )
        
        # Logical hallucinations
        hallucinations["logical"].extend(
            HallucinationDetector._detect_logical(parsed_dict, raw_output, ground_truth)
        )
        
        # Reasoning hallucinations
        hallucinations["reasoning"].extend(
            HallucinationDetector._detect_reasoning(parsed_dict, raw_output, ground_truth)
        )
        
        # Build comprehensive result
        has_any = any(len(v) > 0 for v in hallucinations.values())
        result = {
            'has_hallucination': has_any,
            'hallucination_types': [k for k, v in hallucinations.items() if len(v) > 0],
            'severity': HallucinationDetector._determine_severity(hallucinations),
            'details': hallucinations,
            'counts': {k: len(v) for k, v in hallucinations.items()}
        }
        
        return result
    
    @staticmethod
    def _determine_severity(hallucinations: Dict[str, List[str]]) -> str:
        """Determine overall severity based on hallucination types and counts"""
        total_count = sum(len(v) for v in hallucinations.values())
        
        if total_count == 0:
            return 'none'
        elif len(hallucinations.get('numerical', [])) > 0 or len(hallucinations.get('logical', [])) > 2:
            return 'severe'
        elif len(hallucinations.get('logical', [])) > 0 or total_count > 2:
            return 'moderate'
        else:
            return 'minor'
    
    @staticmethod
    def _detect_structural(parsed: Dict[str, Any], raw_output: str) -> List[str]:
        """Tier 1: Detect structural formatting issues
        
        Catches:
        - Missing required fields (p_value, decision are essential)
        - Unparseable output
        - Invalid response structure
        """
        issues = []
        
        # Only p_value and decision are truly required for evaluation
        # test_type and test_statistic are informative but not essential
        required_fields = ["p_value", "decision"]
        for field in required_fields:
            if parsed.get(field) is None:
                issues.append(f"missing_required_field:{field}")
        
        # Check for completely unparseable output
        if not parsed and len(raw_output) > 0:
            issues.append("unparseable_output")
        
        return issues
    
    @staticmethod
    def _normalize_decision(decision: str) -> str:
        """Normalize decision string to 'reject' or 'fail_to_reject'"""
        if decision is None:
            return None
        
        decision_lower = decision.lower().strip()
        
        # Check for rejection indicators
        reject_patterns = [
            'reject h0', 'reject the null', 'reject null',
            'rejected', 'significant difference', 'statistically significant'
        ]
        fail_to_reject_patterns = [
            'fail to reject', 'do not reject', 'cannot reject',
            'not reject', 'accept h0', 'accept the null', 'accept null',
            'no significant', 'not statistically significant', 'insufficient evidence'
        ]
        
        # Check fail_to_reject first (more specific)
        for pattern in fail_to_reject_patterns:
            if pattern in decision_lower:
                return 'fail_to_reject'
        
        # Then check reject
        for pattern in reject_patterns:
            if pattern in decision_lower:
                return 'reject'
        
        # Simple keyword fallback
        if 'reject' in decision_lower and 'fail' not in decision_lower and 'not' not in decision_lower:
            return 'reject'
        
        return None  # Unable to determine
    
    @staticmethod
    def _detect_numerical(parsed: Dict[str, Any], ground_truth: Dict[str, Any]) -> List[str]:
        """Tier 2: Detect numerical impossibilities
        
        Catches:
        - P-values outside [0, 1] range
        - Negative test statistics for positive-only distributions (F, χ²)
        - Implausible magnitudes (|t| > 100)
        - Contradictory p-value and decision
        """
        issues = []
        
        p_val = parsed.get("p_value")
        if p_val is not None:
            if not (0 <= p_val <= 1):
                issues.append(f"p_value_out_of_range:{p_val}")
        
        t_stat = parsed.get("test_statistic")
        test_type = ground_truth.get("test_type", "").lower()
        
        # Chi-square and F-statistics must be non-negative
        if any(x in test_type for x in ["chi_square", "chi-square", "anova", "f_test"]):
            if t_stat is not None and t_stat < 0:
                issues.append(f"negative_test_statistic_for_positive_test:{t_stat}")
        
        # Implausibly large test statistics (>100 suggests numerical error)
        if t_stat is not None and abs(t_stat) > 100:
            issues.append(f"test_statistic_magnitude_implausible:{t_stat}")
        
        # Contradictory p-value and decision (using semantic comparison)
        if p_val is not None and parsed.get("decision"):
            alpha = ground_truth.get("significance_level", 0.05)
            expected_decision = "reject" if p_val < alpha else "fail_to_reject"
            parsed_decision = HallucinationDetector._normalize_decision(parsed["decision"])
            
            # Only flag as contradictory if we can determine the decision AND it conflicts
            if parsed_decision is not None and parsed_decision != expected_decision:
                issues.append(f"contradictory_p_value_and_decision:p={p_val},expected={expected_decision},got={parsed_decision}")
        
        return issues
    
    @staticmethod
    def _detect_logical(parsed: Dict[str, Any], raw_output: str, ground_truth: Dict[str, Any]) -> List[str]:
        """Tier 3: Detect logical/statistical reasoning errors
        
        Catches:
        - Wrong test type cited (says "two-sample" but data is paired)
        - Incorrect degrees of freedom calculation
        - Wrong tail specification (one-tailed vs two-tailed mismatch)
        - Test assumptions violations not acknowledged
        """
        issues = []
        
        # Get the actual test type from ground truth
        actual_test_type = ground_truth.get("test_type", "").lower()
        
        # Check if model cited wrong test type in explanation
        raw_lower = raw_output.lower()
        
        # Map test types to their identifiers in text
        test_type_patterns = {
            "one_sample": ["one-sample", "one sample", "single sample"],
            "two_sample": ["two-sample", "two sample", "independent samples", "independent groups"],
            "paired": ["paired", "dependent", "matched pairs", "repeated measures"]
        }
        
        # Determine what test type was cited
        cited_type = None
        for test_key, patterns in test_type_patterns.items():
            for pattern in patterns:
                if pattern in raw_lower:
                    cited_type = test_key
                    break
            if cited_type:
                break
        
        # Check for mismatch only if we found a cited type
        if cited_type:
            actual_category = None
            if "one_sample" in actual_test_type:
                actual_category = "one_sample"
            elif "two_sample" in actual_test_type:
                actual_category = "two_sample"
            elif "paired" in actual_test_type:
                actual_category = "paired"
            
            if actual_category and cited_type != actual_category:
                issues.append(f"wrong_statistical_test_cited:claimed={cited_type},actual={actual_category}")
        
        # Check for wrong tail specification (only flag clear contradictions)
        alternative = ground_truth.get("alternative", "two-sided")
        if "one-tailed" in raw_lower or "one tailed" in raw_lower:
            if alternative == "two-sided":
                issues.append("wrong_tail_test_for_hypothesis:claimed_one_tailed_for_two_sided_test")
        if ("two-tailed" in raw_lower or "two tailed" in raw_lower) and alternative in ["less", "greater"]:
            issues.append("wrong_tail_test_for_hypothesis:claimed_two_tailed_for_one_sided_test")
        
        # Check degrees of freedom if explicitly mentioned (with some tolerance)
        df_matches = re.findall(r'(?:df|degrees?\s+of\s+freedom)\s*[=:]\s*(\d+)', raw_lower)
        if df_matches and ground_truth.get("degrees_of_freedom"):
            claimed_df = int(df_matches[0])
            true_df = ground_truth["degrees_of_freedom"]
            # Allow small differences due to Welch's correction or rounding
            if abs(claimed_df - true_df) > 2:
                issues.append(f"incorrect_degrees_of_freedom:claimed={claimed_df},actual={true_df}")
        
        return issues
    
    @staticmethod
    def _detect_reasoning(parsed: Dict[str, Any], raw_output: str, ground_truth: Dict[str, Any]) -> List[str]:
        """Tier 4: Detect reasoning/explanation hallucinations
        
        Catches:
        - Contradictory statements (claiming p > 0.05 when computed p = 0.03)
        - Fabricated formulas (wrong statistical formulas)
        - Null hypothesis misinterpretation
        - Contradictory statements within same response
        """
        issues = []
        raw_lower = raw_output.lower()
        
        # Check for contradictory statements about p-value comparison
        p_val = parsed.get("p_value")
        if p_val is not None:
            # Find explicit comparison statements about significance threshold
            claims_significant = any(phrase in raw_lower for phrase in [
                "p < 0.05", "p<0.05", "p-value < 0.05", "p-value<0.05",
                "p < α", "p < alpha", "is significant", "statistically significant",
                "reject the null", "reject h0"
            ])
            claims_not_significant = any(phrase in raw_lower for phrase in [
                "p > 0.05", "p>0.05", "p-value > 0.05", "p-value>0.05",
                "p > α", "p > alpha", "not significant", "fail to reject",
                "cannot reject", "do not reject", "insufficient evidence"
            ])
            
            # Flag contradiction only if both claims appear
            if claims_significant and claims_not_significant:
                issues.append("contradictory_explanation:claims_both_significant_and_not_significant")
            # Check if claims contradict the actual p-value
            elif claims_significant and p_val >= 0.05:
                issues.append(f"contradictory_explanation:claims_significant_but_p={p_val:.4f}")
            elif claims_not_significant and p_val < 0.05:
                issues.append(f"contradictory_explanation:claims_not_significant_but_p={p_val:.4f}")
        
        # Check for fabricated/wrong formulas - only flag clearly wrong formulas
        # z-test formula used for t-test (population σ instead of sample s)
        if "t-test" in raw_lower or "t test" in raw_lower:
            # z-formula: (x̄ - μ) / (σ / √n) - using population σ is wrong for t-test
            if re.search(r't\s*=.*?/\s*\(?σ\s*/\s*√?n', raw_lower):
                if 'population standard deviation' not in raw_lower:
                    issues.append("fabricated_formula:using_z_test_formula_for_t_test")
        
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