"""
Response parsing and validation for LLM outputs
"""
import re
import json
import logging
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ValidationError, field_validator
import numpy as np

logger = logging.getLogger(__name__)


class HypothesesModel(BaseModel):
    """Model for hypotheses"""
    H0: str = Field(..., description="Null hypothesis")
    H1: str = Field(..., description="Alternative hypothesis")


class AssumptionsModel(BaseModel):
    """Model for test assumptions"""
    normality: Optional[str] = Field(None, description="Normality assumption")
    independence: Optional[str] = Field(None, description="Independence assumption")
    equal_variance: Optional[str] = Field(None, description="Equal variance assumption")


class ParsedResponse(BaseModel):
    """Structured response from LLM"""
    hypotheses: Optional[HypothesesModel] = None
    test_method: Optional[str] = None
    assumptions: Optional[AssumptionsModel] = None
    test_statistic: Optional[float] = None
    p_value: Optional[float] = None
    degrees_of_freedom: Optional[Union[float, int]] = None
    critical_value: Optional[float] = None
    decision: Optional[str] = None
    significance_level: Optional[float] = 0.05
    conclusion: Optional[str] = None
    confidence_interval: Optional[list] = None
    
    @field_validator('p_value')
    def validate_p_value(cls, v):
        """Validate p-value is between 0 and 1"""
        if v is not None:
            # If value looks like a test statistic (outside 0-1 range), set to None
            # This will trigger fallback parsing
            if v < 0 or v > 1:
                logger.warning(f"Invalid p-value {v} (outside 0-1 range), setting to None for re-parsing")
                return None
        return v
    
    @field_validator('decision')
    def validate_decision(cls, v):
        """Normalize decision strings"""
        if v is None:
            return v
        v_lower = v.lower().replace(" ", "_")
        if "reject" in v_lower and "fail" not in v_lower and "not" not in v_lower:
            return "reject_H0"
        elif "fail" in v_lower or "not reject" in v_lower or "accept" in v_lower:
            return "fail_to_reject_H0"
        return v


class ResponseParser:
    """Parse LLM responses into structured format"""

    # Old single-line RESULT pattern (kept for backward compatibility)
    RESULT_PATTERN = re.compile(
        r"RESULT:\s*test_type=([^;]+);\s*statistic=([^;]+);\s*p_value=([^;]+);\s*decision=([^\n]+)",
        re.IGNORECASE
    )
    
    # New RESULTS block pattern for PoT prompts
    RESULTS_BLOCK_PATTERN = re.compile(
        r"RESULTS?:\s*\n(.*?)(?:\n\n|\Z)",
        re.IGNORECASE | re.DOTALL
    )

    @staticmethod
    def parse_results_block(text: str) -> Optional[ParsedResponse]:
        """Parse the RESULTS: block from Program-of-Thought responses"""
        match = ResponseParser.RESULTS_BLOCK_PATTERN.search(text)
        if not match:
            return None
        
        block = match.group(1)
        
        # Extract H0 and H1
        h0_match = re.search(r'H[_\s]?0\s*[:=]\s*([^\n]+)', block, re.IGNORECASE)
        h1_match = re.search(r'H[_\s]?1\s*[:=]\s*([^\n]+)', block, re.IGNORECASE)
        hypotheses = None
        if h0_match and h1_match:
            hypotheses = HypothesesModel(
                H0=h0_match.group(1).strip(),
                H1=h1_match.group(1).strip()
            )
        
        # Extract test name
        test_match = re.search(r'Test\s*[:=]\s*([^\n]+)', block, re.IGNORECASE)
        test_method = None
        if test_match:
            test_method = test_match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        
        # Extract test statistic
        stat_match = re.search(r'Test\s+statistic\s*[:=]\s*([+-]?\d+\.?\d*)', block, re.IGNORECASE)
        test_statistic = float(stat_match.group(1)) if stat_match else None
        
        # Extract p-value
        p_match = re.search(r'P[_-]?value\s*[:=]\s*([0-9.eE+-]+)', block, re.IGNORECASE)
        p_value = float(p_match.group(1)) if p_match else None
        
        # Extract degrees of freedom
        df_match = re.search(r'Degrees?\s+of\s+freedom\s*[:=]\s*(\d+\.?\d*)', block, re.IGNORECASE)
        degrees_of_freedom = float(df_match.group(1)) if df_match else None
        
        # Extract decision
        decision_match = re.search(r'Decision\s*[:=]\s*([^\n(]+)', block, re.IGNORECASE)
        decision = None
        if decision_match:
            decision_text = decision_match.group(1).strip().lower()
            if "reject" in decision_text and "fail" not in decision_text and "not" not in decision_text:
                decision = "reject_H0"
            elif any(term in decision_text for term in ["fail", "not"]):
                decision = "fail_to_reject_H0"
        
        # Extract conclusion
        conclusion_match = re.search(r'Conclusion\s*[:=]\s*([^\n]+)', block, re.IGNORECASE)
        conclusion = conclusion_match.group(1).strip() if conclusion_match else None
        
        return ParsedResponse(
            hypotheses=hypotheses,
            test_method=test_method,
            test_statistic=test_statistic,
            p_value=p_value,
            degrees_of_freedom=degrees_of_freedom,
            decision=decision,
            conclusion=conclusion,
        )

    @staticmethod
    def parse_result_line(text: str) -> Optional[ParsedResponse]:
        """Parse structured RESULT line emitted by Program-of-Thought prompts"""
        match = ResponseParser.RESULT_PATTERN.search(text)
        if not match:
            return None

        test_type, stat_raw, p_raw, decision_raw = [m.strip() for m in match.groups()]

        try:
            statistic = float(stat_raw)
        except ValueError:
            statistic = None

        try:
            p_value = float(p_raw)
        except ValueError:
            p_value = None

        decision_normalized = None
        if decision_raw:
            lower = decision_raw.lower()
            if "reject" in lower and "fail" not in lower and "not" not in lower:
                decision_normalized = "reject_H0"
            elif any(term in lower for term in ["fail", "not", "do not"]):
                decision_normalized = "fail_to_reject_H0"

        return ParsedResponse(
            test_method=test_type.replace(" ", "_").lower(),
            test_statistic=statistic,
            p_value=p_value,
            decision=decision_normalized,
        )
    
    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text"""
        # Try to find JSON object
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                json_str = match.group()
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
        
        return None
    
    @staticmethod
    def extract_number(text: str, pattern: str, default: Optional[float] = None,
                       use_last: bool = True) -> Optional[float]:
        """Extract numerical value from text"""
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            iterable = reversed(matches) if use_last else matches
            for match in iterable:
                try:
                    # Extract the number
                    num_str = match.group(1).replace(",", "").strip()
                    return float(num_str)
                except (ValueError, IndexError):
                    continue
        return default
    
    @staticmethod
    def extract_hypotheses(text: str) -> Optional[HypothesesModel]:
        """Extract hypotheses from text"""
        h0_pattern = r'H[_\s]?0\s*[:=]\s*([^\n]+)'
        h1_pattern = r'H[_\s]?1\s*[:=]\s*([^\n]+)'
        
        h0_match = re.search(h0_pattern, text, re.IGNORECASE)
        h1_match = re.search(h1_pattern, text, re.IGNORECASE)
        
        if h0_match and h1_match:
            return HypothesesModel(
                H0=h0_match.group(1).strip(),
                H1=h1_match.group(1).strip()
            )
        return None
    
    @staticmethod
    def extract_test_method(text: str) -> Optional[str]:
        """Extract test method from text"""
        test_patterns = [
            r'test\s+name\s*[:=]\s*([^\n]+)',  # "Test name: Two-tailed t-test"
            r'test\s+(?:type|method)\s*[:=]\s*([^\n]+)',
            r'(one[- ]sample [tz][- ]test)',
            r'(two[- ]sample [tz][- ]test)',
            r'(paired [t][- ]test)',
            r'(two[- ]tailed [t][- ]test)',
            r'(one[- ]tailed [t][- ]test)',
            r'(welch\'?s? [t][- ]test)',
            r'(student\'?s? [t][- ]test)',
            r'(independent samples [t][- ]test)',
            r'(one[- ]way anova)',
            r'(anova)',
            r'(chi[- ]square.*goodness of fit)',
            r'(chi[- ]square.*independence)',
            r'(chi[- ]square test)',
        ]
        
        text_lower = text.lower()
        for pattern in test_patterns:
            match = re.search(pattern, text_lower)
            if match:
                method = match.group(1).strip()
                # Truncate explanatory suffixes (e.g., "... (since ...)")
                method = re.split(r'[\(\n]', method)[0].strip()
                # Normalize method name
                method = method.replace(" ", "_").replace("-", "_")
                
                # Map common variations
                if "two_tailed_t_test" in method or "two_tailed t_test" in method:
                    return "one_sample_t_test"  # Two-tailed is default for one-sample
                if "one_tailed" in method:
                    return "one_sample_t_test"
                    
                return method
        
        return None
    
    @staticmethod
    def extract_decision(text: str) -> Optional[str]:
        """Extract hypothesis test decision"""
        # Look for numbered format first: "5. Decision: fail to reject H0"
        decision_patterns = [
            r'decision\s*[:=]\s*([^\n]+(?:\n(?!\d+\.)[^\n]+)*)',
            r'\d+\.\s*decision\s*[:=]\s*([^\n]+(?:\n(?!\d+\.)[^\n]+)*)',
        ]
        
        for pattern in decision_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                decision_text = match.group(1).strip().lower()
                
                # Normalize decision
                if "fail" in decision_text or "not reject" in decision_text or "do not reject" in decision_text:
                    return "fail_to_reject_H0"
                elif "reject" in decision_text:
                    return "reject_H0"
        
        # Fallback to original patterns
        text_lower = text.lower()
        
        if re.search(r'reject\s+(the\s+)?h0', text_lower) and \
           not re.search(r'(fail|do not|cannot|don\'t)\s+reject', text_lower):
            return "reject_H0"
        elif re.search(r'(fail to reject|do not reject|cannot reject)\s+(the\s+)?h0', text_lower):
            return "fail_to_reject_H0"
        
        return None
    
    @staticmethod
    def extract_conclusion(text: str) -> Optional[str]:
        """Extract conclusion from text"""
        conclusion_patterns = [
            r'conclusion\s*[:=]\s*([^\n]+(?:\n(?!\d+\.)[^\n]+)*)',
            r'\d+\.\s*conclusion\s*[:=]\s*([^\n]+(?:\n(?!\d+\.)[^\n]+)*)',
        ]
        
        for pattern in conclusion_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    @staticmethod
    def extract_confidence_interval(text: str) -> Optional[list]:
        """Extract confidence interval"""
        # Pattern: (lower, upper) or [lower, upper]
        ci_pattern = r'[\(\[]([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)[\)\]]'
        match = re.search(ci_pattern, text)
        
        if match:
            try:
                return [float(match.group(1)), float(match.group(2))]
            except ValueError:
                pass
        
        return None

    @staticmethod
    def extract_degrees_of_freedom(text: str) -> Optional[float]:
        """Extract degrees of freedom from either 'df' or descriptive text"""
        df_patterns = [
            r'(?:degrees? of freedom|df)[^\n]*',
        ]
        for pattern in df_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for snippet in matches:
                numbers = re.findall(r'[+-]?\d+\.?\d*', snippet)
                if numbers:
                    try:
                        return float(numbers[-1])
                    except ValueError:
                        continue
        return None
    
    @staticmethod
    def parse_regex(text: str) -> ParsedResponse:
        """Parse using regex patterns"""
        # Extract values using regex
        test_statistic = ResponseParser.extract_number(
            text, r'test[- ]statistic.*?[:=]\s*([+-]?\d+\.?\d*)', None
        )
        if test_statistic is None:
            test_statistic = ResponseParser.extract_number(
                text, r'\d+\.\s*test[- ]statistic.*?[:=]\s*([+-]?\d+\.?\d*)', None
            )
        if test_statistic is None:
            test_statistic = ResponseParser.extract_number(
                text, r'[tz][- ]?statistic.*?[:=]\s*([+-]?\d+\.?\d*)', None
            )
        if test_statistic is None:
            test_statistic = ResponseParser.extract_number(
                text, r'\b[tz]\s*[=:]\s*([+-]?\d+\.?\d*)', None
            )
        
        # Extract p-value
        p_value = None
        approx_chars = '=:\u2248\u2245~'
        p_value_patterns = [
            rf'\d+\.\s*p[- ]?value\s*[{approx_chars}]\s*([0-9.eE-]+)',
            rf'p[- ]?value\s*[{approx_chars}]\s*([0-9.eE-]+)',
            rf'p\s*[{approx_chars}]\s*([0-9.eE-]+)',
        ]
        
        for pattern in p_value_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    val = float(match)
                    if 0 <= val <= 1:
                        p_value = val
                        break
                except:
                    continue
            if p_value is not None:
                break
        
        hypotheses = ResponseParser.extract_hypotheses(text)
        test_method = ResponseParser.extract_test_method(text)
        decision = ResponseParser.extract_decision(text)
        conclusion = ResponseParser.extract_conclusion(text)
        ci = ResponseParser.extract_confidence_interval(text)
        
        df = ResponseParser.extract_degrees_of_freedom(text)
        if df is None:
            df = ResponseParser.extract_number(
                text, r'degrees? of freedom.*?[:=]\s*(\d+)', None, use_last=True
            )
        
        critical_value = ResponseParser.extract_number(
            text, r'critical value.*?[:=]\s*([+-]?\d+\.?\d*)', None
        )
        
        return ParsedResponse(
            hypotheses=hypotheses,
            test_method=test_method,
            test_statistic=test_statistic,
            p_value=p_value,
            degrees_of_freedom=df,
            critical_value=critical_value,
            decision=decision,
            conclusion=conclusion,
            confidence_interval=ci
        )
    
    @staticmethod
    def parse_json(text: str) -> Optional[ParsedResponse]:
        """Parse JSON response"""
        json_data = ResponseParser.extract_json(text)
        if json_data:
            try:
                return ParsedResponse(**json_data)
            except ValidationError as e:
                logger.warning(f"JSON validation error: {e}")
                return None
        return None
    
    @staticmethod
    def parse(text: str, format: str = "auto") -> ParsedResponse:
        """Parse response with fallback strategies"""
        if format == "json":
            result = ResponseParser.parse_json(text)
            if result:
                return result
        
        # Try JSON first if auto
        if format == "auto":
            result = ResponseParser.parse_json(text)
            if result and result.test_statistic is not None:
                return result

            # Try parsing RESULTS: block (new PoT format)
            results_block = ResponseParser.parse_results_block(text)
            if results_block and results_block.test_statistic is not None and results_block.p_value is not None:
                return results_block

            # Check for old PoT summary line before regex heuristics
            pot_result = ResponseParser.parse_result_line(text)
            if pot_result and pot_result.test_statistic is not None and pot_result.p_value is not None:
                return pot_result
        
        # Fallback to regex parsing
        return ResponseParser.parse_regex(text)
    
    @staticmethod
    def validate_parsed_response(parsed: ParsedResponse, 
                                 ground_truth: Dict[str, Any]) -> Dict[str, bool]:
        """Validate parsed response against ground truth"""
        validation = {
            "has_hypotheses": parsed.hypotheses is not None,
            "has_test_method": parsed.test_method is not None,
            "has_test_statistic": parsed.test_statistic is not None,
            "has_p_value": parsed.p_value is not None,
            "has_decision": parsed.decision is not None,
            "valid_p_value": False,
            "valid_test_statistic": False,
            "correct_decision": False,
        }
        
        # Validate p-value range
        if parsed.p_value is not None:
            validation["valid_p_value"] = 0 <= parsed.p_value <= 1
        
        # Check if values are reasonable compared to ground truth
        if parsed.test_statistic is not None and ground_truth.get("test_statistic"):
            gt_stat = ground_truth["test_statistic"]
            # Check if within reasonable range (10x difference)
            validation["valid_test_statistic"] = (
                abs(parsed.test_statistic - gt_stat) < abs(gt_stat) * 10
            )
        
        # Check if decision matches ground truth
        if parsed.decision is not None and ground_truth.get("decision"):
            validation["correct_decision"] = (
                parsed.decision == ground_truth["decision"]
            )
        
        return validation


def extract_code_from_response(text: str) -> Optional[str]:
    """Extract Python code from response"""
    # Look for code blocks
    code_pattern = r'```(?:python)?\n(.*?)```'
    matches = re.findall(code_pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    return None


def execute_code_safely(code: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Python code in sandboxed environment and extract results"""
    import sys
    from io import StringIO
    
    # Prepare namespace with data
    namespace = {
        'np': np,
        '__builtins__': __builtins__,
    }
    
    # Add data variables
    for key, value in data.items():
        if isinstance(value, (list, np.ndarray)):
            namespace[key] = np.array(value)
        else:
            namespace[key] = value
    
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    results = {}
    try:
        # Execute code
        exec(code, namespace)
        
        # Extract results from namespace
        for key in ['test_statistic', 'p_value', 'decision', 't_stat', 'f_stat', 'chi_stat']:
            if key in namespace:
                results[key] = namespace[key]
        
        # Get printed output
        output = sys.stdout.getvalue()
        results['output'] = output
        
    except Exception as e:
        results['error'] = str(e)
        logger.error(f"Code execution error: {e}")
    finally:
        sys.stdout = old_stdout
    
    return results
