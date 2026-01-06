"""
Prompt templates for hypothesis testing evaluation.

Supports both synthetic and real-world data scenarios with appropriate context formatting.
"""
from typing import Dict, Any, List, Optional
import numpy as np


def create_scenario_description(data: Dict[str, Any]) -> str:
    """
    Create a rich contextual description for real-world data scenarios.
    
    Args:
        data: Scenario dictionary with optional 'context' key for real data
        
    Returns:
        Formatted description string with domain-specific context
    """
    if "context" not in data or data.get("data_source") != "real":
        return ""
    
    ctx = data["context"]
    parts = []
    
    # Domain and dataset info
    domain = ctx.get("domain", "").title()
    dataset = ctx.get("dataset", "")
    if domain or dataset:
        parts.append(f"Domain: {domain}")
        parts.append(f"Dataset: {dataset}")
    
    # Main description
    description = ctx.get("description", "")
    if description:
        parts.append(f"\nScenario: {description}")
    
    # Test-specific description
    test_desc = ctx.get("test_description", "")
    if test_desc:
        parts.append(f"Research Question: {test_desc}")
    
    # Group names for two-sample tests
    group1_name = ctx.get("group1_name")
    group2_name = ctx.get("group2_name")
    if group1_name and group2_name:
        parts.append(f"\nGroup 1: {group1_name}")
        parts.append(f"Group 2: {group2_name}")
    
    # Event info for paired tests (time-series comparisons)
    event = ctx.get("event")
    event_date = ctx.get("event_date")
    if event:
        event_str = f"Event: {event}"
        if event_date:
            event_str += f" ({event_date})"
        parts.append(f"\n{event_str}")
    
    # Intervention info for healthcare paired tests
    intervention = ctx.get("intervention")
    if intervention:
        parts.append(f"\nIntervention: {intervention}")
    
    # Units
    units = ctx.get("units")
    if units:
        parts.append(f"Units: {units}")
    
    # Note if present
    note = ctx.get("note")
    if note:
        parts.append(f"\nNote: {note}")
    
    return "\n".join(parts) if parts else ""


class PromptTemplate:
    """Base class for prompt templates"""
    
    @staticmethod
    def format_data(data: Dict[str, Any]) -> str:
        """Format data for prompt, including real-world context if available."""
        formatted = []
        
        # Add real-world scenario context if present
        scenario_desc = create_scenario_description(data)
        if scenario_desc:
            formatted.append("=== Real-World Context ===")
            formatted.append(scenario_desc)
            formatted.append("\n=== Statistical Data ===")
        
        if "sample1" in data:
            sample1 = data["sample1"]
            formatted.append(f"Sample 1: {list(sample1[:10])}{'...' if len(sample1) > 10 else ''}")
            formatted.append(f"Sample 1 size: {len(sample1)}")
            formatted.append(f"Sample 1 mean: {np.mean(sample1):.4f}")
            formatted.append(f"Sample 1 std: {np.std(sample1, ddof=1):.4f}")
        
        if "sample2" in data:
            sample2 = data["sample2"]
            formatted.append(f"\nSample 2: {list(sample2[:10])}{'...' if len(sample2) > 10 else ''}")
            formatted.append(f"Sample 2 size: {len(sample2)}")
            formatted.append(f"Sample 2 mean: {np.mean(sample2):.4f}")
            formatted.append(f"Sample 2 std: {np.std(sample2, ddof=1):.4f}")
        
        if "groups" in data:
            formatted.append(f"\nNumber of groups: {len(data['groups'])}")
            for i, group in enumerate(data["groups"]):
                formatted.append(f"Group {i+1}: size={len(group)}, mean={np.mean(group):.4f}, std={np.std(group, ddof=1):.4f}")
        
        if "observed" in data and "expected" in data:
            formatted.append(f"\nObserved frequencies: {data['observed']}")
            formatted.append(f"Expected frequencies: {data['expected']}")
        
        if "contingency_table" in data:
            formatted.append(f"\nContingency table:\n{np.array(data['contingency_table'])}")
        
        if "population_mean" in data:
            formatted.append(f"\nHypothesized population mean: {data['population_mean']}")
        
        if "population_std" in data:
            formatted.append(f"Known population standard deviation: {data['population_std']}")
        
        return "\n".join(formatted)


class ZeroShotPrompt(PromptTemplate):
    """Zero-shot prompt template"""
    
    @staticmethod
    def create(data: Dict[str, Any], test_context: str = "") -> str:
        """Create zero-shot prompt"""
        data_str = PromptTemplate.format_data(data)
        
        prompt = f"""Analyze this data and perform the appropriate hypothesis test. Be concise and direct.

{test_context}

Data:
{data_str}

Provide your answer in this exact format:
H0: <null hypothesis, e.g., mu1 = mu2>
H1: <alternative hypothesis, e.g., mu1 != mu2>
Test_type: <test name, e.g., Two-sample t-test>
t-statistic: <exact numerical value, e.g., 5.14>
p-value: <exact numerical value, e.g., 0.0008>
Decision: <Reject H0 or Fail to reject H0>
Conclusion: <one sentence interpretation>

IMPORTANT: 
- Use only plain ASCII text. No markdown formatting (no **, __, ##). Be brief.
- P-values and t-statistics MUST be exact numerical values (e.g., 0.0342, 0.00001), NOT inequalities like "> 0.05" or "< 0.0001"

Notation: mu=population mean, xb=sample mean, sig=population std dev, s=sample std dev, alp=significance level."""
        
        return prompt


class FewShotPrompt(PromptTemplate):
    """Few-shot prompt with examples"""
    
    EXAMPLES = [
        {
            "data": "Sample 1: [12.3, 14.1, 13.5, 15.2, 14.8]\nSample 2: [10.1, 11.3, 10.8, 11.9, 10.5]",
            "analysis": """
H0: mu1 = mu2
H1: mu1 != mu2
Test_type: Two-sample t-test
t-statistic: 5.14
p-value: 0.0008
Decision: Reject H0
Conclusion: The two population means are significantly different.
"""
        }
    ]
    
    @staticmethod
    def create(data: Dict[str, Any], test_context: str = "") -> str:
        """Create few-shot prompt"""
        data_str = PromptTemplate.format_data(data)
        
        examples_str = "\n\n---\n\n".join([
            f"Example:\nData:\n{ex['data']}\n\nAnalysis:{ex['analysis']}"
            for ex in FewShotPrompt.EXAMPLES
        ])
        
        prompt = f"""Perform hypothesis test following this example format. Be brief and direct.

{examples_str}

---

{test_context}

Data:
{data_str}

Provide analysis in the same concise format. Use only plain ASCII text. No markdown formatting.

IMPORTANT: P-values and t-statistics MUST be exact numerical values (e.g., 0.0342, 0.00001), NOT inequalities like "> 0.05" or "< 0.0001"

Notation: mu=population mean, xb=sample mean, sig=population std dev, s=sample std dev, alp=significance level."""
        
        return prompt


class ChainOfThoughtPrompt(PromptTemplate):
    """Chain-of-Thought prompt for step-by-step reasoning"""
    
    @staticmethod
    def create(data: Dict[str, Any], test_context: str = "") -> str:
        """Create CoT prompt"""
        data_str = PromptTemplate.format_data(data)
        
        prompt = f"""Solve this hypothesis test step by step. Be concise.

{test_context}

Data:
{data_str}

Steps:
1. Identify test type
2. State H0 and H1 (use notation: mu for population mean, xb for sample mean)
3. Calculate test statistic
4. Find p-value
5. Make decision (alp = 0.05)
6. State conclusion

Provide your answer in this exact format:
H0: <null hypothesis, e.g., mu1 = mu2>
H1: <alternative hypothesis, e.g., mu1 != mu2>
Test_type: <test name, e.g., Two-sample t-test>
t-statistic: <exact numerical value, e.g., 5.14>
p-value: <exact numerical value, e.g., 0.0008>
Decision: <Reject H0 or Fail to reject H0>
Conclusion: <one sentence interpretation>

Be brief and direct. Show key values only. Use only plain ASCII text. No markdown formatting.

IMPORTANT: P-values and t-statistics MUST be exact numerical values (e.g., 0.0342, 0.00001), NOT inequalities like "> 0.05" or "< 0.0001"

Notation: mu=population mean, xb=sample mean, sig=population std dev, s=sample std dev, alp=significance level."""
        
        return prompt


class ProgramOfThoughtPrompt(PromptTemplate):
    """Program-of-Thought prompt for code-based reasoning"""
    
    @staticmethod
    def create(data: Dict[str, Any], test_context: str = "") -> str:
        """Create PoT prompt"""
        data_str = PromptTemplate.format_data(data)
        
        prompt = f"""Solve this hypothesis test by writing and executing Python code mentally. Show your code, then provide the final results.

{test_context}

Data:
{data_str}

Instructions:
1. Write Python code using numpy/scipy to perform the appropriate hypothesis test
2. Execute the code mentally and compute the actual numerical results
3. After your code, output the RESULTS section with the computed values

Your response format:
```python
import numpy as np
from scipy import stats

# Your computation code here
# ...
```

Provide your answer in this exact format:
H0: <null hypothesis, e.g., mu1 = mu2>
H1: <alternative hypothesis, e.g., mu1 != mu2>
Test_type: <test name, e.g., Two-sample t-test>
t-statistic: <exact numerical value, e.g., 5.14>
p-value: <exact numerical value, e.g., 0.0008>
Decision: <Reject H0 or Fail to reject H0>
Conclusion: <one sentence interpretation>

IMPORTANT:
- State hypotheses using notation: mu for population mean, mu1/mu2 for group means
- Provide actual computed numerical values, not variable names or formulas
- The RESULTS section must contain the final numerical answers
- P-values and t-statistics MUST be exact numerical values (e.g., 0.0342, 0.00001), NOT inequalities like "> 0.05" or "< 0.0001"

Notation: mu=population mean, xb=sample mean, sig=population std dev, s=sample std dev, alp=significance level."""
        
        return prompt


class StructuredOutputPrompt(PromptTemplate):
    """Prompt designed for structured JSON output"""
    
    @staticmethod
    def create(data: Dict[str, Any], test_context: str = "") -> str:
        """Create structured output prompt"""
        data_str = PromptTemplate.format_data(data)
        
        prompt = f"""Analyze data and return JSON only. No extra text.

{test_context}

Data:
{data_str}

Return this exact JSON structure:
{{
    "hypotheses": {{"H0": "...", "H1": "..."}},
    "test_method": "test name",
    "test_statistic": number,
    "p_value": number,
    "degrees_of_freedom": number or null,
    "decision": "reject_H0" or "fail_to_reject_H0",
    "conclusion": "brief conclusion"
}}

Numbers only, no strings. Be concise.

Notation in hypotheses: mu=population mean, xb=sample mean, sig=population std dev, s=sample std dev, alp=significance level."""
        
        return prompt


# Response schema for structured outputs
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "hypotheses": {
            "type": "object",
            "properties": {
                "H0": {"type": "string"},
                "H1": {"type": "string"}
            }
        },
        "test_method": {"type": "string"},
        "assumptions": {
            "type": "object",
            "properties": {
                "normality": {"type": "string"},
                "independence": {"type": "string"},
                "equal_variance": {"type": "string"}
            }
        },
        "test_statistic": {"type": "number"},
        "p_value": {"type": "number"},
        "degrees_of_freedom": {"type": ["number", "null"]},
        "critical_value": {"type": ["number", "null"]},
        "decision": {"type": "string"},
        "significance_level": {"type": "number"},
        "conclusion": {"type": "string"},
        "confidence_interval": {
            "type": ["array", "null"],
            "items": {"type": "number"}
        }
    }
}


def get_prompt(prompt_type: str, data: Dict[str, Any], test_context: str = "") -> str:
    """Get prompt based on type"""
    prompts = {
        "zero_shot": ZeroShotPrompt.create,
        "few_shot": FewShotPrompt.create,
        "chain_of_thought": ChainOfThoughtPrompt.create,
        "program_of_thought": ProgramOfThoughtPrompt.create,
        "structured": StructuredOutputPrompt.create,
    }
    
    if prompt_type not in prompts:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    return prompts[prompt_type](data, test_context)
