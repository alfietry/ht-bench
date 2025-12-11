"""
Prompt templates for hypothesis testing evaluation
"""
from typing import Dict, Any, List
import numpy as np


class PromptTemplate:
    """Base class for prompt templates"""
    
    @staticmethod
    def format_data(data: Dict[str, Any]) -> str:
        """Format data for prompt"""
        formatted = []
        
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

Provide response in the format below but on the same line:
1. H0 and H1 (use plain text like mean1, mean2, not Greek symbols, you can also use =, !=, <, >, <=, >=, etc.)
2. Test name: (one-sample t-test)
3. Test statistic value: (5.14)
4. P-value: (0.003)
5. Decision: (reject/fail to reject H0)
6. Brief conclusion: (one sentence)
7. Degrees of freedom: (number or N/A)
8. Critical value(s): (number(s) or N/A)
9. Assumptions checked: (normality, independence, equal variance - yes/no)

IMPORTANT: Use only plain ASCII text. No markdown formatting (no **, __, ##). Be brief.

Notation requirements:
- mu for population mean
- xb for sample mean
- sig for population standard deviation
- s for sample standard deviation
- alp for significance level
- Follow this pattern for all Greek letters (use ASCII abbreviations)"""
        
        return prompt


class FewShotPrompt(PromptTemplate):
    """Few-shot prompt with examples"""
    
    EXAMPLES = [
        {
            "data": "Sample 1: [12.3, 14.1, 13.5, 15.2, 14.8]\nSample 2: [10.1, 11.3, 10.8, 11.9, 10.5]",
            "analysis": """
H0: mu1 = mu2
H1: mu1 != mu2
Test: Two-sample t-test
t-statistic: 5.14
p-value: 0.0008
Decision: Reject H0 (alp = 0.05)
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

Be brief and direct. Show key values only. Use only plain ASCII text. No markdown formatting.

Notation: mu=population mean, xb=sample mean, sig=population std dev, s=sample std dev, alp=significance level."""
        
        return prompt


class ProgramOfThoughtPrompt(PromptTemplate):
    """Program-of-Thought prompt for code-based reasoning"""
    
    @staticmethod
    def create(data: Dict[str, Any], test_context: str = "") -> str:
        """Create PoT prompt"""
        # Convert data to Python code
        code_lines = ["import numpy as np", "from scipy import stats", ""]
        
        if "sample1" in data:
            sample1_list = list(data["sample1"])
            code_lines.append(f"sample1 = np.array({sample1_list})")
        
        if "sample2" in data:
            sample2_list = list(data["sample2"])
            code_lines.append(f"sample2 = np.array({sample2_list})")
        
        if "groups" in data:
            for i, group in enumerate(data["groups"]):
                code_lines.append(f"group{i+1} = np.array({list(group)})")
        
        if "observed" in data:
            code_lines.append(f"observed = np.array({data['observed']})")
        
        if "expected" in data:
            code_lines.append(f"expected = np.array({data['expected']})")
        
        if "population_mean" in data:
            code_lines.append(f"population_mean = {data['population_mean']}")
        
        if "population_std" in data:
            code_lines.append(f"population_std = {data['population_std']}")
        
        data_code = "\n".join(code_lines)
        
        prompt = f"""Write concise Python (numpy/scipy) code that performs the correct hypothesis test using scipy.stats ONLY. Keep imports to `import numpy as np` and `from scipy import stats`.

    {test_context}

    Data (already valid Python):
    {data_code}

    Your response must:
    1. Detect and name the appropriate test (e.g., "two_sample_t_test").
    2. Run the computation inside real Python code (no pseudocode) using numpy/scipy.
    3. Print the test name, test statistic, p-value, and the decision at alp = 0.05.
    4. After the code, emit a single plain-text summary line exactly in this format:
       RESULT: test_type=<name>; statistic=<value>; p_value=<value>; decision=<reject/fail_to_reject>

    Example summary line:
    RESULT: test_type=two_sample_t_test; statistic=2.41; p_value=0.021; decision=reject

    Keep everything ASCII. Do not use markdown fences. Ensure the summary line appears once after the code so downstream parsers can read it.

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
