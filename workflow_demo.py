#!/usr/bin/env python3
"""
================================================================================
LLM HYPOTHESIS TESTING BENCHMARK - WORKFLOW DEMONSTRATION
================================================================================

This script provides a comprehensive, step-by-step visualization of the 
data and control flow in the LLM-based hypothesis testing research codebase.

Purpose: Professor/stakeholder presentation to establish trustworthiness and
rigor of the experimental workflow, from synthetic data generation to final 
metric computation and dashboard reporting.

Author: LLM Hypothesis Testing Benchmark Team
Date: December 2025

WORKFLOW STAGES:
    1. Synthetic Data Generation
    2. Prompt Construction/Styling
    3. Orchestration and LLM Client Interaction (REAL API CALL)
    4. Response Parsing
    5. Ground Truth Identification
    6. Comparison and Evaluation (Scoring)
    7. Metric Computation
    8. Dashboard Data Transfer (Simulated)
================================================================================
"""

import json
import asyncio
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# ============================================================================
# IMPORTS FROM THE ACTUAL CODEBASE (for demonstration authenticity)
# ============================================================================
import config
from data_generator import DataGenerator
from prompts import (
    ZeroShotPrompt, FewShotPrompt, ChainOfThoughtPrompt, 
    ProgramOfThoughtPrompt, PromptTemplate, get_prompt
)
from llm_clients import get_client, LLMClient
from response_parser import ResponseParser, ParsedResponse, HypothesesModel
from statistical_engine import StatisticalEngine
from evaluator import EvaluationMetrics

# ============================================================================
# DISPLAY UTILITIES FOR PRESENTATION
# ============================================================================

def print_header(title: str, char: str = "=", width: int = 80):
    """Print a formatted header for each workflow stage."""
    print("\n" + char * width)
    print(f" {title}")
    print(char * width)

def print_subheader(title: str, char: str = "-", width: int = 60):
    """Print a formatted subheader."""
    print(f"\n{char * 5} {title} {char * 5}")

def print_json(data: Any, indent: int = 2):
    """Pretty-print JSON data."""
    def default_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return str(obj)
    print(json.dumps(data, indent=indent, default=default_serializer))

def print_box(content: str, title: str = None, width: int = 76):
    """Print content in a box for emphasis."""
    border = "+" + "-" * (width - 2) + "+"
    print(border)
    if title:
        title_line = f"| {title.center(width - 4)} |"
        print(title_line)
        print("|" + "-" * (width - 2) + "|")
    for line in content.split('\n'):
        # Handle long lines by wrapping
        while len(line) > width - 4:
            print(f"| {line[:width-4]} |")
            line = line[width-4:]
        print(f"| {line.ljust(width - 4)} |")
    print(border)


# ============================================================================
# USER SELECTION UTILITIES
# ============================================================================

def get_available_models() -> Dict[str, list]:
    """Get available models based on configured API keys."""
    available = {}
    
    # Check each provider's API key
    provider_models = {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-5.1", "gpt-5-mini"],
        "anthropic": ["claude-sonnet-4-5-20250929", "claude-opus-4-5-20251101", "claude-haiku-4-5-20251001"],
        "google": ["gemini-2.5-pro", "gemini-2.5-flash"],
        "grok": ["grok-3", "grok-3-mini", "grok-4-fast"],
        "deepseek": ["deepseek-chat"],
    }
    
    for provider, models in provider_models.items():
        if config.API_KEYS.get(provider):
            available[provider] = models
    
    return available

def display_model_menu(available_models: Dict[str, list]) -> Tuple[str, str]:
    """Display model selection menu and get user choice."""
    print("\n" + "=" * 60)
    print(" SELECT LLM MODEL")
    print("=" * 60)
    
    # Flatten models with indices
    model_list = []
    idx = 1
    for provider, models in available_models.items():
        print(f"\n  📦 {provider.upper()}")
        for model in models:
            print(f"      [{idx}] {model}")
            model_list.append((provider, model))
            idx += 1
    
    print("\n" + "-" * 60)
    
    while True:
        try:
            choice = input("  Enter model number (or press Enter for default [1]): ").strip()
            if choice == "":
                choice = 1
            else:
                choice = int(choice)
            
            if 1 <= choice <= len(model_list):
                provider, model = model_list[choice - 1]
                print(f"\n  ✅ Selected: {provider}/{model}")
                return provider, model
            else:
                print(f"  ❌ Invalid choice. Please enter 1-{len(model_list)}")
        except ValueError:
            print("  ❌ Please enter a valid number")

def display_prompt_menu() -> str:
    """Display prompt style selection menu and get user choice."""
    print("\n" + "=" * 60)
    print(" SELECT PROMPTING STYLE")
    print("=" * 60)
    
    prompt_styles = [
        ("zero_shot", "Zero-Shot", "Direct question, baseline performance"),
        ("few_shot", "Few-Shot", "Includes worked examples for guidance"),
        ("chain_of_thought", "Chain-of-Thought (CoT)", "Step-by-step reasoning elicitation"),
        ("program_of_thought", "Program-of-Thought (PoT)", "Code-based reasoning with RESULTS block"),
    ]
    
    for idx, (key, name, desc) in enumerate(prompt_styles, 1):
        print(f"\n  [{idx}] {name}")
        print(f"      └── {desc}")
    
    print("\n" + "-" * 60)
    
    while True:
        try:
            choice = input("  Enter prompt style number (or press Enter for default [1]): ").strip()
            if choice == "":
                choice = 1
            else:
                choice = int(choice)
            
            if 1 <= choice <= len(prompt_styles):
                key, name, _ = prompt_styles[choice - 1]
                print(f"\n  ✅ Selected: {name}")
                return key
            else:
                print(f"  ❌ Invalid choice. Please enter 1-{len(prompt_styles)}")
        except ValueError:
            print("  ❌ Please enter a valid number")


def display_test_type_menu() -> str:
    """Display statistical test type selection menu and get user choice."""
    print("\n" + "=" * 60)
    print(" SELECT STATISTICAL TEST TYPE")
    print("=" * 60)
    
    test_types = [
        ("one_sample_t_test", "One-Sample T-Test", "Compare sample mean to a known population mean"),
        ("two_sample_t_test", "Two-Sample T-Test", "Compare means of two independent groups"),
        ("paired_t_test", "Paired T-Test", "Compare means of two related/paired samples"),
    ]
    
    for idx, (key, name, desc) in enumerate(test_types, 1):
        print(f"\n  [{idx}] {name}")
        print(f"      └── {desc}")
    
    print("\n" + "-" * 60)
    
    while True:
        try:
            choice = input("  Enter test type number (or press Enter for default [1]): ").strip()
            if choice == "":
                choice = 1
            else:
                choice = int(choice)
            
            if 1 <= choice <= len(test_types):
                key, name, _ = test_types[choice - 1]
                print(f"\n  ✅ Selected: {name}")
                return key
            else:
                print(f"  ❌ Invalid choice. Please enter 1-{len(test_types)}")
        except ValueError:
            print("  ❌ Please enter a valid number")


# ============================================================================
# MAIN DEMONSTRATION WORKFLOW
# ============================================================================

async def run_demo(provider: str, model_name: str, prompt_type: str, test_type: str = "one_sample_t_test"):
    """
    Execute the full workflow demonstration with visualizations at each stage.
    
    Args:
        provider: LLM provider (openai, anthropic, google, grok, deepseek)
        model_name: Specific model name
        prompt_type: Prompting style (zero_shot, few_shot, chain_of_thought, program_of_thought)
        test_type: Statistical test type (one_sample_t_test, two_sample_t_test, paired_t_test)
    """
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " LLM HYPOTHESIS TESTING BENCHMARK - WORKFLOW DEMONSTRATION ".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" + f" Presentation Date: {datetime.now().strftime('%B %d, %Y at %H:%M')} ".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    print(f"\n🎯 CONFIGURATION: {provider}/{model_name} | Prompt: {prompt_type} | Test: {test_type}")
    
    # ========================================================================
    # STAGE 1: SYNTHETIC DATA GENERATION
    # ========================================================================
    print_header("STAGE 1: SYNTHETIC DATA GENERATION", "=")
    
    print("""
    PURPOSE: Generate reproducible synthetic data with known statistical 
             properties to evaluate LLM hypothesis testing capabilities.
    
    KEY FEATURES:
    • Seeded random number generator (seed=42) for reproducibility
    • Supports multiple test types: one-sample t-test, two-sample t-test, and
      paired t-test. 
    • Configurable sample sizes and distributional parameters
    • Extensibility: New statistical tests, new prompting styles, and LLM providers can be added easily.
    """)
    
    print_subheader("Initializing Data Generator (seed=42)")
    
    # Use the actual DataGenerator from the codebase
    data_generator = DataGenerator(seed=42)
    
    # Generate scenario based on selected test type
    test_type_display = test_type.replace('_', ' ').title()
    print_subheader(f"Generating {test_type_display} Scenario")
    
    if test_type == "one_sample_t_test":
        scenario = data_generator.generate_one_sample_t_test(
            sample_size=30,
            true_mean=12.5,      # The actual population mean
            std=2.5,             # Standard deviation
            null_mean=10.0       # Hypothesized mean under H0
        )
        scenario_info = f"""
Test Type: {scenario['test_type']}
Sample Size: {scenario['metadata']['sample_size']}
True Population Mean (μ): {scenario['metadata']['true_mean']}
Hypothesized Mean (H0: μ = ?): {scenario['metadata']['null_mean']}
Population Std Dev (σ): {scenario['metadata']['std']}
True Effect Size: {scenario['true_effect']} (true_mean - null_mean)
"""
    elif test_type == "two_sample_t_test":
        scenario = data_generator.generate_two_sample_t_test(
            sample_size1=30,
            sample_size2=30,
            mean1=10.0,          # Mean of group 1
            mean2=12.5,          # Mean of group 2
            std1=2.5,            # Std dev of group 1
            std2=2.5,            # Std dev of group 2
            paired=False
        )
        scenario_info = f"""
Test Type: {scenario['test_type']}
Sample Size (Group 1): {scenario['metadata']['sample_size1']}
Sample Size (Group 2): {scenario['metadata']['sample_size2']}
True Mean (Group 1): {scenario['metadata']['mean1']}
True Mean (Group 2): {scenario['metadata']['mean2']}
Std Dev (Group 1): {scenario['metadata']['std1']}
Std Dev (Group 2): {scenario['metadata']['std2']}
True Effect Size: {scenario['true_effect']} (mean2 - mean1)
"""
    elif test_type == "paired_t_test":
        scenario = data_generator.generate_two_sample_t_test(
            sample_size1=30,
            sample_size2=30,
            mean1=10.0,          # Mean before treatment
            mean2=12.5,          # Mean after treatment
            std1=2.5,
            std2=2.5,
            paired=True
        )
        scenario_info = f"""
Test Type: {scenario['test_type']}
Sample Size (Paired): {scenario['metadata']['sample_size1']}
True Mean (Before): {scenario['metadata']['mean1']}
True Mean (After): {scenario['metadata']['mean2']}
Std Dev: {scenario['metadata']['std1']}
True Effect Size: {scenario['true_effect']} (after - before)
"""
    else:
        print(f"\n❌ Unsupported test type: {test_type}")
        return None
    
    print("\n📊 GENERATED SCENARIO DATA:")
    print_box(scenario_info, title="SYNTHETIC DATA ARTIFACT")
    
    print("\n📈 SAMPLE DATA (first 15 values):")
    sample_preview = scenario['sample1'][:15]
    print(f"    Sample 1: {np.round(sample_preview, 4).tolist()}")
    print(f"    ... ({len(scenario['sample1'])} total observations)")
    
    if 'sample2' in scenario:
        sample2_preview = scenario['sample2'][:15]
        print(f"    Sample 2: {np.round(sample2_preview, 4).tolist()}")
        print(f"    ... ({len(scenario['sample2'])} total observations)")
    
    print("\n📊 SAMPLE STATISTICS:")
    print(f"    Sample 1 Mean (x̄₁):  {np.mean(scenario['sample1']):.4f}")
    print(f"    Sample 1 Std (s₁):   {np.std(scenario['sample1'], ddof=1):.4f}")
    if 'sample2' in scenario:
        print(f"    Sample 2 Mean (x̄₂):  {np.mean(scenario['sample2']):.4f}")
        print(f"    Sample 2 Std (s₂):   {np.std(scenario['sample2'], ddof=1):.4f}")
    
    # Store for later stages
    generated_data = scenario
    
    
    # ========================================================================
    # STAGE 2: PROMPT CONSTRUCTION/STYLING
    # ========================================================================
    print_header("STAGE 2: PROMPT CONSTRUCTION/STYLING", "=")
    
    print(f"""
    PURPOSE: Transform synthetic data into natural language prompts using
             different prompting strategies to evaluate LLM performance.
    
    SELECTED PROMPT STYLE: {prompt_type.upper().replace('_', ' ')}
    
    AVAILABLE STRATEGIES:
    • Zero-Shot: Direct question, baseline performance
    • Few-Shot: Includes worked examples for guidance  
    • Chain-of-Thought (CoT): Step-by-step reasoning elicitation
    • Program-of-Thought (PoT): Expects code/computation output
    """)
    
    # Build test context based on selected test type
    test_contexts = {
        "one_sample_t_test": """You are performing a one-sample t-test.
The goal is to determine if the sample mean significantly differs from 
the hypothesized population mean.""",
        "two_sample_t_test": """You are performing an independent two-sample t-test.
The goal is to determine if there is a significant difference between 
the means of two independent groups.""",
        "paired_t_test": """You are performing a paired t-test (dependent samples t-test).
The goal is to determine if there is a significant difference between 
two related measurements (e.g., before and after treatment)."""
    }
    test_context = test_contexts.get(test_type, test_contexts["one_sample_t_test"])
    
    # Use the actual get_prompt function from the codebase
    print_subheader(f"{prompt_type.replace('_', ' ').title()} Prompt Construction")
    
    full_prompt = get_prompt(prompt_type, generated_data, test_context)
    
    print(f"\n📝 FULL {prompt_type.upper().replace('_', ' ')} PROMPT (sent to LLM):")
    # Show full prompt - no truncation for presentation clarity
    print_box(full_prompt, title=f"{prompt_type.upper().replace('_', ' ')} PROMPT ARTIFACT")
    
    
    # ========================================================================
    # STAGE 3: ORCHESTRATION AND LLM CLIENT INTERACTION (REAL API CALL)
    # ========================================================================
    print_header("STAGE 3: ORCHESTRATION & LLM CLIENT INTERACTION (REAL API CALL)", "=")
    
    print(f"""
    PURPOSE: Manage API calls to various LLM providers with:
             • Async execution with semaphore-controlled concurrency
             • Retry logic for transient failures
             • Provider-specific parameter handling
    
    ⏳ THIS IS A LIVE API CALL ⏳
    
    SELECTED PROVIDER: {provider.upper()}
    SELECTED MODEL: {model_name}
    """)
    
    print_subheader("Initializing LLM Client")
    
    # Create the actual LLM client
    try:
        client = get_client(provider, model_name)
        print(f"\n✅ Successfully initialized {provider.upper()} client")
        print(f"    └── Model: {model_name}")
        print(f"    └── Temperature: 0.0 (deterministic)")
        print(f"    └── Max Concurrent Requests: {config.MAX_CONCURRENT_REQUESTS}")
    except Exception as e:
        print(f"\n❌ Failed to initialize client: {e}")
        return None
    
    print_subheader("Making LIVE API Call")
    
    # Determine the API endpoint based on provider
    endpoint_info = {
        "openai": "POST https://api.openai.com/v1/chat/completions",
        "anthropic": "POST https://api.anthropic.com/v1/messages",
        "google": "POST https://generativelanguage.googleapis.com/v1beta/...",
        "grok": "POST https://api.x.ai/v1/chat/completions",
        "deepseek": "POST https://api.deepseek.com/chat/completions",
    }
    
    print(f"\n⏳ Calling {provider.upper()} API...")
    print(f"    └── {endpoint_info.get(provider, 'Unknown endpoint')}")
    print(f"    └── Model: {model_name}")
    print(f"    └── Awaiting response...")
    
    # Make the REAL API call
    start_time = time.perf_counter()
    try:
        raw_response = await client.generate(full_prompt, temperature=0.0)
        latency_seconds = time.perf_counter() - start_time
        print(f"\n✅ Response received in {latency_seconds:.2f} seconds")
    except Exception as e:
        latency_seconds = time.perf_counter() - start_time
        print(f"\n❌ API call failed after {latency_seconds:.2f}s: {e}")
        return None
    
    print("\n📨 RAW LLM RESPONSE:")
    # Truncate for display if very long
    if len(raw_response) > 2000:
        print_box(raw_response[:2000] + "\n\n... [truncated for display]", 
                  title="RAW RESPONSE ARTIFACT (LIVE)")
    else:
        print_box(raw_response, title="RAW RESPONSE ARTIFACT (LIVE)")
    
    
    # ========================================================================
    # STAGE 4: RESPONSE PARSING
    # ========================================================================
    print_header("STAGE 4: RESPONSE PARSING", "=")
    
    print("""
    PURPOSE: Extract structured data from free-form LLM output using:
             • Regex-based pattern matching
             • JSON extraction (for structured outputs)
             • Pydantic validation for data integrity
             • Cascading fallback strategies
    
    EXTRACTED FIELDS:
    • Hypotheses (H0, H1)
    • Test method/name
    • Test statistic value
    • P-value
    • Degrees of freedom
    • Decision (reject/fail to reject H0)
    • Conclusion
    """)
    
    print_subheader("Parsing Raw Response")
    
    # Use the actual ResponseParser from the codebase
    parsed_response = ResponseParser.parse(raw_response, format="auto")
    
    print("\n🔍 PARSED RESPONSE (structured format):")
    
    parsed_dict = {
        "hypotheses": {
            "H0": parsed_response.hypotheses.H0 if parsed_response.hypotheses else None,
            "H1": parsed_response.hypotheses.H1 if parsed_response.hypotheses else None
        },
        "test_method": parsed_response.test_method,
        "test_statistic": parsed_response.test_statistic,
        "p_value": parsed_response.p_value,
        "degrees_of_freedom": parsed_response.degrees_of_freedom,
        "decision": parsed_response.decision,
        "conclusion": parsed_response.conclusion,
        "critical_value": parsed_response.critical_value
    }
    
    print_json(parsed_dict)
    
    print("\n✅ PARSING VALIDATION:")
    print(f"    • P-value in valid range [0,1]: {0 <= (parsed_response.p_value or 0) <= 1}")
    print(f"    • Decision normalized: {parsed_response.decision}")
    print(f"    • Test method extracted: {parsed_response.test_method is not None}")
    
    
    # ========================================================================
    # STAGE 5: GROUND TRUTH IDENTIFICATION
    # ========================================================================
    print_header("STAGE 5: GROUND TRUTH IDENTIFICATION", "=")
    
    print("""
    PURPOSE: Compute authoritative ground truth statistics using SciPy.
             This ensures all LLM outputs are compared against mathematically
             correct reference values, NOT hardcoded answers.
    
    COMPUTATION ENGINE: scipy.stats
    • Provides reliable statistical test implementations
    • Computes exact p-values and test statistics
    • Handles edge cases and numerical precision
    """)
    
    print_subheader("Computing Ground Truth via StatisticalEngine")
    
    # Use the actual StatisticalEngine from the codebase
    ground_truth = StatisticalEngine.compute_ground_truth(generated_data)
    
    print("\n🎯 GROUND TRUTH VALUES (computed via SciPy):")
    print_box(f"""
Test Method: {ground_truth['test_method']}

Hypotheses:
  H0: {ground_truth['hypotheses']['H0']}
  H1: {ground_truth['hypotheses']['H1']}

Test Statistic (t): {ground_truth['test_statistic']:.6f}
P-value: {ground_truth['p_value']:.10f}
Degrees of Freedom: {ground_truth['degrees_of_freedom']}
Critical Value (α=0.05, two-tailed): {ground_truth['critical_value']:.6f}

Decision: {ground_truth['decision']}
Confidence Interval (95%): ({ground_truth['confidence_interval'][0]:.4f}, {ground_truth['confidence_interval'][1]:.4f})
""", title="GROUND TRUTH ARTIFACT")
    
    
    # ========================================================================
    # STAGE 6: COMPARISON AND EVALUATION (SCORING)
    # ========================================================================
    print_header("STAGE 6: COMPARISON AND EVALUATION (SCORING)", "=")
    
    print("""
    PURPOSE: Systematically compare LLM output against ground truth across
             multiple dimensions of correctness and quality.
    
    EVALUATION DIMENSIONS:
    • Test Method Accuracy: Did the LLM select the correct test?
    • P-value Accuracy: Is the p-value within tolerance (±0.05)?
    • Test Statistic Accuracy: Is the statistic within tolerance (±0.1)?
    • Decision Accuracy: Did the LLM make the correct reject/fail decision?
    • Reasoning Quality: Rubric-based scoring of explanation quality
    • Hallucination Detection: Identify impossible/inconsistent values
    """)
    
    print_subheader("Performing Comprehensive Evaluation")
    
    # Use the actual EvaluationMetrics from the codebase
    evaluation_result = EvaluationMetrics.comprehensive_evaluation(
        parsed_response, ground_truth, raw_response
    )
    
    print("\n📊 COMPARISON RESULTS:")
    
    # Test Method
    print(f"\n  ┌─ TEST METHOD COMPARISON")
    print(f"  │   Predicted: {parsed_response.test_method}")
    print(f"  │   Ground Truth: {ground_truth['test_method']}")
    match_status = "✅ MATCH" if evaluation_result['test_method'] == 1.0 else "❌ MISMATCH"
    print(f"  │   Result: {match_status} (score: {evaluation_result['test_method']:.2f})")
    
    # P-value
    print(f"\n  ┌─ P-VALUE COMPARISON")
    print(f"  │   Predicted: {parsed_response.p_value}")
    print(f"  │   Ground Truth: {ground_truth['p_value']:.10f}")
    print(f"  │   Absolute Error: {evaluation_result['p_value']['error']:.6f}" if evaluation_result['p_value']['error'] else "  │   Error: N/A")
    p_match = "✅ WITHIN TOLERANCE" if evaluation_result['p_value']['within_tolerance'] else "❌ OUTSIDE TOLERANCE"
    print(f"  │   Result: {p_match}")
    
    # Test Statistic
    print(f"\n  ┌─ TEST STATISTIC COMPARISON")
    print(f"  │   Predicted: {parsed_response.test_statistic}")
    print(f"  │   Ground Truth: {ground_truth['test_statistic']:.6f}")
    print(f"  │   Absolute Error: {evaluation_result['test_statistic']['error']:.6f}" if evaluation_result['test_statistic']['error'] else "  │   Error: N/A")
    stat_match = "✅ WITHIN TOLERANCE" if evaluation_result['test_statistic']['within_tolerance'] else "❌ OUTSIDE TOLERANCE"
    print(f"  │   Result: {stat_match}")
    
    # Decision
    print(f"\n  ┌─ DECISION COMPARISON")
    print(f"  │   Predicted: {parsed_response.decision}")
    print(f"  │   Ground Truth: {ground_truth['decision']}")
    decision_match = "✅ CORRECT" if evaluation_result['decision']['correct'] else "❌ INCORRECT"
    print(f"  │   Result: {decision_match}")
    
    # Hallucinations
    print(f"\n  ┌─ HALLUCINATION CHECK")
    if evaluation_result['hallucinations']['has_hallucinations']:
        print(f"  │   ⚠️  HALLUCINATIONS DETECTED: {evaluation_result['hallucinations']['count']}")
        for h in evaluation_result['hallucinations']['details']:
            print(f"  │      - {h['type']}: {h['message']}")
    else:
        print(f"  │   ✅ No hallucinations detected")
    
    
    # ========================================================================
    # STAGE 7: METRIC COMPUTATION
    # ========================================================================
    print_header("STAGE 7: METRIC COMPUTATION", "=")
    
    print("""
    PURPOSE: Aggregate comparison results into standardized performance metrics
             for benchmarking and model comparison.
    
    KEY METRICS:
    • Overall Accuracy: Mean of (test_method, p_value, statistic, decision)
    • Decision Accuracy: Binary correct/incorrect for H0 decision
    • Reasoning Quality: Rubric score (0-100%)
    • Hallucination Rate: Proportion of responses with impossible values
    """)
    
    print_subheader("Computing Performance Metrics")
    
    # Extract metrics from evaluation
    overall_accuracy = evaluation_result['overall_accuracy']
    decision_accuracy = 1.0 if evaluation_result['decision']['correct'] else 0.0
    reasoning_score = evaluation_result['reasoning_quality']['percentage']
    hallucination_flag = 1.0 if evaluation_result['hallucinations']['has_hallucinations'] else 0.0
    
    print("\n📈 COMPUTED METRICS FOR THIS EVALUATION:")
    print_box(f"""
OVERALL ACCURACY:     {overall_accuracy * 100:.1f}%
  └─ Test Method:     {evaluation_result['test_method'] * 100:.1f}%
  └─ P-value:         {100 if evaluation_result['p_value']['within_tolerance'] else 0:.1f}%
  └─ Test Statistic:  {100 if evaluation_result['test_statistic']['within_tolerance'] else 0:.1f}%
  └─ Decision:        {decision_accuracy * 100:.1f}%

REASONING QUALITY:    {reasoning_score:.1f}%
  └─ Hypothesis Clarity:    {evaluation_result['reasoning_quality']['scores'].get('hypothesis_clarity', 0)}/1
  └─ Test Justification:    {evaluation_result['reasoning_quality']['scores'].get('test_justification', 0):.2f}/1
  └─ Assumption Checking:   {evaluation_result['reasoning_quality']['scores'].get('assumption_checking', 0)}/1
  └─ Correct Interpretation:{evaluation_result['reasoning_quality']['scores'].get('correct_interpretation', 0)}/1
  └─ Statistical Rigor:     {evaluation_result['reasoning_quality']['scores'].get('statistical_rigor', 0):.2f}/1

HALLUCINATION FLAG:   {"⚠️ YES" if hallucination_flag else "✅ NO"}

RESPONSE COMPLETENESS:
  └─ Has Hypotheses:      {"✅" if evaluation_result['completeness']['has_hypotheses'] else "❌"}
  └─ Has Test Method:     {"✅" if evaluation_result['completeness']['has_test_method'] else "❌"}
  └─ Has Test Statistic:  {"✅" if evaluation_result['completeness']['has_test_statistic'] else "❌"}
  └─ Has P-value:         {"✅" if evaluation_result['completeness']['has_p_value'] else "❌"}
  └─ Has Decision:        {"✅" if evaluation_result['completeness']['has_decision'] else "❌"}
""", title="METRIC COMPUTATION ARTIFACT")
    
    
    # ========================================================================
    # STAGE 8: DASHBOARD DATA TRANSFER (SIMULATED)
    # ========================================================================
    print_header("STAGE 8: DASHBOARD DATA TRANSFER (SIMULATED)", "=")
    
    print("""
    PURPOSE: Package all results into a structured JSON payload for:
             • Persistent storage in results/ directory
             • Consumption by Streamlit dashboard (dashboard/app.py)
             • Aggregation and visualization
    
    DATA FLOW:
    ht.py (orchestrator) → results/*.json → dashboard/app.py → Visualization
    """)
    
    print_subheader("Constructing Dashboard Payload")
    
    # Build metadata based on test type (different structures for different tests)
    if test_type == "one_sample_t_test":
        input_metadata = {
            "sample_size": generated_data['metadata']['sample_size'],
            "true_mean": generated_data['metadata']['true_mean'],
            "null_mean": generated_data['metadata']['null_mean'],
            "std": generated_data['metadata']['std']
        }
    else:  # two_sample_t_test or paired_t_test
        input_metadata = {
            "sample_size1": generated_data['metadata']['sample_size1'],
            "sample_size2": generated_data['metadata']['sample_size2'],
            "mean1": generated_data['metadata']['mean1'],
            "mean2": generated_data['metadata']['mean2'],
            "std1": generated_data['metadata']['std1'],
            "std2": generated_data['metadata']['std2'],
            "paired": generated_data['metadata'].get('paired', False)
        }
    
    # Construct the final payload (mirrors actual output format)
    dashboard_payload = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "provider": provider,
        "prompt_type": prompt_type,
        "input_data": {
            "test_type": generated_data['test_type'],
            "metadata": input_metadata
        },
        "prompt": full_prompt[:200] + "... [truncated]",  # Truncated for display
        "raw_response": raw_response[:200] + "... [truncated]",
        "parsed_results": {
            "hypotheses": parsed_dict['hypotheses'],
            "test_method": parsed_dict['test_method'],
            "test_statistic": parsed_dict['test_statistic'],
            "p_value": parsed_dict['p_value'],
            "degrees_of_freedom": parsed_dict['degrees_of_freedom'],
            "decision": parsed_dict['decision']
        },
        "ground_truth": {
            "test_method": ground_truth['test_method'],
            "test_statistic": round(ground_truth['test_statistic'], 6),
            "p_value": round(ground_truth['p_value'], 10),
            "decision": ground_truth['decision'],
            "degrees_of_freedom": ground_truth['degrees_of_freedom']
        },
        "evaluation": {
            "overall_accuracy": round(overall_accuracy, 4),
            "test_method_accuracy": evaluation_result['test_method'],
            "p_value_within_tolerance": evaluation_result['p_value']['within_tolerance'],
            "decision_correct": evaluation_result['decision']['correct'],
            "reasoning_quality_pct": round(reasoning_score, 2),
            "has_hallucinations": evaluation_result['hallucinations']['has_hallucinations']
        },
        "latency_seconds": round(latency_seconds, 2)
    }
    
    print("\n📦 DASHBOARD DATA PAYLOAD (JSON format):")
    print_json(dashboard_payload)
    
    print("\n💾 STORAGE DESTINATION:")
    print("    └── results/results_YYYYMMDD_HHMMSS.json")
    print("    └── Each benchmark run creates a timestamped JSON file")
    print("    └── Dashboard reads and aggregates all JSON files")
    
    
    # ========================================================================
    # WORKFLOW SUMMARY
    # ========================================================================
    print_header("WORKFLOW DEMONSTRATION COMPLETE", "█")
    
    print(f"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                        WORKFLOW SUMMARY                                ║
    ╠════════════════════════════════════════════════════════════════════════╣
    ║                                                                        ║
    ║  CONFIGURATION:                                                        ║
    ║    Model: {(provider + '/' + model_name).ljust(40)}              ║
    ║    Prompt Style: {prompt_type.ljust(35)}              ║
    ║                                                                        ║
    ║  ┌─────────────────┐                                                   ║
    ║  │ 1. Data Gen     │ → Synthetic scenario with known parameters        ║
    ║  └────────┬────────┘                                                   ║
    ║           ▼                                                            ║
    ║  ┌─────────────────┐                                                   ║
    ║  │ 2. Prompt Build │ → {prompt_type.ljust(40)}   ║
    ║  └────────┬────────┘                                                   ║
    ║           ▼                                                            ║
    ║  ┌─────────────────┐                                                   ║
    ║  │ 3. LLM Call     │ → ⏳ LIVE API CALL ({latency_seconds:.1f}s latency)              ║
    ║  └────────┬────────┘                                                   ║
    ║           ▼                                                            ║
    ║  ┌─────────────────┐                                                   ║
    ║  │ 4. Parse Output │ → Regex + JSON extraction + Pydantic             ║
    ║  └────────┬────────┘                                                   ║
    ║           ▼                                                            ║
    ║  ┌─────────────────┐                                                   ║
    ║  │ 5. Ground Truth │ → SciPy-computed reference values                ║
    ║  └────────┬────────┘                                                   ║
    ║           ▼                                                            ║
    ║  ┌─────────────────┐                                                   ║
    ║  │ 6. Compare      │ → Multi-dimensional accuracy checks              ║
    ║  └────────┬────────┘                                                   ║
    ║           ▼                                                            ║
    ║  ┌─────────────────┐                                                   ║
    ║  │ 7. Compute Metrics│ → Overall accuracy, reasoning, hallucinations  ║
    ║  └────────┬────────┘                                                   ║
    ║           ▼                                                            ║
    ║  ┌─────────────────┐                                                   ║
    ║  │ 8. Dashboard    │ → JSON storage → Streamlit visualization         ║
    ║  └─────────────────┘                                                   ║
    ║                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n📊 KEY RESULTS FROM THIS DEMONSTRATION:")
    print(f"    • Provider: {provider}")
    print(f"    • Model Evaluated: {model_name}")
    print(f"    • Prompt Style: {prompt_type}")
    print(f"    • Test Type: {generated_data['test_type']}")
    print(f"    • Overall Accuracy: {overall_accuracy * 100:.1f}%")
    print(f"    • Decision Correct: {'Yes ✅' if evaluation_result['decision']['correct'] else 'No ❌'}")
    print(f"    • Reasoning Quality: {reasoning_score:.1f}%")
    print(f"    • Hallucinations: {'Detected ⚠️' if hallucination_flag else 'None ✅'}")
    print(f"    • API Latency: {latency_seconds:.2f}s")
    
    print("\n" + "═" * 80)
    print(" END OF DEMONSTRATION - Thank you for your attention!")
    print("═" * 80 + "\n")
    
    return dashboard_payload


def main():
    """
    Main entry point - handles user selection and runs the demo.
    """
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " LLM HYPOTHESIS TESTING BENCHMARK ".center(78) + "█")
    print("█" + " INTERACTIVE WORKFLOW DEMONSTRATION ".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # Check for available models
    available_models = get_available_models()
    
    if not available_models:
        print("\n❌ ERROR: No API keys configured!")
        print("   Please set up your .env file with at least one of:")
        print("   - OPENAI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - GOOGLE_API_KEY")
        print("   - GROK_API_KEY")
        print("   - DEEPSEEK_API_KEY")
        return None
    
    print(f"\n✅ Found {sum(len(m) for m in available_models.values())} models across {len(available_models)} providers")
    
    # Get user selections
    provider, model_name = display_model_menu(available_models)
    prompt_type = display_prompt_menu()
    test_type = display_test_type_menu()
    
    print("\n" + "=" * 60)
    print(" STARTING DEMONSTRATION")
    print("=" * 60)
    print(f"\n  🚀 Launching workflow with:")
    print(f"     • Provider: {provider}")
    print(f"     • Model: {model_name}")
    print(f"     • Prompt Style: {prompt_type}")
    print(f"     • Test Type: {test_type}")
    
    input("\n  Press Enter to begin the demonstration...")
    
    # Run the async demonstration
    result = asyncio.run(run_demo(provider, model_name, prompt_type, test_type))
    
    return result


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Run the workflow demonstration.
    
    Usage:
        python workflow_demo.py
        
        # Or with command-line arguments to skip interactive selection:
        python workflow_demo.py --provider openai --model gpt-4o --prompt zero_shot
    
    This script is designed for live presentations and will:
    1. Let you select an LLM model from available providers
    2. Let you choose a prompting style
    3. Make a REAL API call to the selected LLM
    4. Display formatted output for each stage of the hypothesis testing workflow
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Hypothesis Testing Workflow Demonstration")
    parser.add_argument("--provider", type=str, help="LLM provider (openai, anthropic, google, grok, deepseek)")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--prompt", type=str, choices=["zero_shot", "few_shot", "chain_of_thought", "program_of_thought"],
                       help="Prompting style")
    parser.add_argument("--test", type=str, choices=["one_sample_t_test", "two_sample_t_test", "paired_t_test"],
                       default="one_sample_t_test", help="Statistical test type (default: one_sample_t_test)")
    
    args = parser.parse_args()
    
    # If all required arguments provided, skip interactive selection
    if args.provider and args.model and args.prompt:
        print("\n" + "█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + " LLM HYPOTHESIS TESTING BENCHMARK ".center(78) + "█")
        print("█" + " WORKFLOW DEMONSTRATION (CLI MODE) ".center(78) + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80)
        
        print(f"\n  🚀 Launching workflow with CLI arguments:")
        print(f"     • Provider: {args.provider}")
        print(f"     • Model: {args.model}")
        print(f"     • Prompt Style: {args.prompt}")
        print(f"     • Test Type: {args.test}")
        
        result = asyncio.run(run_demo(args.provider, args.model, args.prompt, args.test))
    else:
        # Interactive mode
        result = main()
