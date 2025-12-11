"""
Configuration management for LLM Hypothesis Testing Benchmark
"""
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# API Configuration
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "google": os.getenv("GOOGLE_API_KEY"),
    "grok": os.getenv("GROK_API_KEY"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY"),
}

# Model configurations
MODELS = {
    "openai": ["gpt-5.1", "gpt-5-mini","gpt-4o", "gpt-4"],
    "anthropic": ["claude-opus-4-5-20251101-thinking-32k", "claude-opus-4-1-20250805"],
    "google": ["gemini-3.0-pro", "gemini-2.5-pro", "gemini-2.5-flash"],
    "grok": ["grok-4.1","grok-3-mini", "grok-4-fast"],
    "deepseek": ["deepseek-v3.2-exp-thinking","deepseek-math", "deepseek-chat"],
    # "ollama": ["llama3.1:8b", "mistral:7b", "qwen2.5:7b"],  # Disabled - requires local server
}

# Canonical model set for the full benchmark mode (mirrors dashboard filtering)
FULL_MODE_MODEL_MAP = {
    "openai": ["gpt-5.1","gpt-5-mini","gpt-4o"],
    "anthropic": ["claude-opus-4-5-20251101", "claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"], 
    "google": ["gemini-2.5-pro", "gemini-2.5-flash"], # , "gemini-3.0-pro"
    "grok": ["grok-3", "grok-4.1-thinking", "grok-3-mini", "grok-4-1-fast-reasoning"],
    "deepseek": ["deepseek-chat"],
}

# Statistical test configurations
TEST_TYPES = [
    "one_sample_t_test",
    "two_sample_t_test",
    "paired_t_test",
    # "one_sample_z_test",
    # "two_sample_z_test",
    # "anova",
    # "chi_square_goodness_of_fit",  # Disabled
    # "chi_square_independence",      # Disabled
]

# Distribution parameters
DISTRIBUTIONS = {
    "normal": {"mean": [0, 10, 50], "std": [1, 5, 15]},
    "t": {"df": [5, 10, 30]},
    "exponential": {"scale": [1, 2, 5]},
    "uniform": {"low": [0, -10], "high": [10, 20]},
    "chi_square": {"df": [2, 5, 10]},
}

# Evaluation settings
EVALUATION = {
    "p_value_tolerance": 0.05,
    "test_statistic_tolerance": 0.1,
    "significance_level": 0.05,
    "reasoning_rubric": {
        "hypothesis_clarity": 1,
        "test_justification": 1,
        "assumption_checking": 1,
        "correct_interpretation": 1,
        "statistical_rigor": 1,
    }
}

# Prompt types
PROMPT_TYPES = [
    "zero_shot",
    "few_shot",
    "chain_of_thought",
    "program_of_thought",
]

# Reproducibility
RANDOM_SEED = 42
SAMPLE_SIZES = [20, 50]

# Async settings
MAX_CONCURRENT_REQUESTS = 5
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds

# Logging
import logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO
