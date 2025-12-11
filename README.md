# LLM Hypothesis Testing Benchmark

A comprehensive, research-grade benchmark for evaluating Large Language Models (LLMs) on statistical hypothesis testing tasks.

## 🎯 Overview

This benchmark systematically evaluates LLMs' ability to:
- Select appropriate statistical tests
- Formulate correct hypotheses
- Calculate test statistics and p-values
- Make correct statistical decisions
- Provide rigorous reasoning

## 🏗️ Architecture

```
final-proj-bench/
├── config.py                 # Configuration and settings
├── ht.py                     # Main orchestration script
├── llm_clients.py           # LLM API integrations (OpenAI, Anthropic, Google, etc.)
├── prompts.py               # Prompt templates (zero-shot, CoT, PoT, few-shot)
├── data_generator.py        # Synthetic data generation for various distributions
├── statistical_engine.py    # Ground truth calculations
├── response_parser.py       # LLM response parsing and validation
├── evaluator.py            # Evaluation metrics and aggregation
├── dashboard/
│   └── app.py              # Streamlit dashboard for visualization
├── data/                   # Generated datasets
├── results/                # Benchmark results (JSON)
└── logs/                   # Execution logs
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env with your API keys
```

### 2. Run Quick Test

```bash
# Test with a subset of models and scenarios
python ht.py --mode quick
```

### 3. View Results

```bash
# Launch interactive dashboard
streamlit run dashboard/app.py
```

## 📊 Usage

### Run Full Benchmark

```bash
# Comprehensive evaluation across all models and test types
python ht.py --mode full
```

### Custom Benchmark

```bash
# Specify models, prompt types, and test types
python ht.py --mode custom \
  --models openai/gpt-4o anthropic/claude-3-5-sonnet-20241022 \
  --prompts zero_shot chain_of_thought \
  --tests one_sample_t_test two_sample_t_test anova \
  --sample-sizes 50 100 500 \
  --scenarios 5
```

### Available Models

- **OpenAI**: gpt-4o, gpt-4o-mini, gpt-4-turbo
- **Anthropic**: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
- **Google**: gemini-1.5-pro, gemini-1.5-flash
- **Grok**: grok-beta
- **DeepSeek**: deepseek-chat
- **Ollama**: llama3.1:8b, mistral:7b, qwen2.5:7b (local)

### Prompt Types

- **Zero-shot**: Direct task instruction
- **Few-shot**: Examples included
- **Chain-of-Thought (CoT)**: Step-by-step reasoning
- **Program-of-Thought (PoT)**: Code-based solutions

### Test Types

- One-sample t-test
- Two-sample t-test
- Paired t-test
- One-sample z-test
- Two-sample z-test
- One-way ANOVA
- Chi-square goodness of fit
- Chi-square test of independence

## 📈 Evaluation Metrics

### Core Metrics
- **Test Method Accuracy**: Correct statistical test selection
- **P-value Accuracy**: Numerical accuracy (tolerance: ±0.05)
- **Test Statistic Accuracy**: Numerical accuracy (tolerance: ±0.1)
- **Decision Accuracy**: Correct hypothesis test decision

### Quality Metrics
- **Reasoning Quality**: 5-point rubric scoring
  - Hypothesis clarity
  - Test justification
  - Assumption checking
  - Correct interpretation
  - Statistical rigor
- **Hallucination Detection**: Invalid values, inconsistent decisions
- **Completeness**: Presence of all required components

## 🎨 Dashboard Features

The Streamlit dashboard provides:

- **Model Comparison**: Radar charts and performance rankings
- **Test Type Analysis**: Heatmaps showing strengths/weaknesses
- **Prompt Analysis**: Performance by prompting strategy
- **Error Analysis**: Distribution of numerical errors
- **Hallucination Tracking**: Frequency and types of errors
- **Detailed Results Table**: Filterable, sortable, exportable

## 🔬 Research Features

### Reproducibility
- Fixed random seeds for data generation
- Comprehensive logging of all evaluations
- Version control friendly output format

### Extensibility
- Modular design for easy addition of:
  - New models (via client interface)
  - New test types (via generator/engine)
  - New prompting strategies (via prompt templates)
  - New evaluation metrics (via evaluator)

### Async Processing
- Parallel LLM queries with rate limiting
- Configurable concurrency (default: 5 concurrent requests)
- Automatic retry with exponential backoff

## 📝 Output Format

Results are saved as JSON with comprehensive metadata:

```json
{
  "timestamp": "2025-11-22T10:30:00",
  "model": "gpt-4o",
  "prompt_type": "chain_of_thought",
  "input_data": {
    "test_type": "two_sample_t_test",
    "metadata": {...}
  },
  "raw_response": "...",
  "parsed_results": {
    "hypotheses": {"H0": "...", "H1": "..."},
    "test_method": "two_sample_t_test",
    "test_statistic": 2.45,
    "p_value": 0.018,
    "decision": "reject_H0",
    "conclusion": "..."
  },
  "ground_truth": {
    "test_method": "two_sample_t_test",
    "test_statistic": 2.47,
    "p_value": 0.017,
    "decision": "reject_H0"
  },
  "evaluation": {
    "overall_accuracy": 0.95,
    "test_method": 1.0,
    "p_value": {"within_tolerance": true, "error": 0.001},
    "decision": {"correct": true},
    "reasoning_quality": {"percentage": 85.0},
    "hallucinations": {"has_hallucinations": false}
  }
}
```

## ⚙️ Configuration

Edit `config.py` to customize:

- API endpoints and keys
- Model lists
- Test types and distributions
- Sample sizes
- Evaluation tolerances
- Concurrency limits
- Random seeds

## 🧪 Testing

```bash
# Run quick test to verify setup
python ht.py --mode quick

# Test specific components
python -c "from data_generator import DataGenerator; dg = DataGenerator(); print(dg.generate_one_sample_t_test(50))"
python -c "from statistical_engine import StatisticalEngine; import numpy as np; print(StatisticalEngine.compute_one_sample_t_test(np.random.normal(10, 2, 50), 10))"
```

## 📚 Dependencies

Core:
- numpy, scipy, pandas (statistical computing)
- openai, anthropic, google-generativeai (LLM APIs)
- aiohttp, asyncio (async processing)
- pydantic (validation)
- tenacity (retries)

Visualization:
- streamlit (dashboard)
- plotly, matplotlib, seaborn (charts)

See `requirements.txt` for complete list.

## 🤝 Contributing

This benchmark is designed for research use. To extend:

1. **Add new models**: Implement `LLMClient` interface in `llm_clients.py`
2. **Add test types**: Add generator in `data_generator.py` and calculator in `statistical_engine.py`
3. **Add prompts**: Create new template class in `prompts.py`
4. **Add metrics**: Extend `EvaluationMetrics` in `evaluator.py`

## 📄 License

For educational and research purposes.

## 🙏 Acknowledgments

Built for comprehensive evaluation of LLM capabilities in statistical reasoning and hypothesis testing.

---

**Note**: Ensure API keys are properly configured in `.env` before running the benchmark. For local models, ensure Ollama is running at `http://localhost:11434`.
