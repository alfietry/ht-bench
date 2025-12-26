# LLM Hypothesis Testing Benchmark - ht-bench

A research benchmark evaluating LLMs on statistical hypothesis testing tasks across 16 models, 4 prompting strategies, and 3 T-test types.

## 🎯 Key Findings

- **Top performers**: Gemini 2.5 Pro (85.5%), Grok-3 (82.4%)
- **Critical insight**: "Outcome-Process Dissociation" — models achieve 98.7% sensitivity but fail on test statistic derivation
- **Prompting paradox**: Program-of-Thought > Few-Shot > Zero-Shot > Chain-of-Thought
- **Systematic weakness**: Paired T-Test accuracy drops ~30% vs other tests

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt
.env  # Add API keys

# Run benchmark
python ht.py --mode quick          # Fast test (1 model, 2 scenarios)
python ht.py --mode full           # All models, all tests
python ht.py --mode custom --models openai/gpt-4o --tests one_sample_t_test --scenarios 5

## 🌐 Live Demo

Try the interactive dashboard: **[ht-bench.streamlit.app](https://ht-bench-alfietry.streamlit.app)**

# View results
streamlit run dashboard/app.py
```

## 📁 Architecture

```
ht.py                 → Orchestrator (async batch evaluation)
├── data_generator.py → Synthetic scenarios (t-tests, sample sizes)
├── prompts.py        → 4 strategies: zero_shot, few_shot, cot, pot
├── llm_clients.py    → OpenAI, Anthropic, Google, Grok, DeepSeek
├── response_parser.py→ JSON/regex extraction cascade
├── statistical_engine.py → SciPy ground truth
└── evaluator.py      → Metrics (accuracy, reasoning, hallucination)
         ↓
results/*.json → dashboard/app.py (Streamlit visualization)
```

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Overall Accuracy | Mean of test-method, p-value, decision accuracy |
| Decision Accuracy | Correct reject/fail-to-reject at α=0.05 |
| P-value Accuracy | Within ±0.05 tolerance |
| Reasoning Quality | Rubric score [0,1] for explanation quality |
| Hallucination Rate | Invalid values, contradictory decisions |

## 🔧 Configuration

Edit `config.py`:
- `FULL_MODE_MODEL_MAP`: Models for full benchmark
- `EVALUATION["p_value_tolerance"]`: Default 0.05
- `RANDOM_SEED = 42`: Reproducibility

## 📝 Adding Components

**New LLM provider:**
1. Subclass `LLMClient` in `llm_clients.py`
2. Add to `config.py` API_KEYS and MODELS
3. Register in `get_client()` factory

**New statistical test:**
1. Add generator in `data_generator.py`
2. Add computation in `statistical_engine.py`
3. Register in `config.TEST_TYPES`

## 📄 License

For educational and research purposes.