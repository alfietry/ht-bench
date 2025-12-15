# LLM Hypothesis Testing Benchmark - AI Coding Instructions

## Architecture Overview

This is an async benchmark framework evaluating LLMs on statistical hypothesis testing. The data flow:

```
ht.py (orchestrator) → data_generator.py → prompts.py → llm_clients.py → response_parser.py → statistical_engine.py → evaluator.py
                                                                                                                            ↓
                                                                                               dashboard/app.py ← results/*.json
```

**Key design decisions:**
- All LLM calls are async with semaphore-controlled concurrency (`MAX_CONCURRENT_REQUESTS=5`)
- Ground truth is computed via SciPy in `statistical_engine.py`, not hardcoded
- Response parsing uses regex fallbacks when structured JSON extraction fails
- Results are JSON files in `results/` consumed by Streamlit dashboard

## Essential Commands

```bash
# Quick test (single model, 2 scenarios)
python ht.py --mode quick

# Full benchmark (all configured models)
python ht.py --mode full

# Custom run - specify provider/model format
python ht.py --mode custom --models openai/gpt-4o anthropic/claude-sonnet-4-5-20250929 --tests one_sample_t_test --scenarios 5

# Dashboard (reads from results/)
streamlit run dashboard/app.py
```

## Code Conventions

### Adding New LLM Providers
1. Create client class in `llm_clients.py` inheriting `LLMClient`
2. Implement `generate()` and `generate_structured()` methods
3. Add provider to `API_KEYS` and `MODELS` dicts in `config.py`
4. Register in `get_client()` factory function in `llm_clients.py`

### Adding New Statistical Tests
1. Add generation method in `data_generator.py` (e.g., `generate_anova()`)
2. Add computation method in `statistical_engine.py` (e.g., `compute_anova()`)
3. Register test type in `config.TEST_TYPES` list
4. Handle in `evaluate_batch()` scenario generation switch in `ht.py`

### Response Parsing Pattern
`response_parser.py` uses a cascading extraction strategy:
1. Try `RESULT:` pattern from Program-of-Thought output
2. Try JSON extraction with regex
3. Fall back to individual field extraction (`extract_number()`, `extract_hypotheses()`)

Always validate parsed `p_value` is in [0,1] range - values outside indicate test statistic confusion.

## Configuration

All tunables live in `config.py`:
- `FULL_MODE_MODEL_MAP`: canonical model set for full benchmarks
- `EVALUATION["p_value_tolerance"]`: default 0.05 for p-value accuracy
- `EVALUATION["significance_level"]`: α = 0.05
- `RANDOM_SEED = 42`: reproducibility seed for data generation

API keys loaded from `.env` file (copy `.env.example`).

## Testing & Validation

```bash
# Run pytest (if tests exist)
pytest tests/ -v

# Validate single model quickly
python ht.py --mode custom --models openai/gpt-4o-mini --tests one_sample_t_test --scenarios 1
```

## Prompt Types

Located in `prompts.py`, each affects evaluation differently:
- `zero_shot`: Direct question, baseline performance
- `few_shot`: Includes worked examples
- `chain_of_thought`: Step-by-step reasoning elicitation
- `program_of_thought`: Expects code/computation output with `RESULT:` line

## Dashboard Filtering

`dashboard/app.py` filters results by models in `FULL_MODE_MODEL_MAP`. When adding models, ensure they appear in both `config.MODELS` (for running) and `config.FULL_MODE_MODEL_MAP` (for dashboard visibility).

## Common Gotchas

- OpenAI models with "mini" or "o1/o2/o3" use `max_completion_tokens` instead of `max_tokens`
- Temperature is fixed for reasoning models (handled in `OpenAIClient._completion_kwargs`)
- Anthropic requires `anthropic-version` header (set in client init)
- Dashboard caches results with `@st.cache_data` - reload page after new benchmark runs
