# LLM Hypothesis Testing Benchmark

Evaluate LLMs on end-to-end statistical hypothesis testing: test selection, statistic/p-value computation, and decision-making at α = 0.05.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # add your API keys
python ht.py --mode quick
streamlit run dashboard/app.py
```

## Run Modes

| Mode | Command |
|------|---------|
| Quick test | `python ht.py --mode quick` |
| Full benchmark | `python ht.py --mode full` |
| Custom | `python ht.py --mode custom --models openai/gpt-4o --tests one_sample_t_test --scenarios 5` |

## Supported Models

- **OpenAI**: gpt-5.1, gpt-5-mini, gpt-4o, gpt-4
- **Anthropic**: claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5
- **Grok**: grok-4-fast, grok-3, grok-3-mini
- **Google**: gemini-2.5-pro, gemini-2.5-flash
- **DeepSeek**: deepseek-chat

## Prompt Strategies

- Zero-Shot
- Few-Shot
- Chain-of-Thought (CoT)
- Program-of-Thought (PoT)

## Test Types

- One-sample t-test
- Two-sample t-test
- Paired t-test

## Evaluation Metrics

- **Overall Accuracy** – combined score  
- **Decision Accuracy** – reject / fail-to-reject correctness  
- **P-value Accuracy** – within tolerance of ground truth  
- **Reasoning Quality** – rubric-based (hypotheses, justification, interpretation)  
- **Hallucination Flag** – contradictory or invalid outputs  
- **Latency** – response time

## Dashboard

Interactive Streamlit UI with leaderboard, radar charts by model family, heatmaps, p-value scatter, and qualitative inspector.

```bash
streamlit run dashboard/app.py
```

## Project Structure

```
ht.py               # orchestrator
config.py           # settings & model lists
llm_clients.py      # API clients (OpenAI, Anthropic, Google, Grok, DeepSeek)
prompts.py          # prompt templates
data_generator.py   # synthetic scenario generator
statistical_engine.py # ground-truth computation
response_parser.py  # LLM output parsing
evaluator.py        # metrics & aggregation
dashboard/app.py    # Streamlit dashboard
results/            # JSON output
```

## Configuration

Edit `config.py` for models, test types, sample sizes, tolerances, and concurrency.

