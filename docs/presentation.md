# Benchmarking Large Language Models on Statistical Hypothesis Testing
## A Comprehensive Evaluation Framework for AI-Enabled Scientific Discovery

---

# Slide 1: Title Slide

## **Benchmarking Large Language Models on Statistical Hypothesis Testing**

### Alfred K. Adzika
School of Electrical Engineering and Computer Science  
Ohio University

📧 aa832423@ohio.edu

**December 2025**

---

# Slide 2: Executive Summary

## 🎯 **Key Findings at a Glance**

| Metric | Result |
|--------|--------|
| **Top Overall Accuracy** | Gemini 2.5 Pro (85.5%) |
| **Decision Sensitivity (Recall)** | 98.7% |
| **Best Prompting Strategy** | Program-of-Thought (PoT) |
| **Critical Failure** | Paired T-Test (~60% accuracy) |
| **Error Bias** | 10:1 False Positive to False Negative ratio |

### **Core Discovery: "Outcome-Process Dissociation"**
> Models achieve correct final decisions but fail intermediate calculations

📊 *[IMAGE: Executive summary infographic showing key statistics with icons]*

---

# Slide 3: Agenda / Roadmap

## 📋 **Presentation Outline**

1. 🔬 **Motivations** – Why benchmark LLMs on hypothesis testing?
2. 🎯 **Goals & Objectives** – What we aim to achieve
3. 🏗️ **Architecture & Methodology** – System design and workflow
4. 📈 **Evaluation Metrics** – How we measure performance
5. 📊 **Results & Analysis** – Key findings
6. 💡 **Key Insights** – Critical observations
7. 🔮 **Future Work** – Next steps

📊 *[IMAGE: Flowchart showing the presentation roadmap with numbered boxes connected by arrows]*

---

# Slide 4: Motivations - The AI Safety Challenge

## ⚠️ **The Challenge: AI in High-Stakes Domains**

**LLMs are being deployed in:**
- 🏥 **Medicine** – Clinical decision support
- 🎓 **Academia** – Research assistance  
- 🛡️ **Defense** – Strategic analysis
- 💰 **Finance** – Risk assessment

### **The Risk**
> Current LLM agent systems carry inherent risks: **unpredictable reasoning**, **potential for deception**, and **embedded biases**
> 
> — Yoshua Bengio, 2025

📊 *[IMAGE: Illustration showing an AI brain with warning symbols connected to icons representing medicine, academia, defense, and finance domains]*

---

# Slide 5: Motivations - The Scientist AI Vision

## 🔬 **Yoshua Bengio's "Scientist AI" Framework**

### **Key Principle:**
> A critical pathway to safe, non-agentic AI is to prioritize a system's ability to **explain the world from observations** rather than merely acting within it.

### **Scientific Research Requires:**

```
┌─────────────────┐      ┌─────────────────┐
│   ROBUST        │  +   │   RIGOROUS      │
│   REASONING     │      │   VALIDATION    │
└────────┬────────┘      └────────┬────────┘
         │                        │
         └──────────┬─────────────┘
                    ▼
         ┌─────────────────────┐
         │   HYPOTHESIS        │
         │   TESTING           │
         └─────────────────────┘
```

📊 *[IMAGE: Diagram showing the Scientist AI framework with hypothesis testing as a central component connecting observation, reasoning, and validation]*

---

# Slide 6: Motivations - The ImageNet Inspiration

## 📐 **Inspired by ImageNet**

### **The ImageNet Revolution (2012)**
- Provided **common metrics** for computer vision
- Enabled **objective measurement** of improvement
- Catalyzed the deep learning revolution

### **Our Vision:**
> The field of **AI-enabled science** requires a rigorous infrastructure to quantify reasoning and validation skills

```
╔═══════════════════════════════════════════════════════════════╗
║    ImageNet → Computer Vision  ≈  Our Benchmark → AI Science  ║
╚═══════════════════════════════════════════════════════════════╝
```

📊 *[IMAGE: Side-by-side comparison showing ImageNet dataset samples on the left, and statistical hypothesis testing problems on the right, with an arrow pointing from "Standardized Benchmarks" to "Revolutionary Progress"]*

---

# Slide 7: The Research Gap

## 🕳️ **Identified Gap in Literature**

### **Existing Benchmarks Cover:**
| Benchmark | Focus |
|-----------|-------|
| LogicBench | Propositional & first-order logic |
| Multi-LogiEval | Multi-step logical reasoning |
| FactReasoner | Long-form factuality |
| Liu et al. | Quantitative & causal reasoning |

### **What's Missing:**
> ❌ No standardized benchmark for **end-to-end hypothesis testing** as a unified task

### **Our Contribution:**
> ✅ First dedicated benchmark evaluating hypothesis testing as an **integrated reasoning AND computational** capability

📊 *[IMAGE: Venn diagram showing existing benchmarks on one side and "Hypothesis Testing" in a separate circle, with our work bridging the gap]*

---

# Slide 8: Goals & Objectives

## 🎯 **Research Objectives**

### **Primary Goal:**
> Design and implement a benchmark that evaluates how effectively modern LLMs can perform hypothesis testing

### **Success Criteria:**

| Dimension | Metric |
|-----------|--------|
| **Test Selection** | Consistent identification of correct test type |
| **P-value Estimation** | Calibrated probability calculations |
| **Decision Making** | Correct reject/fail-to-reject at α = 0.05 |

### **Contributions:**
1. 📋 A **formal benchmark** for hypothesis testing with LLMs
2. 📊 An **empirical evaluation** across major architectures
3. 📝 An **analysis of strengths and limitations**

📊 *[IMAGE: Target/bullseye diagram with three concentric rings labeled with the three success criteria]*

---

# Slide 9: Architecture Overview

## 🏗️ **System Architecture**

### **Modular Pipeline Design:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ht.py (Orchestrator)                             │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ data_generator│  │   prompts.py  │  │ llm_clients.py│
│     .py       │  │               │  │               │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                ┌─────────────────────┐
                │ response_parser.py  │
                └──────────┬──────────┘
                           ▼
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
┌───────────────────┐              ┌───────────────────┐
│ statistical_engine│              │   evaluator.py    │
│       .py         │              │                   │
└─────────┬─────────┘              └─────────┬─────────┘
          │                                  │
          └────────────────┬─────────────────┘
                           ▼
                 ┌───────────────────┐
                 │  dashboard/app.py │
                 │    (Streamlit)    │
                 └───────────────────┘
```

📊 *[IMAGE: Full architecture flowchart with colored boxes for each component, showing data flow with arrows - include the ht-bench-art.png referenced in the report]*

---

# Slide 10: Core Components - Data Generation

## 🔢 **Component 1: Synthetic Data Generation**

### **DataGenerator Class** (`data_generator.py`)

```python
class DataGenerator:
    def __init__(self, seed: int = 42):  # Reproducibility
        self.rng = np.random.default_rng(seed)
    
    def generate_one_sample_t_test(self, sample_size, ...):
        sample = self.generate_normal(true_mean, std, sample_size)
        return {
            "test_type": "one_sample_t_test",
            "sample1": sample,
            "population_mean": null_mean,
            ...
        }
```

### **Supported Test Types:**
| Test | Description |
|------|-------------|
| **One-Sample T-Test** | Compare sample mean to known population mean |
| **Two-Sample T-Test** | Compare means of two independent groups |
| **Paired T-Test** | Compare means of two related samples |

📊 *[IMAGE: Visual representation of synthetic data generation - bell curves with sample points marked]*

---

# Slide 11: Core Components - Ground Truth

## 📐 **Component 2: Ground Truth Computation**

### **StatisticalEngine Class** (`statistical_engine.py`)

> All ground truth computed via **SciPy** – NOT hardcoded!

```python
@staticmethod
def compute_one_sample_t_test(sample, population_mean):
    n = len(sample)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    
    # Test statistic
    t_stat = (sample_mean - population_mean) / (sample_std / np.sqrt(n))
    
    # P-value via scipy
    p_value = 2 * stats.t.sf(np.abs(t_stat), df=n-1)
    
    # Decision at α = 0.05
    decision = "reject_H0" if p_value < 0.05 else "fail_to_reject_H0"
```

### **Why This Matters:**
✅ Mathematically correct reference values  
✅ Handles edge cases and numerical precision  
✅ Same implementation for ALL models = Fair comparison

📊 *[IMAGE: Diagram showing SciPy logo computing t-statistic formula with sample data flowing in and ground truth results flowing out]*

---

# Slide 12: Core Components - Prompt Engine

## 📝 **Component 3: Prompt Construction**

### **Four Prompting Strategies:**

| Strategy | Description |
|----------|-------------|
| **Zero-Shot** | Direct question, no examples |
| **Few-Shot** | Includes worked examples |
| **Chain-of-Thought (CoT)** | Step-by-step reasoning |
| **Program-of-Thought (PoT)** | Code-based computation |

### **Zero-Shot Example:**
```text
Analyze this data and perform the appropriate hypothesis test.

Data:
Sample 1: [12.3, 14.1, 13.5, ...] (n=50)
Sample 1 mean: 13.84
Hypothesized population mean: 12.0

Provide:
1. H0 and H1
2. Test name
3. Test statistic value
4. P-value (exact numerical value)
5. Decision (reject/fail to reject H0)
...
```

📊 *[IMAGE: Four boxes showing each prompting strategy with a visual icon - lightbulb for zero-shot, examples for few-shot, thought bubbles for CoT, code brackets for PoT]*

---

# Slide 13: Core Components - LLM Clients

## 🤖 **Component 4: Multi-Provider LLM Integration**

### **Supported Providers & Models:**

```
┌─────────────────────────────────────────────────────────────┐
│                    LLMClient (Abstract Base)                │
└─────────────────────────────┬───────────────────────────────┘
                              │
    ┌─────────────┬───────────┼───────────┬─────────────┐
    ▼             ▼           ▼           ▼             ▼
┌────────┐  ┌──────────┐ ┌─────────┐ ┌────────┐  ┌──────────┐
│OpenAI  │  │Anthropic │ │ Google  │ │  Grok  │  │DeepSeek  │
│Client  │  │ Client   │ │ Client  │ │ Client │  │ Client   │
└────────┘  └──────────┘ └─────────┘ └────────┘  └──────────┘
    │             │           │           │             │
    ▼             ▼           ▼           ▼             ▼
 GPT-4o      Claude-4.5   Gemini-2.5   Grok-3/4   DeepSeek
 GPT-5.1      Opus/Son.    Pro/Flash    Mini       Chat
```

### **Design Benefits:**
- ✅ Async execution with semaphore concurrency control
- ✅ Provider-specific parameter handling
- ✅ Easy extensibility – just add new client class

📊 *[IMAGE: Provider logos (OpenAI, Anthropic, Google, xAI, DeepSeek) connected to a central abstract client interface]*

---

# Slide 14: Core Components - Response Parsing

## 🔍 **Component 5: Response Parsing**

### **ResponseParser** - Cascading Extraction Strategy:

```
┌──────────────────────────────────────────────────────────────┐
│                    Raw LLM Response                          │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
              ┌─────│ Try RESULT: pattern │──── Found? ───┐
              │     │    (PoT outputs)    │               │
              │     └─────────────────────┘               │
              │ Not Found                                 ▼
              ▼                                   ┌───────────────┐
     ┌─────────────────────┐                     │ ParsedResponse│
     │ Try JSON extraction │                     │   (Pydantic)  │
     └──────────┬──────────┘                     └───────────────┘
               │ Fails                                    ▲
               ▼                                          │
     ┌─────────────────────┐                              │
     │ Regex-based fallback│──────────────────────────────┘
     │ (hypotheses, p-val, │
     │  decision, etc.)    │
     └─────────────────────┘
```

### **Validation Checks:**
- P-value ∈ [0, 1]
- Decision normalized to `reject_H0` / `fail_to_reject_H0`

📊 *[IMAGE: Flowchart showing parsing cascade with sample raw text input and structured output]*

---

# Slide 15: Evaluation Metrics

## 📏 **How We Measure Performance**

### **Core Metrics:**

| Metric | Description | Tolerance |
|--------|-------------|-----------|
| **Test-Method Accuracy** | Correct test family identification | Exact match |
| **P-Value Accuracy** | Numerical p-value correctness | ± 0.05 |
| **Test Statistic Accuracy** | t-value correctness | ± 0.1 |
| **Decision Accuracy** | Reject/Fail-to-reject correct | Binary |
| **Reasoning Quality** | Rubric score [0-1] | Qualitative |
| **Hallucination Flag** | Impossible/contradictory values | Binary |
| **Latency** | Response time | Seconds |

### **Overall Accuracy Formula:**
```
Overall = mean(test_method_acc, p_value_acc, statistic_acc, decision_acc)
```

📊 *[IMAGE: Dashboard-style gauge charts for each metric type with example values]*

---

# Slide 16: Execution Flow

## 🔄 **Benchmark Execution Pipeline**

### **Step-by-Step Process:**

```
┌────────────────────────────────────────────────────────────────────────┐
│ Step 1: Generate Scenarios                                             │
│   • Sample test types × sample sizes                                   │
│   • DataGenerator creates synthetic data                               │
└────────────────────────────────────────┬───────────────────────────────┘
                                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Step 2: Construct Prompts                                              │
│   • For each (provider, model, prompt_type, scenario)                  │
│   • get_prompt() generates natural language input                      │
└────────────────────────────────────────┬───────────────────────────────┘
                                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Step 3: Async API Calls                                                │
│   • Semaphore-controlled concurrency (MAX_CONCURRENT = 5)              │
│   • Provider-specific client handles request                           │
└────────────────────────────────────────┬───────────────────────────────┘
                                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Step 4: Parse & Evaluate                                               │
│   • ResponseParser extracts structured data                            │
│   • StatisticalEngine computes ground truth                            │
│   • EvaluationMetrics scores response                                  │
└────────────────────────────────────────┬───────────────────────────────┘
                                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Step 5: Aggregate & Visualize                                          │
│   • Results saved as timestamped JSON                                  │
│   • Dashboard displays leaderboards, charts, analysis                  │
└────────────────────────────────────────────────────────────────────────┘
```

📊 *[IMAGE: Pipeline diagram with icons for each step and sample data flowing through]*

---

# Slide 17: Benchmark Modes

## ⚙️ **Flexible Execution Modes**

### **Three Benchmark Modes:**

```bash
# Quick Test - Development/debugging
python ht.py --mode quick
# → Single model, 2 test types, 2 scenarios each

# Full Benchmark - Comprehensive evaluation  
python ht.py --mode full
# → All configured models, all test types, 5 scenarios each

# Custom Run - Targeted testing
python ht.py --mode custom \
    --models openai/gpt-4o anthropic/claude-sonnet-4-5-20250929 \
    --tests one_sample_t_test paired_t_test \
    --scenarios 10
```

### **Configuration:**
```python
# config.py
FULL_MODE_MODEL_MAP = {
    "openai": ["gpt-5.1", "gpt-5-mini", "gpt-4", "gpt-4o", "gpt-4o-mini"],
    "anthropic": ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"],
    "google": ["gemini-2.5-pro", "gemini-2.5-flash"],
    "grok": ["grok-3", "grok-4-fast", "grok-3-mini"],
    "deepseek": ["deepseek-chat"],
}
```

📊 *[IMAGE: Three boxes representing each mode with increasing complexity visual]*

---

# Slide 18: Results - Leaderboard Overview

## 🏆 **Model Performance Leaderboard**

### **Top Performers by Overall Accuracy:**

| Rank | Model | N | Overall Acc. | Decision Acc. | Reasoning | Halluc. | Latency |
|:----:|-------|:---:|:------------:|:-------------:|:---------:|:-------:|:-------:|
| 🥇 | **Gemini-2.5-Pro** | 111 | **85.5%** | 84.7% | 0.73 | 10.8% | 22.7s |
| 🥈 | **Gemini-2.5-Flash** | 143 | **83.2%** | 88.1% | 0.74 | 9.8% | 17.2s |
| 🥉 | **Grok-3** | 356 | **82.4%** | 91.6% | 0.74 | 24.4% | 4.6s |
| 4 | Grok-4-1-f-r | 117 | 81.0% | 88.9% | 0.74 | 17.9% | 70.2s |
| 5 | Grok-4-fast | 248 | 80.8% | 91.5% | 0.70 | 13.7% | 24.5s |
| 6 | Claude-Sonnet-4-5 | 240 | 79.3% | 87.9% | 0.73 | 30.8% | 10.0s |
| 7 | Claude-Opus-4-5 | 200 | 79.3% | 89.5% | 0.73 | 18.0% | 5.8s |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 16 | GPT-4o-mini | 152 | 58.6% | 81.6% | 0.69 | **5.9%** | 7.7s |

📊 *[IMAGE: Bar chart showing Overall Accuracy with confidence intervals for top 8 models]*

---

# Slide 19: Results - Key Patterns

## 📊 **Leaderboard Analysis**

### **Pattern 1: Gemini Dominance**
> Gemini 2.5 models lead in overall accuracy (83-85%)

### **Pattern 2: Decision vs. Overall Disconnect**
> **Grok-3-mini** achieves **96.6% Decision Accuracy** despite lower overall score
> 
> → Models can make correct final decisions even with imperfect reasoning

### **Pattern 3: Hallucination-Performance Tradeoff**
```
┌──────────────────────────────────────────────────────────────┐
│  GPT-4o-mini:  Low Hallucination (5.9%)  ↔  Low Overall (58.6%)  │
│  GPT-5.1:      High Hallucination (46.2%) ↔  Mid Overall (68.2%)  │
│  Claude-Son:   High Hallucination (30.8%) ↔  Good Overall (79.3%) │
└──────────────────────────────────────────────────────────────┘
```

### **Pattern 4: Speed Champions**
> **Grok-3** (4.6s) and **Claude-Haiku** (3.7s) offer fastest responses

📊 *[IMAGE: Scatter plot of Hallucination Rate vs Overall Accuracy with model labels]*

---

# Slide 20: Results - Family Radar Charts

## 📡 **Model Family Capability Profiles**

### **Five Evaluation Dimensions:**
1. Test-Method Accuracy
2. Decision Accuracy
3. P-Value Accuracy
4. Reasoning Quality
5. Completeness

### **GPT Family:**

📊 *[IMAGE: Radar chart for GPT models (GPT-4o, GPT-5.1, GPT-4o-mini) showing:
- GPT-4o: Strong and balanced profile
- GPT-5.1/GPT-4o-mini: Uneven coverage, dips in p-value accuracy]*

> GPT-4o provides strongest, most balanced profile
> "Mini" variants show inconsistent coverage

---

# Slide 21: Results - More Family Profiles

## 📡 **Grok & Claude Families**

### **Grok Family:**

📊 *[IMAGE: Radar chart for Grok models showing:
- Consistently high, rounded footprint
- Strong on test-method selection, p-value accuracy, and decision correctness
- "Broadest capability coverage of all families"]*

### **Claude Family:**

📊 *[IMAGE: Radar chart for Claude models showing:
- Skewed toward reasoning quality and completeness
- Well-structured, interpretable arguments
- "Slightly conservative on borderline decisions"]*

---

# Slide 22: Results - Prompting Strategy Impact

## 📝 **Prompt Strategy Performance**

### **Clear Hierarchy Revealed:**

```
    PoT (Program of Thought)    ████████████████████████████████ ~95%
    Few-Shot                    ██████████████████████████████   ~90%
    Zero-Shot                   ██████████████████████████       ~80%
    Chain-of-Thought (CoT)      ████████████████████             ~65%
```

### **Surprising Finding: CoT Underperforms!**

| Strategy | Strengths | Weaknesses |
|----------|-----------|------------|
| **PoT** | Near 100% ceiling for strong models | High variance – fails for weak models |
| **Few-Shot** | Consistent improvements across all | Slightly lower ceiling than PoT |
| **Zero-Shot** | Surprisingly strong baseline | No guidance |
| **CoT** | Verbose reasoning | **Degrades performance!** Creates hallucination opportunities |

📊 *[IMAGE: Grouped bar chart showing accuracy by prompt strategy across all models - his-pmt.jpg referenced in report]*

---

# Slide 23: The CoT Paradox

## ⚠️ **Why Chain-of-Thought Fails for Statistics**

### **The Paradox:**
> CoT is gold standard for general reasoning, but **worst** for hypothesis testing

### **Explanation:**

```
┌─────────────────────────────────────────────────────────────────┐
│  Chain of Thought in Statistics:                                │
│                                                                 │
│  "Let me think step by step..."                                 │
│       │                                                         │
│       ├──► Opportunity to miscalculate                          │
│       ├──► Opportunity to confuse test types                    │
│       ├──► Opportunity to hallucinate intermediate values       │
│       ├──► Opportunity to lose track of procedure               │
│       │                                                         │
│       ▼                                                         │
│  Accumulated errors compound!                                   │
└─────────────────────────────────────────────────────────────────┘
```

### **Recommendation:**
> For statistical tasks, **less verbose reasoning is better** unless you're providing examples (Few-Shot) or code (PoT)

📊 *[IMAGE: Visual showing a CoT reasoning chain with error accumulation points marked with red X symbols]*

---

# Slide 24: Results - Test Type Performance

## 📋 **Performance by T-Test Variation**

### **The Paired T-Test Problem:**

```
                    One-Sample    Two-Sample    Paired
                    ──────────    ──────────    ──────
Gemini-2.5-Flash      92.9%         93.3%       59.5%  ⚠️
Gemini-2.5-Pro        91.2%         90.1%       62.3%  ⚠️
Grok-3                87.1%         89.4%       74.7%  ←Best
Claude-Opus-4-5       85.3%         86.8%       61.2%  ⚠️
GPT-4o                82.6%         84.1%       55.9%  ⚠️
```

### **The Pattern:**
> **20-30% accuracy drop** on paired data across nearly ALL models!

### **Why?**
- Models struggle with **dependency reasoning**
- Failing to apply proper **variance corrections**
- Confusion between paired vs. independent samples

📊 *[IMAGE: Heatmap showing test type performance - hm-tt.jpg referenced in report]*

---

# Slide 25: Results - P-Value Calibration

## 📈 **P-Value Prediction Accuracy**

### **Correlation: Predicted vs Ground Truth P-Values**

| Model | P-Value Correlation |
|-------|:------------------:|
| **Grok-4-1-fast-reasoning** | **0.9991** |
| **Claude-Opus-4-5** | **0.9973** |
| Gemini-2.5-Pro | 0.9845 |
| Grok-3 | 0.9712 |
| GPT-4o | 0.9456 |

### **Key Insight:**
> Top models achieve **near-perfect P-value calibration**
> 
> They understand "significance" semantically, even if arithmetic is imperfect

📊 *[IMAGE: Scatter plot of Predicted vs Ground Truth P-values with regression line - hm-p-g.jpg reference]*

---

# Slide 26: Results - Test Statistic Problem

## ⚠️ **The Test Statistic Disconnect**

### **Correlation: Predicted vs Ground Truth Test Statistics**

| Model | P-Value Corr. | Test Stat Corr. | Gap |
|-------|:-------------:|:---------------:|:---:|
| Grok-4-1-f-r | 0.999 | 0.71 | **0.29** |
| Claude-Opus | 0.997 | 0.68 | **0.32** |
| GPT-4o | 0.946 | 0.52 | **0.43** |
| Some models | 0.95+ | **-0.21** | **>1.0** |

### **The "Outcome-Process Dissociation":**

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ✅ Correct P-value    BUT    ❌ Wrong Test Statistic          │
│                                                                 │
│   This suggests:                                                │
│   • Pattern matching on problem text                            │
│   • NOT executing logical mathematical derivation               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

📊 *[IMAGE: Dual scatter plots side by side - one for P-value correlation (tight clustering) and one for test statistic (scattered) - cor-ts.jpg reference]*

---

# Slide 27: Results - Confusion Matrix

## 🎯 **Decision Bias Analysis**

### **Aggregate Confusion Matrix:**

```
                          Predicted
                    ┌──────────────────────────┐
                    │  Reject H0 │ Fail Reject │
         ┌──────────┼────────────┼─────────────┤
Actual   │ Reject   │    2022    │     26      │  (Sensitivity: 98.7%)
         │          │    (TP)    │    (FN)     │
         ├──────────┼────────────┼─────────────┤
         │ Fail     │    258     │    983      │  (Specificity: 79.2%)
         │ Reject   │    (FP)    │    (TN)     │
         └──────────┴────────────┴─────────────┘
```

### **Critical Finding: Liberal Bias**

> **10:1 ratio** of False Positives to False Negatives!
> 
> (258 FP vs 26 FN)

### **Implication:**
> ⚠️ LLMs are **"trigger-happy"** toward rejecting H0
> 
> → High risk of **Type I errors** (claiming false discoveries)

📊 *[IMAGE: Visual confusion matrix with heatmap coloring - cm.jpg reference]*

---

# Slide 28: Results - Error Type Analysis

## ⚖️ **Type I vs Type II Errors**

### **The Asymmetry:**

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Type I Error (False Positive)     Type II Error (False Neg)   │
│   "False Discovery"                 "Missed Discovery"          │
│                                                                 │
│         258 cases                        26 cases               │
│          ██████████                        █                    │
│          ██████████                                             │
│                                                                 │
│           10x more common!                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### **Scientific Impact:**
| Error Type | LLM Behavior | Risk |
|------------|--------------|------|
| **Type I** | Frequently claims significance where none exists | False discoveries, wasted resources |
| **Type II** | Rarely misses a true effect | Less problematic |

### **Recommendation:**
> ⚠️ Treat LLM "Reject H0" conclusions with **skepticism**
> 
> ✅ "Fail to Reject" conclusions are **highly trustworthy**

📊 *[IMAGE: Asymmetric bar chart or pie chart showing 10:1 error ratio with scientific impact icons]*

---

# Slide 29: Results - Reasoning Quality

## 💭 **Reasoning Quality Distribution**

### **"High Ceiling, Unstable Floor" Dynamic:**

📊 *[IMAGE: Box-and-whisker plot for all models showing:
- Median scores cluster 0.70-0.80
- Significant outliers below 0.30 for all models
- Reference: bw-r.jpg]*

### **Key Observations:**

| Model Type | Median | Variance | Outliers |
|------------|:------:|:--------:|:--------:|
| Gemini 2.5 Pro | 0.78 | Low | Few |
| GPT-4o | 0.75 | Medium | Some |
| Grok-3-Mini | 0.73 | **High** | Many |
| GPT-4o-mini | 0.69 | **High** | Many |

### **Recommendation:**
> Avoid "Mini/Fast" variants for complex statistical reasoning
> 
> **Human-in-the-loop verification essential** for critical conclusions

---

# Slide 30: Results - Latency Analysis

## ⏱️ **Response Time Comparison**

### **Latency Distribution:**

```
claude-haiku-4-5    ████ 3.7s                     ← Fastest
grok-3              █████ 4.6s
gpt-4o              █████ 4.2s
claude-opus-4-5     ███████ 5.8s
gpt-4o-mini         █████████ 7.7s
deepseek-chat       ██████████ 9.0s
gpt-5-mini          ██████████████████████ 22.7s
gemini-2.5-pro      ██████████████████████ 22.7s
grok-4-fast         ███████████████████████████ 24.5s
grok-4-1-f-r        ███████████████████████████████████████████████████████████████████████ 70.2s  ← 14x slowest
```

### **Speed vs. Accuracy Tradeoff:**
| Model | Latency | Overall Acc. | Verdict |
|-------|:-------:|:------------:|---------|
| Grok-3 | 4.6s | 82.4% | ⭐ Best value |
| GPT-4o | 4.2s | 74.2% | Good balance |
| Grok-4-1-f-r | 70.2s | 81.0% | Not worth 14x wait |

📊 *[IMAGE: Horizontal bar chart of latencies - his-l.jpg reference]*

---

# Slide 31: Key Insights Summary

## 💡 **Critical Insights**

### **1. The Outcome-Process Dissociation**
> Models get correct P-values but wrong test statistics
> → **Pattern matching, not mathematical reasoning**

### **2. Dependency Blindness**
> Systematic failure on Paired T-Tests
> → **Cannot reason about data dependencies**

### **3. The Liberal Bias**
> 10:1 False Positive to False Negative ratio
> → **Trigger-happy toward "significant" results**

### **4. The Prompting Paradox**
> CoT **degrades** performance; PoT is superior
> → **Code execution > verbose reasoning**

### **5. Exceptional Sensitivity**
> 98.7% recall for true effects
> → **Excellent as screening tools**

📊 *[IMAGE: Visual summary with 5 key insight icons and brief descriptions]*

---

# Slide 32: Successes of Current LLMs

## ✅ **What LLMs Do Well**

### **Success 1: Exceptional Sensitivity (Recall)**

```
True Positives:  2,022
False Negatives:    26
───────────────────────
Recall = 98.7%
```

> LLMs rarely miss a true effect → Excellent for **first-pass screening**

### **Success 2: High P-Value Fidelity**

| Model | P-Value Correlation |
|-------|:------------------:|
| Grok-4-1-fast-reasoning | **0.9991** |
| Claude-Opus-4-5 | **0.9973** |

> Near-perfect understanding of "statistical significance"

### **Practical Implication:**
> ✅ Use LLMs to **screen for potential discoveries**
> 
> ✅ Trust "not significant" conclusions
> 
> ❌ Verify "significant" conclusions manually

📊 *[IMAGE: Success metrics visualization with checkmarks and high scores]*

---

# Slide 33: Failure Modes

## ❌ **Critical Failure Modes**

### **Failure 1: Dependency Blindness**
```
Paired T-Test Accuracy:
┌─────────────────────────────────────────┐
│  Most Models:  55-65%   ⚠️ Near random! │
│  Best (Grok-3): 74.7%   Still weak     │
└─────────────────────────────────────────┘
```

### **Failure 2: Process Errors**
> Correct answers via wrong methods
> 
> → Cannot trust intermediate reasoning

### **Failure 3: Liberal Bias**
```
In scientific context:
  False Discoveries → Wasted resources
  False Discoveries → Replication crisis
  False Discoveries → Eroded trust
```

### **Failure 4: Outlier Reasoning Failures**
> Even best models occasionally produce catastrophically wrong reasoning (scores < 0.30)

📊 *[IMAGE: Warning/alert style visualization with failure modes listed]*

---

# Slide 34: Recommendations

## 📋 **Practical Recommendations**

### **For Researchers Using LLMs for Statistics:**

| Task | Recommended Approach |
|------|---------------------|
| **Prompting** | Use **Program-of-Thought** for strong models, **Few-Shot** as fallback |
| **Model Selection** | **Gemini 2.5** or **Grok-3/4** for best accuracy |
| **Paired Data** | Use **Grok-3** (best on paired) or verify manually |
| **Fast Results** | **Grok-3** (4.6s) or **Claude-Haiku** (3.7s) |
| **"Reject H0" Results** | Always **verify manually** (high FP rate) |
| **"Fail to Reject" Results** | Generally **trustworthy** (low FN rate) |

### **Do NOT:**
- ❌ Use Chain-of-Thought for statistical tasks
- ❌ Trust LLMs as autonomous statisticians
- ❌ Deploy "Mini" variants for complex reasoning

📊 *[IMAGE: Checklist style visualization with do's and don'ts]*

---

# Slide 35: Future Work - Overview

## 🔮 **Future Research Directions**

### **Five Key Areas:**

```
┌─────────────────────────────────────────────────────────────────┐
│  1. STATISTICAL AGENT                                           │
│     Never calculate in text – always execute code               │
├─────────────────────────────────────────────────────────────────┤
│  2. PROCESS REWARD MODEL                                        │
│     Reward correct intermediate steps, not just final answer    │
├─────────────────────────────────────────────────────────────────┤
│  3. EXPANDED TEST COVERAGE                                      │
│     ANOVA, regression, chi-square, non-parametric tests         │
├─────────────────────────────────────────────────────────────────┤
│  4. AGENTIC WORKFLOWS                                           │
│     Iterative data requests, diagnostics, pilot simulations     │
├─────────────────────────────────────────────────────────────────┤
│  5. REFINED HALLUCINATION METRICS                               │
│     Distinguish verbosity vs logical errors vs misleading claims│
└─────────────────────────────────────────────────────────────────┘
```

📊 *[IMAGE: Future roadmap diagram with 5 connected nodes]*

---

# Slide 36: Future Work - Statistical Agent

## 🤖 **Future Direction 1: Statistical Agent**

### **The Vision:**
> A specialized agent that **never calculates in text**

### **Architecture:**

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Natural Language Problem                          │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     LLM: Problem Understanding                        │
│                     "This is a paired t-test with..."                │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     Code Generation                                   │
│     from scipy import stats                                          │
│     t_stat, p_value = stats.ttest_rel(sample1, sample2)              │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     Sandbox Execution                                 │
│                     Guaranteed correct computation                    │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     LLM: Result Interpretation                        │
│                     "The p-value of 0.023 indicates..."              │
└──────────────────────────────────────────────────────────────────────┘
```

📊 *[IMAGE: Agent architecture diagram with sandbox and code execution components]*

---

# Slide 37: Future Work - Process Reward Model

## 🎯 **Future Direction 2: Process Reward Model**

### **Current Problem:**
> Models rewarded only for final "Reject/Fail to Reject" decision
> 
> → Can be right for wrong reasons

### **Proposed Solution:**
> Train reward model on **intermediate statistical derivations**

### **Reward Points:**

| Step | Reward |
|------|--------|
| ✅ Correct degrees of freedom identification | +0.2 |
| ✅ Correct standard error calculation | +0.2 |
| ✅ Correct test statistic formula | +0.2 |
| ✅ Correct p-value computation | +0.2 |
| ✅ Correct final decision | +0.2 |

### **Expected Benefit:**
> Models learn the **process**, not just pattern-matching to outcomes

📊 *[IMAGE: Reward model training diagram showing step-wise rewards for statistical derivation]*

---

# Slide 38: Conclusion

## 📌 **Conclusion**

### **Key Takeaway:**

> While LLMs have achieved **near-human proficiency** in statistical decision-making, they fundamentally **lack the robust procedural reasoning** required for reliable automated science.

### **The Path Forward:**

```
Current State                          Future State
─────────────                          ────────────
LLMs as "Statistical Assistants"  →    LLMs as "Statistical Agents"
  - Screening tools                      - Autonomous computation
  - Human verification required          - Neuro-symbolic execution
  - Pattern matching                     - True procedural reasoning
```

### **Bottom Line:**
> 🔬 **Program of Thought + Code Execution** is the only reliable path to near-100% accuracy
> 
> 🧠 Current models are excellent **sensitivity tools** but not trusted **autonomous statisticians**

📊 *[IMAGE: Evolution diagram from current assistants to future agents]*

---

# Slide 39: Thank You & Questions

## 🙏 **Thank You!**

### **Contact:**
**Alfred K. Adzika**  
📧 aa832423@ohio.edu  
🏫 School of Electrical Engineering and Computer Science  
Ohio University

### **Resources:**
- 📊 Dashboard: `streamlit run dashboard/app.py`
- 💻 Benchmark: `python ht.py --mode full`
- 📁 Results: `results/*.json`

### **Questions?**

📊 *[IMAGE: Contact information with QR code linking to project repository]*

---

# Slide 40: Appendix - Code Sample: Data Generation

## 📝 **Appendix A: Data Generation Code**

```python
class DataGenerator:
    """Generate synthetic data for hypothesis testing"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)  # Reproducibility
    
    def generate_one_sample_t_test(self, sample_size: int, 
                                   true_mean: float = 10,
                                   std: float = 2,
                                   null_mean: float = 10):
        """Generate data for one-sample t-test"""
        sample = self.generate_normal(true_mean, std, sample_size)
        
        return {
            "test_type": "one_sample_t_test",
            "sample1": sample,
            "population_mean": null_mean,
            "true_effect": true_mean - null_mean,
            "metadata": {
                "sample_size": sample_size,
                "true_mean": true_mean,
                "std": std,
                "null_mean": null_mean
            }
        }
```

---

# Slide 41: Appendix - Code Sample: Ground Truth

## 📝 **Appendix B: Ground Truth Computation**

```python
class StatisticalEngine:
    """Compute ground truth statistical test results"""
    
    @staticmethod
    def compute_one_sample_t_test(sample: np.ndarray, 
                                  population_mean: float):
        n = len(sample)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        
        # Test statistic
        t_stat = (sample_mean - population_mean) / (sample_std / np.sqrt(n))
        
        # P-value via SciPy
        p_value = 2 * stats.t.sf(np.abs(t_stat), df=n-1)
        
        # Decision at α = 0.05
        decision = "reject_H0" if p_value < 0.05 else "fail_to_reject_H0"
        
        return {
            "test_method": "one_sample_t_test",
            "test_statistic": float(t_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": n - 1,
            "decision": decision,
            ...
        }
```

---

# Slide 42: Appendix - Models Evaluated

## 📝 **Appendix C: Complete Model List**

### **16 Models Across 5 Providers:**

| Provider | Models | Count |
|----------|--------|:-----:|
| **OpenAI** | GPT-5.1, GPT-5-mini, GPT-4, GPT-4o, GPT-4o-mini | 5 |
| **Anthropic** | Claude-Opus-4-5, Claude-Opus-4-1, Claude-Sonnet-4-5, Claude-Haiku-4-5 | 4 |
| **Google** | Gemini-2.5-Pro, Gemini-2.5-Flash | 2 |
| **xAI (Grok)** | Grok-3, Grok-4.1-thinking, Grok-3-mini, Grok-4-1-fast-reasoning | 4 |
| **DeepSeek** | DeepSeek-Chat | 1 |

### **Total Evaluations:** ~3,300 individual test runs

---

# Slide 43: Appendix - Prompt Examples

## 📝 **Appendix D: Prompt Strategy Examples**

### **Program-of-Thought (PoT) Prompt:**
```text
Write Python code to solve this hypothesis testing problem:

Data:
Sample 1: [12.3, 14.1, 13.5, ...] (n=50)
Hypothesized population mean: 12.0

Requirements:
1. Import scipy.stats
2. Compute test statistic and p-value
3. Make decision at α = 0.05
4. Print results in format:
   RESULT: test=one_sample_t_test, t_stat=X.XX, p_value=X.XXXX, decision=reject_H0/fail_to_reject_H0
```

### **Why PoT Works:**
- Forces computation via reliable library (SciPy)
- Eliminates mental arithmetic errors
- Structured output format for parsing

---

# Slide 44: Appendix - Dashboard Overview

## 📝 **Appendix E: Interactive Dashboard**

### **Dashboard Components:**

| Tab | Content |
|-----|---------|
| **Leaderboard** | Model rankings with confidence intervals |
| **Radar Charts** | Family capability profiles |
| **Heatmaps** | Performance by test type × prompt strategy |
| **Correlation Plots** | Predicted vs ground truth analysis |
| **Error Analysis** | Confusion matrix, Type I/II breakdown |
| **Qualitative** | Side-by-side prompt/response/truth viewer |

### **Launch Command:**
```bash
streamlit run dashboard/app.py
```

📊 *[IMAGE: Screenshot montage of dashboard tabs showing leaderboard, radar charts, and heatmaps]*

---

# END OF PRESENTATION

## 📊 Total: 44 Slides

### **Image Placeholders Summary:**
Throughout this presentation, the following images should be created or sourced:

1. Executive summary infographic
2. Presentation roadmap flowchart
3. AI in high-stakes domains illustration
4. Scientist AI framework diagram
5. ImageNet vs Our Benchmark comparison
6. Research gap Venn diagram
7. Target/bullseye for objectives
8. System architecture diagram (ht-bench-art.png)
9. Synthetic data generation visualization
10. SciPy ground truth computation diagram
11. Prompt strategy icons
12. LLM provider integration diagram
13. Response parsing flowchart
14. Metric gauge charts
15. Execution pipeline diagram
16. Benchmark modes visualization
17. Leaderboard bar chart
18. Hallucination vs Accuracy scatter plot
19. GPT family radar chart (rad-op.jpg)
20. Grok family radar chart (rad-gr.jpg)
21. Claude family radar chart (rad-cl.jpg)
22. Google/DeepSeek radar charts (rad-geds.jpg)
23. Prompt strategy histogram (his-pmt.jpg)
24. CoT error accumulation visualization
25. Test type heatmap (hm-tt.jpg)
26. P-value correlation scatter plot (hm-p-g.jpg)
27. Test statistic correlation (cor-ts.jpg)
28. Confusion matrix heatmap (cm.jpg)
29. Error type asymmetry visualization
30. Reasoning quality box plot (bw-r.jpg)
31. Latency bar chart (his-l.jpg)
32. Key insights summary icons
33. Success metrics visualization
34. Failure modes alerts
35. Recommendations checklist
36. Future roadmap diagram
37. Statistical agent architecture
38. Process reward model diagram
39. Evolution diagram (assistant to agent)
40. Contact/QR code slide
41. Dashboard screenshot montage
