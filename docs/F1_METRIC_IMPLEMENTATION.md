# F1 Score Metric Implementation

## Overview

Added F1 score, precision, and recall metrics to measure the reasoning quality of LLMs on hypothesis testing tasks. This addresses the class imbalance problem and provides better insight into Type I and Type II error rates.

## Motivation

As stated in "Measuring mathematical problem solving with the MATH dataset" (Hendrycks et al.), F1 scores are valuable for reasoning tasks because:

1. **Handles Class Imbalance**: When scenarios heavily favor "reject H₀" or "fail to reject H₀", accuracy can be misleading
2. **Balances Error Types**: F1 penalizes both false positives (Type I errors - incorrect rejections) and false negatives (Type II errors - missed rejections)
3. **Reasoning Quality**: Measures the model's ability to make correct statistical conclusions, not just compute p-values

## Implementation Details

### Key Files Modified

1. **[evaluator.py](evaluator.py)** - Added `calculate_metrics()` function
   - Computes precision, recall, and F1 score
   - Includes confusion matrix (TP, FP, FN, TN)
   - Works with nested result structure from benchmark

2. **[ht.py](ht.py)** - Integrated metrics into benchmark flow
   - Added metrics calculation in `generate_summary()`
   - Updated `print_summary()` to display F1 scores and confusion matrix
   - Imports `calculate_metrics` from evaluator

3. **[test_f1_metric.py](test_f1_metric.py)** - Comprehensive unit tests
   - Tests perfect score, false positives, false negatives
   - Tests balanced errors and edge cases
   - Validates confusion matrix calculations

## Metrics Explained

### Confusion Matrix
- **True Positives (TP)**: Correctly rejected H₀ when should reject
- **False Positives (FP)**: Incorrectly rejected H₀ (Type I error)
- **False Negatives (FN)**: Failed to reject H₀ when should reject (Type II error)
- **True Negatives (TN)**: Correctly failed to reject H₀

### Formulas
```
Precision = TP / (TP + FP)  # How many rejections were correct
Recall = TP / (TP + FN)     # How many true effects were detected
F1 = 2 × (Precision × Recall) / (Precision + Recall)  # Harmonic mean
```

## Testing

### Unit Tests
```bash
python test_f1_metric.py
```

Runs 5 test scenarios:
- Perfect predictions (F1 = 1.0)
- Over-rejection (high recall, low precision)
- Under-rejection (high precision, low recall)
- Balanced errors
- Edge case: no rejections

### Real Benchmark
```bash
# Quick test with one model
python ht.py --mode custom --models openai/gpt-4o-mini --tests one_sample_t_test --scenarios 2

# Full benchmark
python ht.py --mode full
```

## Example Output

```
================================================================================
BENCHMARK SUMMARY
================================================================================

Total Evaluations: 16

--- Overall Reasoning Metrics ---
  Accuracy: 100.00%
  P-value Accuracy: 93.75%
  Test Selection Accuracy: 93.75%

  Conclusion Quality (F1 Metrics):
    Precision: 1.000
    Recall: 1.000
    F1 Score: 1.000

  Confusion Matrix:
    True Positives:  8 (correctly rejected H0)
    False Positives: 0 (incorrectly rejected H0)
    False Negatives: 0 (missed rejecting H0)
    True Negatives:  8 (correctly failed to reject H0)

--- Performance by Model ---
...
```

## Dashboard Integration (Future Work)

To display F1 scores in the Streamlit dashboard (`dashboard/app.py`):

```python
def display_f1_metrics(results_df):
    """Display F1 score metrics in dashboard"""
    st.subheader("📊 F1 Score Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Precision", f"{precision:.3f}")
    with col2:
        st.metric("Recall", f"{recall:.3f}")
    with col3:
        st.metric("F1 Score", f"{f1_score:.3f}")
    
    # Confusion matrix heatmap
    fig = px.imshow([[tp, fp], [fn, tn]], 
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Reject', 'Not Reject'],
                    y=['Reject', 'Not Reject'])
    st.plotly_chart(fig)
```

## Benefits

1. **Better Error Analysis**: See which models over-reject (Type I errors) vs under-reject (Type II errors)
2. **Prompt Type Comparison**: Compare F1 across zero-shot, CoT, PoT, few-shot
3. **Literature Alignment**: F1 is standard for reasoning benchmarks (MATH, GSM8K)
4. **Actionable Insights**: Confusion matrix shows exactly where models fail

## References

- Hendrycks, D., et al. (2021). "Measuring mathematical problem solving with the MATH dataset." *NeurIPS*.
- Standard practice in ML evaluation for imbalanced classification tasks
