"""Extract accurate metrics from results files for report table"""
import json
from pathlib import Path
from collections import defaultdict
import re

RESULTS_DIR = Path("results")
EXCLUDED_MODELS = ["claude-3-haiku", "2024"]
GEMINI_CUTOFF_DATE = 20251214

def shorten_model_name(name: str) -> str:
    """Shorten model names for display"""
    name = re.sub(r'-\d{8}$', '', name)
    replacements = {
        'fast-reasoning': 'f-r',
        'thinking': 't',
        'experimental': 'exp',
        'preview': 'prev',
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name

def extract_date_from_filename(filename: str) -> int:
    """Extract date as int from filename like results_20251215_080520.json"""
    match = re.search(r'results_(\d{8})_', filename)
    if match:
        return int(match.group(1))
    return 0

def is_valid_result(result: dict) -> bool:
    """Check if result has complete response data"""
    # Check for empty response
    raw_response = result.get("raw_response", "") or result.get("response", "")
    if not raw_response or not raw_response.strip():
        return False
    
    # Check for incomplete parsing
    parsed = result.get("parsed_response", {})
    if parsed:
        p_value = parsed.get("p_value")
        test_stat = parsed.get("test_statistic")
        decision = parsed.get("decision")
        if p_value is None and test_stat is None and decision is None:
            return False
    
    return True

def extract_reasoning_score(reasoning_quality):
    """Extract numeric score from reasoning_quality field"""
    if reasoning_quality is None:
        return None
    if isinstance(reasoning_quality, (int, float)):
        return float(reasoning_quality)
    if isinstance(reasoning_quality, dict):
        # Try to get 'score' or 'overall_score' key
        if 'score' in reasoning_quality:
            return float(reasoning_quality['score'])
        if 'overall_score' in reasoning_quality:
            return float(reasoning_quality['overall_score'])
        # Calculate from components if available
        components = ['hypothesis_clarity', 'test_justification', 'assumption_checking', 
                      'correct_interpretation', 'statistical_rigor']
        scores = [reasoning_quality.get(c, 0) for c in components if c in reasoning_quality]
        if scores:
            return sum(scores) / len(scores)
    return None

def load_all_results():
    """Load and filter all results"""
    all_results = []
    
    for file in RESULTS_DIR.glob("results_*.json"):
        file_date = extract_date_from_filename(file.name)
        
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                
            for result in data:
                model = result.get("model", "")
                
                # Skip excluded models
                if any(exc in model for exc in EXCLUDED_MODELS):
                    continue
                
                # Skip Gemini results before cutoff
                if "gemini" in model.lower() and file_date < GEMINI_CUTOFF_DATE:
                    continue
                
                # Skip invalid/incomplete results
                if not is_valid_result(result):
                    continue
                
                all_results.append(result)
                
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return all_results

def compute_metrics(results):
    """Compute aggregated metrics per model"""
    model_data = defaultdict(lambda: {
        'count': 0,
        'overall_acc': [],
        'decision_acc': [],
        'p_value_errors': [],
        'reasoning_quality': [],
        'latency': [],
        'hallucination': []
    })
    
    for r in results:
        model = r.get("model", "unknown")
        metrics = r.get("evaluation", {})
        
        model_data[model]['count'] += 1
        
        # Overall accuracy
        if 'overall_accuracy' in metrics:
            model_data[model]['overall_acc'].append(metrics['overall_accuracy'])
        
        # Decision accuracy
        if 'decision_correct' in metrics:
            model_data[model]['decision_acc'].append(1.0 if metrics['decision_correct'] else 0.0)
        
        # P-value MAE
        if 'p_value_error' in metrics and metrics['p_value_error'] is not None:
            model_data[model]['p_value_errors'].append(abs(metrics['p_value_error']))
        
        # Reasoning quality - handle dict or numeric
        if 'reasoning_quality' in metrics:
            score = extract_reasoning_score(metrics['reasoning_quality'])
            if score is not None:
                model_data[model]['reasoning_quality'].append(score)
        
        # Latency
        if 'latency_seconds' in r and r['latency_seconds'] is not None:
            model_data[model]['latency'].append(r['latency_seconds'])
        
        # Hallucination
        if 'hallucination_flags' in metrics:
            flags = metrics['hallucination_flags']
            has_hallucination = any(flags.values()) if isinstance(flags, dict) else False
            model_data[model]['hallucination'].append(1.0 if has_hallucination else 0.0)
    
    # Compute averages
    summary = []
    for model, data in model_data.items():
        if data['count'] == 0:
            continue
            
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0.0
        
        summary.append({
            'model': model,
            'short_name': shorten_model_name(model),
            'n': data['count'],
            'overall_acc': avg(data['overall_acc']) * 100,
            'decision_acc': avg(data['decision_acc']) * 100,
            'p_value_mae': avg(data['p_value_errors']),
            'reasoning_quality': avg(data['reasoning_quality']),
            'latency': avg(data['latency']),
            'hallucination_rate': avg(data['hallucination']) * 100
        })
    
    # Sort by overall accuracy descending
    summary.sort(key=lambda x: x['overall_acc'], reverse=True)
    return summary

def print_latex_table(summary):
    """Print LaTeX table format"""
    print("\n% LaTeX Table for Report")
    print("\\begin{table}[htbp]")
    print("\\caption{Model Performance Comparison (Dashboard Summary)}")
    print("\\begin{center}")
    print("\\small")
    print("\\begin{tabular}{|l|c|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Model} & \\textbf{N} & \\textbf{Overall} & \\textbf{Decision} & \\textbf{P-Value} & \\textbf{Reasoning} & \\textbf{Latency} \\\\")
    print("      &  & \\textbf{Acc.}    & \\textbf{Acc.}     & \\textbf{MAE}     & \\textbf{Quality}   & \\textbf{(s)} \\\\")
    print("\\hline")
    
    for m in summary:
        print(f"{m['short_name']:<18} & {m['n']:>3} & {m['overall_acc']:.1f}\\% & {m['decision_acc']:.1f}\\% & {m['p_value_mae']:.4f} & {m['reasoning_quality']:.2f} & {m['latency']:.1f} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\label{tab1}")
    print("\\end{center}")
    print("\\end{table}")

def main():
    print("Loading results...")
    results = load_all_results()
    print(f"Loaded {len(results)} valid results")
    
    print("\nComputing metrics...")
    summary = compute_metrics(results)
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY (sorted by overall accuracy)")
    print("="*80)
    print(f"{'Model':<25} {'N':>5} {'Overall':>8} {'Decision':>9} {'P-MAE':>8} {'Reason':>8} {'Latency':>8}")
    print("-"*80)
    
    for m in summary:
        print(f"{m['short_name']:<25} {m['n']:>5} {m['overall_acc']:>7.1f}% {m['decision_acc']:>8.1f}% {m['p_value_mae']:>8.4f} {m['reasoning_quality']:>8.2f} {m['latency']:>7.1f}s")
    
    print_latex_table(summary)

if __name__ == "__main__":
    main()