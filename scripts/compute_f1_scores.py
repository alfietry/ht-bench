"""
Compute F1 scores for all existing benchmark results
Saves enhanced results with F1 metrics
"""
import json
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluator import calculate_metrics
from src import config


def load_result_file(filepath: Path) -> List[Dict]:
    """Load a single result file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'results' in data:
            return data['results']
        else:
            return []
    except Exception as e:
        print(f"Error loading {filepath.name}: {e}")
        return []


def compute_f1_for_results(results: List[Dict]) -> Dict:
    """Compute F1 metrics from results"""
    if not results:
        return {
            "accuracy": 0.0,
            "p_value_accuracy": 0.0,
            "test_selection_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "total_scenarios": 0,
            "confusion_matrix": {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "true_negatives": 0
            }
        }
    
    return calculate_metrics(results)


def compute_f1_by_model(results: List[Dict]) -> Dict[str, Dict]:
    """Compute F1 scores grouped by model"""
    model_results = {}
    
    for result in results:
        model = result.get("model", "unknown")
        if model not in model_results:
            model_results[model] = []
        model_results[model].append(result)
    
    model_metrics = {}
    for model, model_res in model_results.items():
        model_metrics[model] = compute_f1_for_results(model_res)
    
    return model_metrics


def compute_f1_by_prompt_type(results: List[Dict]) -> Dict[str, Dict]:
    """Compute F1 scores grouped by prompt type"""
    prompt_results = {}
    
    for result in results:
        prompt_type = result.get("prompt_type", "unknown")
        if prompt_type not in prompt_results:
            prompt_results[prompt_type] = []
        prompt_results[prompt_type].append(result)
    
    prompt_metrics = {}
    for prompt_type, prompt_res in prompt_results.items():
        prompt_metrics[prompt_type] = compute_f1_for_results(prompt_res)
    
    return prompt_metrics


def compute_f1_by_test_type(results: List[Dict]) -> Dict[str, Dict]:
    """Compute F1 scores grouped by test type"""
    test_results = {}
    
    for result in results:
        test_type = result.get("input_data", {}).get("test_type", "unknown")
        if test_type not in test_results:
            test_results[test_type] = []
        test_results[test_type].append(result)
    
    test_metrics = {}
    for test_type, test_res in test_results.items():
        test_metrics[test_type] = compute_f1_for_results(test_res)
    
    return test_metrics


def process_all_results():
    """Process all result files and compute F1 scores"""
    results_dir = config.RESULTS_DIR
    
    # Get all result files
    result_files = sorted(results_dir.glob("results_*.json"))
    
    if not result_files:
        print("No result files found!")
        return
    
    print(f"Found {len(result_files)} result files")
    print("=" * 80)
    
    all_results = []
    file_metrics = {}
    
    for result_file in result_files:
        print(f"\nProcessing: {result_file.name}")
        results = load_result_file(result_file)
        
        if not results:
            print(f"  [WARNING] No results in file")
            continue
        
        # Compute metrics for this file
        overall_metrics = compute_f1_for_results(results)
        model_metrics = compute_f1_by_model(results)
        prompt_metrics = compute_f1_by_prompt_type(results)
        test_metrics = compute_f1_by_test_type(results)
        
        file_metrics[result_file.name] = {
            "filepath": str(result_file),
            "timestamp": result_file.stem.replace("results_", ""),
            "total_evaluations": len(results),
            "overall": overall_metrics,
            "by_model": model_metrics,
            "by_prompt_type": prompt_metrics,
            "by_test_type": test_metrics
        }
        
        print(f"  [OK] {len(results)} evaluations")
        print(f"    Overall F1: {overall_metrics['f1_score']:.3f}")
        print(f"    Precision: {overall_metrics['precision']:.3f}, Recall: {overall_metrics['recall']:.3f}")
        
        # Add to all results for combined metrics
        all_results.extend(results)
    
    # Compute combined metrics across all files
    print("\n" + "=" * 80)
    print("COMBINED METRICS (All Result Files)")
    print("=" * 80)
    
    combined_metrics = {
        "total_evaluations": len(all_results),
        "total_files": len(result_files),
        "overall": compute_f1_for_results(all_results),
        "by_model": compute_f1_by_model(all_results),
        "by_prompt_type": compute_f1_by_prompt_type(all_results),
        "by_test_type": compute_f1_by_test_type(all_results),
        "per_file": file_metrics
    }
    
    overall = combined_metrics["overall"]
    print(f"\nTotal Evaluations: {len(all_results)}")
    print(f"Overall F1 Score: {overall['f1_score']:.3f}")
    print(f"Precision: {overall['precision']:.3f}")
    print(f"Recall: {overall['recall']:.3f}")
    print(f"Accuracy: {overall['accuracy']:.2%}")
    
    print("\n--- F1 by Model ---")
    for model, metrics in sorted(combined_metrics["by_model"].items(), 
                                 key=lambda x: x[1]["f1_score"], reverse=True):
        print(f"  {model:30s} F1: {metrics['f1_score']:.3f}  "
              f"P: {metrics['precision']:.3f}  R: {metrics['recall']:.3f}  "
              f"(n={metrics['total_scenarios']})")
    
    print("\n--- F1 by Prompt Type ---")
    for prompt_type, metrics in sorted(combined_metrics["by_prompt_type"].items(),
                                       key=lambda x: x[1]["f1_score"], reverse=True):
        print(f"  {prompt_type:25s} F1: {metrics['f1_score']:.3f}  "
              f"P: {metrics['precision']:.3f}  R: {metrics['recall']:.3f}  "
              f"(n={metrics['total_scenarios']})")
    
    # Save combined metrics
    output_file = results_dir / f"f1_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(combined_metrics, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("[SUCCESS] F1 metrics saved to: " + output_file.name)
    print("=" * 80)
    
    return combined_metrics


if __name__ == "__main__":
    print("=" * 80)
    print("F1 SCORE COMPUTATION FOR ALL RESULTS")
    print("=" * 80)
    
    try:
        metrics = process_all_results()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
