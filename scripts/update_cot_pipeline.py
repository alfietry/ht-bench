"""
Complete pipeline to re-parse and re-evaluate CoT responses after normalization.
This script:
1. Re-parses all responses using the updated parser (handles < and > symbols)
2. Re-evaluates using the newly parsed data
3. Updates result files for dashboard consumption
"""
import json
from pathlib import Path
from datetime import datetime
import shutil
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.response_parser import ResponseParser
from src.evaluator import EvaluationMetrics
from src.config import RESULTS_DIR, EVALUATION

def create_backup_dir():
    """Create backup directory for original files"""
    backup_dir = Path(RESULTS_DIR) / "backups"
    backup_dir.mkdir(exist_ok=True)
    return backup_dir

def backup_file(file_path: Path, backup_dir: Path):
    """Create timestamped backup of file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
    shutil.copy2(file_path, backup_path)
    return backup_path

def reparse_and_evaluate_result(result: dict, parser: ResponseParser, metrics: EvaluationMetrics) -> dict:
    """Re-parse response and re-evaluate against ground truth"""
    # Get the response text (normalized or original)
    response_text = result.get('raw_response') or result.get('response', '')
    
    if not response_text:
        return result
    
    # Parse with improved parser
    parsed = parser.parse(response_text)
    
    # Update parsed fields in result
    if parsed.hypotheses:
        result['hypotheses'] = {
            'H0': parsed.hypotheses.H0,
            'H1': parsed.hypotheses.H1
        }
    
    if parsed.test_method:
        result['test_method'] = parsed.test_method
    
    if parsed.test_statistic is not None:
        result['test_statistic'] = parsed.test_statistic
    
    if parsed.p_value is not None:
        result['p_value'] = parsed.p_value
    
    if parsed.decision:
        result['decision'] = parsed.decision
    
    if parsed.conclusion:
        result['conclusion'] = parsed.conclusion
    
    # Re-evaluate if we have ground truth - use correct nested structure for dashboard
    if 'ground_truth' in result:
        ground_truth = result['ground_truth']
        
        # Get or create evaluation dict - PRESERVE existing structure
        evaluation = result.get('evaluation', {})
        if not isinstance(evaluation, dict):
            evaluation = {}
        
        # P-value evaluation - nested structure required by dashboard
        p_val_pred = parsed.p_value
        p_val_true = ground_truth.get('p_value')
        
        if p_val_pred is not None and p_val_true is not None:
            p_error = abs(p_val_pred - p_val_true)
            p_tolerance = EVALUATION.get('p_value_tolerance', 0.05)
            p_relative_error = p_error / p_val_true if p_val_true != 0 else 0
            
            evaluation['p_value'] = {
                'exact_match': p_error < 0.001,
                'within_tolerance': p_error <= p_tolerance,
                'error': p_error,
                'relative_error': p_relative_error,
                'predicted': p_val_pred,
                'ground_truth': p_val_true,
                'valid_range': 0 <= p_val_pred <= 1,
                'correct_significance': (p_val_pred < 0.05) == (p_val_true < 0.05)
            }
        
        # Test statistic evaluation - nested structure
        stat_pred = parsed.test_statistic
        stat_true = ground_truth.get('test_statistic')
        
        if stat_pred is not None and stat_true is not None:
            stat_error = abs(stat_pred - stat_true)
            stat_relative_error = stat_error / abs(stat_true) if stat_true != 0 else 0
            
            evaluation['test_statistic'] = {
                'exact_match': stat_error < 0.001,
                'within_tolerance': stat_error <= 0.5,
                'error': stat_error,
                'relative_error': stat_relative_error,
                'predicted': stat_pred,
                'ground_truth': stat_true
            }
        
        # Decision evaluation - nested structure
        decision_pred = parsed.decision
        decision_true = ground_truth.get('decision')
        
        if decision_pred and decision_true:
            evaluation['decision'] = {
                'correct': decision_pred == decision_true,
                'predicted': decision_pred,
                'ground_truth': decision_true
            }
        
        # Preserve or create completeness
        if 'completeness' not in evaluation:
            evaluation['completeness'] = {
                'has_hypotheses': parsed.hypotheses is not None,
                'has_test_method': parsed.test_method is not None,
                'has_test_statistic': parsed.test_statistic is not None,
                'has_p_value': parsed.p_value is not None,
                'has_decision': parsed.decision is not None
            }
        else:
            # Update existing completeness
            evaluation['completeness']['has_p_value'] = parsed.p_value is not None
            evaluation['completeness']['has_decision'] = parsed.decision is not None
            evaluation['completeness']['has_test_statistic'] = parsed.test_statistic is not None
        
        # Preserve existing reasoning_quality and hallucinations if present
        # (don't overwrite what was there)
        
        # Calculate overall accuracy using the correct formula
        scores = []
        
        # Decision contributes 0.5 if correct
        if evaluation.get('decision', {}).get('correct', False):
            scores.append(0.5)
        else:
            scores.append(0.0)
        
        # P-value within tolerance contributes 0.25
        if evaluation.get('p_value', {}).get('within_tolerance', False):
            scores.append(0.25)
        else:
            scores.append(0.0)
        
        # Test statistic within tolerance contributes 0.125
        if evaluation.get('test_statistic', {}).get('within_tolerance', False):
            scores.append(0.125)
        else:
            scores.append(0.0)
        
        # Test method contributes 0.125 (use existing if present)
        test_method_score = evaluation.get('test_method', 0)
        if isinstance(test_method_score, (int, float)):
            scores.append(test_method_score * 0.125)
        
        evaluation['overall_accuracy'] = sum(scores)
        
        result['evaluation'] = evaluation
    
    return result

def process_file(file_path: Path, backup: bool = True, dry_run: bool = False) -> dict:
    """Process a single result file"""
    stats = {
        "total_results": 0,
        "cot_results": 0,
        "reparsed": 0,
        "improved_p_value": 0,
        "improved_decision": 0,
        "improved_overall": 0,
        "errors": 0
    }
    
    parser = ResponseParser()
    metrics = EvaluationMetrics()
    
    try:
        # Load file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        stats["total_results"] = len(data)
        
        # Track improvements
        updated_data = []
        
        for result in data:
            old_p_value = result.get('p_value')
            old_decision = result.get('decision')
            old_eval = result.get('evaluation', {})
            old_overall = old_eval.get('overall_accuracy', 0) if isinstance(old_eval, dict) else 0
            
            # Only process CoT responses for this pipeline
            is_cot = result.get('prompt_type') == 'chain_of_thought'
            if is_cot:
                stats["cot_results"] += 1
            
            try:
                # Re-parse and re-evaluate
                updated_result = reparse_and_evaluate_result(result, parser, metrics)
                
                if is_cot:
                    stats["reparsed"] += 1
                    
                    # Track improvements
                    new_p_value = updated_result.get('p_value')
                    new_decision = updated_result.get('decision')
                    new_eval = updated_result.get('evaluation', {})
                    new_overall = new_eval.get('overall_accuracy', 0) if isinstance(new_eval, dict) else 0
                    
                    if new_p_value is not None and old_p_value is None:
                        stats["improved_p_value"] += 1
                    
                    if new_decision is not None and old_decision is None:
                        stats["improved_decision"] += 1
                    
                    if new_overall > old_overall:
                        stats["improved_overall"] += 1
                
                updated_data.append(updated_result)
                
            except Exception as e:
                print(f"    Error processing result: {e}")
                stats["errors"] += 1
                updated_data.append(result)  # Keep original on error
        
        # Save if not dry run
        if not dry_run and stats["reparsed"] > 0:
            # Backup original
            if backup:
                backup_dir = create_backup_dir()
                backup_file(file_path, backup_dir)
            
            # Save updated data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, indent=2, ensure_ascii=False)
    
    except Exception as e:
        print(f"    Error loading file: {e}")
        stats["errors"] += 1
    
    return stats

def main(dry_run: bool = False, backup: bool = True):
    """Run complete pipeline on all result files"""
    results_dir = Path(RESULTS_DIR)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    json_files = list(results_dir.glob("*.json"))
    
    if not json_files:
        print(f"❌ No JSON files found in {results_dir}")
        return
    
    print("=" * 70)
    print(f"{'🔍 DRY RUN MODE' if dry_run else '🔄 RE-PARSING AND RE-EVALUATING CoT RESPONSES'}")
    print(f"📁 Processing {len(json_files)} files in {results_dir}")
    if backup and not dry_run:
        print(f"💾 Backups will be created in {results_dir}/backups")
    print("=" * 70)
    print()
    
    total_stats = {
        "total_results": 0,
        "cot_results": 0,
        "reparsed": 0,
        "improved_p_value": 0,
        "improved_decision": 0,
        "improved_overall": 0,
        "errors": 0
    }
    
    files_with_improvements = []
    
    for file in json_files:
        stats = process_file(file, backup=backup, dry_run=dry_run)
        
        # Aggregate stats
        for key in total_stats:
            total_stats[key] += stats[key]
        
        # Report significant changes
        if stats["improved_p_value"] > 0 or stats["improved_decision"] > 0 or stats["improved_overall"] > 0:
            files_with_improvements.append(file.name)
            print(f"  ✅ {file.name}:")
            print(f"     CoT: {stats['cot_results']}, Reparsed: {stats['reparsed']}")
            print(f"     Improved: +{stats['improved_p_value']} p-values, "
                  f"+{stats['improved_decision']} decisions, "
                  f"+{stats['improved_overall']} overall accuracy")
    
    print()
    print("=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Total results: {total_stats['total_results']}")
    print(f"CoT results: {total_stats['cot_results']}")
    print(f"✅ Re-parsed: {total_stats['reparsed']}")
    print()
    print("🎯 IMPROVEMENTS:")
    print(f"   +{total_stats['improved_p_value']} p-value extractions")
    print(f"   +{total_stats['improved_decision']} decision extractions")
    print(f"   +{total_stats['improved_overall']} overall accuracy scores")
    print()
    print(f"❌ Errors: {total_stats['errors']}")
    print()
    print(f"📁 Files with improvements: {len(files_with_improvements)}")
    
    if dry_run:
        print()
        print("=" * 70)
        print("💡 Run without --dry-run flag to apply changes:")
        print("   python update_cot_pipeline.py")
        print("=" * 70)
    else:
        print()
        print("=" * 70)
        print("✅ PIPELINE COMPLETE!")
        print(f"📊 Dashboard will now show updated CoT metrics")
        print(f"🔄 Refresh your dashboard to see improvements")
        print("=" * 70)

if __name__ == "__main__":
    # Check for flags
    dry_run = "--dry-run" in sys.argv
    no_backup = "--no-backup" in sys.argv
    
    main(dry_run=dry_run, backup=not no_backup)
