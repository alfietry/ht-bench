"""
Re-parse normalized CoT responses to update evaluation metrics.
This script re-parses all responses using the updated parser that handles
normalized CoT formats and inequality symbols (<, >).
"""
import json
from pathlib import Path
from response_parser import ResponseParser
from evaluator import EvaluationMetrics
import sys

def reparse_result(result: dict, parser: ResponseParser) -> dict:
    """Re-parse a single result and update evaluation metrics"""
    # Get the response text
    response_text = result.get('raw_response') or result.get('response', '')
    
    if not response_text:
        return result
    
    # Parse the response
    parsed = parser.parse(response_text)
    
    # Update parsed fields
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
    
    return result

def reparse_file(file_path: Path, dry_run: bool = False) -> dict:
    """Re-parse all responses in a single result file"""
    stats = {
        "total_results": 0,
        "reparsed": 0,
        "improved_p_value": 0,
        "improved_decision": 0,
        "failed": 0
    }
    
    parser = ResponseParser()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        stats["total_results"] = len(data)
        
        for result in data:
            old_p_value = result.get('p_value')
            old_decision = result.get('decision')
            
            try:
                # Re-parse the result
                result = reparse_result(result, parser)
                
                stats["reparsed"] += 1
                
                # Track improvements
                new_p_value = result.get('p_value')
                new_decision = result.get('decision')
                
                if new_p_value is not None and old_p_value is None:
                    stats["improved_p_value"] += 1
                
                if new_decision is not None and old_decision is None:
                    stats["improved_decision"] += 1
                    
            except Exception as e:
                print(f"  Error reparsing result: {e}")
                stats["failed"] += 1
        
        # Save back to file if not dry run
        if not dry_run and stats["reparsed"] > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        print(f"  Error processing file {file_path.name}: {e}")
    
    return stats

def main(dry_run: bool = False):
    """Re-parse all responses in results folder"""
    results_dir = Path(__file__).parent / "results"
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    json_files = list(results_dir.glob("*.json"))
    
    if not json_files:
        print(f"❌ No JSON files found in {results_dir}")
        return
    
    print(f"{'🔍 DRY RUN MODE - No files will be modified' if dry_run else '🔄 RE-PARSING RESPONSES'}")
    print(f"📁 Found {len(json_files)} JSON files in {results_dir}")
    print("")
    
    total_stats = {
        "total_results": 0,
        "reparsed": 0,
        "improved_p_value": 0,
        "improved_decision": 0,
        "failed": 0
    }
    
    for file in json_files:
        stats = reparse_file(file, dry_run=dry_run)
        
        for key in total_stats:
            total_stats[key] += stats[key]
        
        if stats["improved_p_value"] > 0 or stats["improved_decision"] > 0:
            print(f"  {file.name}: +{stats['improved_p_value']} p-values, +{stats['improved_decision']} decisions")
    
    print("")
    print("=" * 60)
    print(f"📊 SUMMARY:")
    print(f"  Total results: {total_stats['total_results']}")
    print(f"  ✅ Re-parsed: {total_stats['reparsed']}")
    print(f"  📈 Improved p-values: {total_stats['improved_p_value']}")
    print(f"  📈 Improved decisions: {total_stats['improved_decision']}")
    print(f"  ❌ Failed: {total_stats['failed']}")
    
    if dry_run:
        print("")
        print("💡 Run without --dry-run flag to apply changes:")
        print("   python reparse_normalized_responses.py")
    else:
        print(f"  📝 Updated {len(json_files)} files")

if __name__ == "__main__":
    # Check for dry-run flag
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
