"""
Retroactively add hallucination detection to all existing benchmark results.
This ensures fairness and consistency across all models by applying the same
hallucination detection to both old and new results.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.hallucination_detector import HallucinationDetector
from src.response_parser import ParsedResponse
from src import config

def backup_results():
    """Create timestamped backup of all results before modification"""
    backup_dir = config.RESULTS_DIR / f"backups/pre-hallucination-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    result_files = list(config.RESULTS_DIR.glob("*.json"))
    print(f"📦 Backing up {len(result_files)} result files to {backup_dir.name}")
    
    for result_file in result_files:
        shutil.copy2(result_file, backup_dir / result_file.name)
    
    print(f"✅ Backup complete: {backup_dir}")
    return backup_dir

def add_hallucinations_to_result(result: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    """Add hallucination detection to a single result record
    
    Args:
        result: The result dict to update
        force: If True, re-run detection even if hallucinations already exist
    """
    
    # Skip if hallucinations already exist (unless force=True)
    if not force and 'hallucinations' in result and result['hallucinations']:
        return result
    
    # Extract necessary fields
    parsed_results = result.get('parsed_results', {})
    raw_response = result.get('raw_response', '')
    ground_truth = result.get('ground_truth', {})
    
    # Skip if missing critical data
    if not parsed_results or not ground_truth:
        result['hallucinations'] = {
            'has_hallucination': False,
            'hallucination_types': [],
            'severity': 'none',
            'details': {'structural': [], 'numerical': [], 'logical': [], 'reasoning': []},
            'counts': {'structural': 0, 'numerical': 0, 'logical': 0, 'reasoning': 0}
        }
        return result
    
    # Convert to ParsedResponse if it's a dict
    if isinstance(parsed_results, dict):
        try:
            parsed = ParsedResponse(**parsed_results)
        except Exception:
            # Use dict directly if conversion fails
            parsed = parsed_results
    else:
        parsed = parsed_results
    
    # Run hallucination detection
    try:
        hallucination_results = HallucinationDetector.detect_all(
            parsed=parsed,
            raw_output=raw_response,
            ground_truth=ground_truth
        )
        result['hallucinations'] = hallucination_results
    except Exception as e:
        print(f"  ⚠️  Hallucination detection failed: {e}")
        result['hallucinations'] = {
            'has_hallucination': False,
            'hallucination_types': [],
            'severity': 'none',
            'details': {'structural': [], 'numerical': [], 'logical': [], 'reasoning': []},
            'counts': {'structural': 0, 'numerical': 0, 'logical': 0, 'reasoning': 0},
            'error': str(e)
        }
    
    return result

def process_result_file(filepath: Path, force: bool = False) -> Dict[str, int]:
    """Process a single result file and add hallucination detection
    
    Args:
        filepath: Path to the result JSON file
        force: If True, re-run detection even if hallucinations already exist
    """
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = {
        'total': 0,
        'added': 0,
        'skipped': 0,
        'errors': 0
    }
    
    # Handle both single result and list of results
    if isinstance(data, list):
        results = data
    else:
        results = [data]
    
    updated_results = []
    for result in results:
        stats['total'] += 1
        
        # Skip only if not forcing and hallucinations exist
        if not force and 'hallucinations' in result and result['hallucinations']:
            stats['skipped'] += 1
            updated_results.append(result)
            continue
        
        try:
            updated_result = add_hallucinations_to_result(result, force=force)
            stats['added'] += 1
            updated_results.append(updated_result)
        except Exception as e:
            print(f"  ❌ Error processing result: {e}")
            stats['errors'] += 1
            updated_results.append(result)
    
    # Save updated results
    with open(filepath, 'w', encoding='utf-8') as f:
        if isinstance(data, list):
            json.dump(updated_results, f, indent=2, ensure_ascii=False)
        else:
            json.dump(updated_results[0], f, indent=2, ensure_ascii=False)
    
    return stats

def main():
    """Main execution: backfill hallucination detection to all existing results"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backfill hallucination detection")
    parser.add_argument("--force", action="store_true", 
                       help="Force re-detection even if hallucinations already exist")
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔍 HALLUCINATION DETECTION BACKFILL")
    print("=" * 70)
    print()
    if args.force:
        print("⚠️  FORCE MODE: Re-detecting hallucinations for ALL results")
        print()
    print("This script will:")
    print("  1. Backup all existing results")
    print("  2. Add hallucination detection to results" + (" (forced re-detection)" if args.force else " lacking it"))
    print("  3. Save updated results with consistent hallucination data")
    print()
    
    # Create backup
    backup_dir = backup_results()
    print()
    
    # Find all result files
    result_files = list(config.RESULTS_DIR.glob("*.json"))
    print(f"📊 Found {len(result_files)} result files to process")
    print()
    
    # Process each file
    total_stats = {
        'total': 0,
        'added': 0,
        'skipped': 0,
        'errors': 0
    }
    
    for filepath in tqdm(result_files, desc="Processing result files"):
        try:
            stats = process_result_file(filepath, force=args.force)
            for key in total_stats:
                total_stats[key] += stats[key]
        except Exception as e:
            print(f"\n❌ Failed to process {filepath.name}: {e}")
            total_stats['errors'] += 1
    
    # Summary
    print()
    print("=" * 70)
    print("📈 BACKFILL SUMMARY")
    print("=" * 70)
    print(f"Total results processed:        {total_stats['total']:,}")
    print(f"Hallucinations {'re-detected' if args.force else 'added'}:           {total_stats['added']:,}")
    if not args.force:
        print(f"Already had hallucinations:     {total_stats['skipped']:,}")
    print(f"Errors encountered:             {total_stats['errors']:,}")
    print()
    print(f"✅ Backfill complete!")
    print(f"📂 Backup location: {backup_dir}")
    print()
    
    if total_stats['added'] > 0:
        print("🎯 Next steps:")
        print("  1. Verify results: Check a few JSON files to ensure hallucination data is present")
        print("  2. Refresh dashboard: streamlit run dashboard/app.py")
        print("  3. Analyze hallucination patterns across all models")
    
    if total_stats['errors'] > 0:
        print()
        print(f"⚠️  Warning: {total_stats['errors']} errors occurred during processing")
        print("   Review the error messages above and check affected files")

if __name__ == "__main__":
    main()
