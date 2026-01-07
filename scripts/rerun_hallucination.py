"""Quick re-run of hallucination detection with fixed logic"""
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from src.hallucination_detector import HallucinationDetector
from src.response_parser import ParsedResponse
from src import config

print("=" * 70)
print("🔍 HALLUCINATION RE-DETECTION (FIXED LOGIC)")
print("=" * 70)

result_files = list(config.RESULTS_DIR.glob("*.json"))
print(f"\n📊 Processing {len(result_files)} result files...")

total = 0
updated = 0
errors = 0

for filepath in tqdm(result_files, desc="Processing"):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data if isinstance(data, list) else [data]
        
        for result in results:
            total += 1
            parsed_results = result.get('parsed_results', {})
            raw_response = result.get('raw_response', '')
            ground_truth = result.get('ground_truth', {})
            
            if not parsed_results or not ground_truth:
                continue
            
            try:
                parsed = ParsedResponse(**parsed_results)
            except:
                parsed = parsed_results
            
            halluc = HallucinationDetector.detect_all(
                parsed=parsed,
                raw_output=raw_response,
                ground_truth=ground_truth
            )
            result['hallucinations'] = halluc
            updated += 1
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data if isinstance(data, list) else results[0], f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"\n❌ Error: {filepath.name}: {e}")
        errors += 1

print(f"\n✅ Complete! Processed {total:,} results, updated {updated:,}, errors: {errors}")
