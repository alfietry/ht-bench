"""Quick analysis of hallucination detection results"""
import json
from pathlib import Path
from collections import defaultdict

halluc_count = 0
total = 0
severity_dist = {'none': 0, 'minor': 0, 'moderate': 0, 'severe': 0}
type_counts = {'structural': 0, 'numerical': 0, 'logical': 0, 'reasoning': 0}
examples = []

for f in Path('results').glob('*.json'):
    try:
        data = json.load(open(f, encoding='utf-8'))
        results = data if isinstance(data, list) else [data]
        
        for r in results:
            total += 1
            h = r.get('hallucinations', {})
            
            if h.get('has_hallucination'):
                halluc_count += 1
                severity_dist[h.get('severity', 'none')] += 1
                
                for t, c in h.get('counts', {}).items():
                    type_counts[t] += c
                
                # Collect examples
                if len(examples) < 5:
                    examples.append({
                        'model': r.get('model'),
                        'test_type': r.get('input_data', {}).get('test_type'),
                        'severity': h.get('severity'),
                        'types': h.get('hallucination_types'),
                        'details': h.get('details')
                    })
            else:
                severity_dist['none'] += 1
    except Exception as e:
        print(f"Error processing {f}: {e}")

print(f'Total results: {total:,}')
print(f'Results with hallucinations: {halluc_count:,} ({halluc_count/total*100:.1f}%)')
print(f'\nSeverity distribution:')
for k, v in severity_dist.items():
    print(f'  {k:10s}: {v:5,} ({v/total*100:.1f}%)')
print(f'\nHallucination type total counts:')
for k, v in type_counts.items():
    print(f'  {k:12s}: {v:5,}')

if examples:
    print(f'\n--- Example Hallucinations ---')
    for i, ex in enumerate(examples, 1):
        print(f'\nExample {i}:')
        print(f"  Model: {ex['model']}")
        print(f"  Test: {ex['test_type']}")
        print(f"  Severity: {ex['severity']}")
        print(f"  Types: {', '.join(ex['types'])}")
        print(f"  Issues:")
        for cat, issues in ex['details'].items():
            if issues:
                print(f"    {cat}: {issues}")
