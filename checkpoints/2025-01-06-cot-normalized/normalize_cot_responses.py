"""
Normalize Chain-of-Thought responses in existing result files to structured format.
"""
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional
import sys

def extract_hypotheses(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract H0 and H1 from CoT response"""
    # Pattern for H0
    h0_patterns = [
        r'H0:\s*([^\n]+)',
        r'Null hypothesis:\s*([^\n]+)',
        r'H₀:\s*([^\n]+)',
    ]
    
    # Pattern for H1
    h1_patterns = [
        r'H1:\s*([^\n]+)',
        r'Alternative hypothesis:\s*([^\n]+)',
        r'H₁:\s*([^\n]+)',
        r'Ha:\s*([^\n]+)',
    ]
    
    h0 = None
    h1 = None
    
    for pattern in h0_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            h0 = match.group(1).strip()
            break
    
    for pattern in h1_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            h1 = match.group(1).strip()
            break
    
    return h0, h1

def extract_test_type(text: str) -> Optional[str]:
    """Extract test type from CoT response"""
    patterns = [
        r'(one[- ]sample.*?t[- ]test)',
        r'(two[- ]sample.*?t[- ]test)',
        r'(paired.*?t[- ]test)',
        r'(independent.*?t[- ]test)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None

def extract_test_statistic(text: str) -> Optional[float]:
    """Extract test statistic from CoT response"""
    # Pattern for t = value or t-statistic = value
    patterns = [
        r't\s*=\s*([+-]?\d+\.?\d*)',
        r't[- ]statistic\s*=\s*([+-]?\d+\.?\d*)',
        r't[- ]stat\s*=\s*([+-]?\d+\.?\d*)',
        # Also catch from equation like "= -0.361"
        r'/\s*sqrt\([^\)]+\)\s*=\s*[^\s]+\s*=\s*([+-]?\d+\.?\d*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    return None

def extract_p_value(text: str) -> Optional[float]:
    """Extract p-value from CoT response, handling < and > symbols"""
    # Patterns for p-value with various formats
    patterns = [
        r'p[- ]?value\s*[≈~]?\s*([0-9.]+)',
        r'p\s*[≈~]?\s*([0-9.]+)',
        r'p[- ]?value\s*<\s*([0-9.]+)',  # p < 0.05
        r'p\s*<\s*([0-9.]+)',
        r'p[- ]?value\s*>\s*([0-9.]+)',  # p > 0.05
        r'p\s*>\s*([0-9.]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                # If p-value is stated as "p < 0.05", use 0.04 as estimate
                # If p-value is stated as "p > 0.05", use 0.06 as estimate
                if '<' in pattern and val == 0.05:
                    return 0.04  # Conservative estimate
                elif '>' in pattern and val == 0.05:
                    return 0.06  # Conservative estimate
                return val
            except ValueError:
                continue
    
    return None

def extract_decision(text: str) -> Optional[str]:
    """Extract decision from CoT response"""
    # Patterns for reject/fail to reject
    if re.search(r'fail\s+to\s+reject', text, re.IGNORECASE):
        return "fail_to_reject_H0"
    elif re.search(r'do\s+not\s+reject', text, re.IGNORECASE):
        return "fail_to_reject_H0"
    elif re.search(r'reject', text, re.IGNORECASE):
        return "reject_H0"
    
    return None

def extract_conclusion(text: str) -> Optional[str]:
    """Extract conclusion from CoT response"""
    # Look for lines starting with 6, "Conclusion:", or similar
    patterns = [
        r'6\.\s*([^\n]+)',
        r'Conclusion:\s*([^\n]+)',
        r'Interpretation:\s*([^\n]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Fallback: last sentence
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    if sentences:
        return sentences[-1]
    
    return None

def normalize_cot_response(response_text: str) -> str:
    """Convert CoT response to structured format"""
    # Extract all components
    h0, h1 = extract_hypotheses(response_text)
    test_type = extract_test_type(response_text)
    test_stat = extract_test_statistic(response_text)
    p_value = extract_p_value(response_text)
    decision = extract_decision(response_text)
    conclusion = extract_conclusion(response_text)
    
    # Build normalized response
    lines = []
    
    if h0:
        lines.append(f"H0: {h0}")
    if h1:
        lines.append(f"H1: {h1}")
    if test_type:
        lines.append(f"Test_type: {test_type}")
    if test_stat is not None:
        lines.append(f"t-statistic: {test_stat:.4f}")
    if p_value is not None:
        lines.append(f"p-value: {p_value:.4f}")
    if decision:
        lines.append(f"Decision: {decision.replace('_', ' ').title()}")
    if conclusion:
        lines.append(f"Conclusion: {conclusion}")
    
    return "\n".join(lines)

def normalize_result_file(file_path: Path, dry_run: bool = False) -> Dict[str, int]:
    """Normalize CoT responses in a single result file"""
    stats = {
        "total_results": 0,
        "cot_results": 0,
        "normalized": 0,
        "failed": 0
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        stats["total_results"] = len(data)
        
        for result in data:
            # Only process CoT responses
            if result.get('prompt_type') != 'chain_of_thought':
                continue
            
            stats["cot_results"] += 1
            
            # Get the response text
            response_text = result.get('raw_response') or result.get('response', '')
            
            if not response_text:
                continue
            
            # Check if already normalized (has "Test_type:" line)
            if "Test_type:" in response_text:
                continue  # Already normalized
            
            try:
                # Normalize the response
                normalized = normalize_cot_response(response_text)
                
                if normalized and len(normalized.split('\n')) >= 3:  # At least 3 fields
                    if not dry_run:
                        # Update the response in the result
                        if 'raw_response' in result:
                            result['raw_response'] = normalized
                        if 'response' in result:
                            result['response'] = normalized
                    
                    stats["normalized"] += 1
                else:
                    stats["failed"] += 1
                    
            except Exception as e:
                print(f"  Error normalizing result: {e}")
                stats["failed"] += 1
        
        # Save back to file if not dry run
        if not dry_run and stats["normalized"] > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        print(f"  Error processing file {file_path.name}: {e}")
    
    return stats

def main(dry_run: bool = False):
    """Normalize all CoT responses in results folder"""
    results_dir = Path(__file__).parent / "results"
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    json_files = list(results_dir.glob("*.json"))
    
    if not json_files:
        print(f"❌ No JSON files found in {results_dir}")
        return
    
    print(f"{'🔍 DRY RUN MODE - No files will be modified' if dry_run else '✍️  NORMALIZING CoT RESPONSES'}")
    print(f"📁 Found {len(json_files)} JSON files in {results_dir}")
    print("")
    
    total_stats = {
        "total_results": 0,
        "cot_results": 0,
        "normalized": 0,
        "failed": 0
    }
    
    for file in json_files:
        stats = normalize_result_file(file, dry_run=dry_run)
        
        for key in total_stats:
            total_stats[key] += stats[key]
        
        if stats["normalized"] > 0 or stats["failed"] > 0:
            print(f"  {file.name}: {stats['normalized']} normalized, {stats['failed']} failed (out of {stats['cot_results']} CoT)")
    
    print("")
    print("=" * 60)
    print(f"📊 SUMMARY:")
    print(f"  Total results: {total_stats['total_results']}")
    print(f"  CoT results: {total_stats['cot_results']}")
    print(f"  ✅ Normalized: {total_stats['normalized']}")
    print(f"  ❌ Failed: {total_stats['failed']}")
    
    if dry_run:
        print("")
        print("💡 Run without --dry-run flag to apply changes:")
        print("   python normalize_cot_responses.py")
    else:
        print(f"  📝 Updated {len(json_files)} files")

if __name__ == "__main__":
    # Check for dry-run flag
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
