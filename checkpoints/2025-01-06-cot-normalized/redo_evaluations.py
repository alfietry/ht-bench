"""
Script to redo evaluations for all existing LLM responses and prepare data for the dashboard.
Automatically creates backups of the original files before overwriting.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from response_parser import ParsedResponse  # Fixed import
from evaluator import EvaluationMetrics  # Fixed import - assuming Evaluator class
from config import RESULTS_DIR

# Directory containing the results JSON files
RESULTS_DIR = Path(RESULTS_DIR)
BACKUP_DIR = RESULTS_DIR / "backups"

def create_backup(file: Path):
    """Create a backup of the given file in the backups directory."""
    if not BACKUP_DIR.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create a timestamped backup file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = BACKUP_DIR / f"{file.stem}_backup_{timestamp}{file.suffix}"
    shutil.copy2(file, backup_file)
    print(f"Backup created: {backup_file}")

def redo_evaluations():
    """Redo evaluations for all existing results."""
    if not RESULTS_DIR.exists():
        print(f"Results directory '{RESULTS_DIR}' does not exist.")
        return

    # Iterate over all JSON files in the results directory
    for file in RESULTS_DIR.glob("*.json"):
        try:
            # Create a backup of the original file
            create_backup(file)

            # Load the existing results file
            with open(file, "r") as f:
                data = json.load(f)

            # Ensure the data is a list of results
            if not isinstance(data, list):
                print(f"Skipping {file.name}: Not a list of results.")
                continue

            updated_results = []

            # Process each result
            for result in data:
                # Parse the raw response using the ParsedResponse class
                raw_response = result.get("raw_response", "")
                parsed_results = ParsedResponse(raw_response)  # Instantiate ParsedResponse

                # Evaluate the parsed response against the ground truth
                ground_truth = result.get("ground_truth", {})
                evaluator = Evaluator()  # Instantiate Evaluator
                evaluation = evaluator.evaluate(parsed_results, ground_truth)  # Call evaluate method

                # Update the result with the new parsed results and evaluation
                result["parsed_results"] = parsed_results.__dict__  # Convert to dict for JSON serialization
                result["evaluation"] = evaluation

                updated_results.append(result)

            # Save the updated results back to the same file
            with open(file, "w") as f:
                json.dump(updated_results, f, indent=4)

            print(f"Updated evaluations for {file.name}")

        except Exception as e:
            print(f"Error processing {file.name}: {e}")

if __name__ == "__main__":
    redo_evaluations()