"""
Main orchestration for LLM Hypothesis Testing Benchmark
"""
import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm.asyncio import tqdm
import numpy as np

import config
from llm_clients import get_client, LLMClient
from prompts import get_prompt, RESPONSE_SCHEMA
from data_generator import DataGenerator, create_test_context
from statistical_engine import StatisticalEngine
from response_parser import ResponseParser
from evaluator import EvaluationMetrics

# Set up logging
config.LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(
            config.LOGS_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            mode='w',
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)


class HypothesisTestingBenchmark:
    """Main benchmark orchestrator"""
    
    def __init__(self, output_dir: Path = config.RESULTS_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.data_generator = DataGenerator()
        self.results = []
    
    async def evaluate_single(self,
                             client: LLMClient,
                             model_name: str,
                             prompt_type: str,
                             scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single scenario"""
        latency_seconds = None
        try:
            # Generate prompt
            test_context = create_test_context(scenario["test_type"])
            prompt = get_prompt(prompt_type, scenario, test_context)
            
            # Get LLM response
            start_time = time.perf_counter()
            if prompt_type == "structured":
                raw_response = await client.generate_structured(
                    prompt, RESPONSE_SCHEMA, temperature=0.0
                )
                # Convert to string for logging
                raw_response_text = json.dumps(raw_response)
            else:
                raw_response_text = await client.generate(prompt, temperature=0.0)
                raw_response = raw_response_text
            latency_seconds = time.perf_counter() - start_time
            
            # Parse response
            if isinstance(raw_response, dict):
                from response_parser import ParsedResponse
                try:
                    parsed = ParsedResponse(**raw_response)
                except Exception as parse_error:
                    logger.warning(f"Failed to parse structured response: {parse_error}")
                    # Fallback to regex parsing
                    parsed = ResponseParser.parse(json.dumps(raw_response), format="auto")
            else:
                parsed = ResponseParser.parse(raw_response_text, format="auto")
            
            # Compute ground truth
            ground_truth = StatisticalEngine.compute_ground_truth(scenario)
            
            # Evaluate
            evaluation = EvaluationMetrics.comprehensive_evaluation(
                parsed, ground_truth, raw_response_text if isinstance(raw_response, str) else ""
            )
            
            # Create result record
            result = {
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "prompt_type": prompt_type,
                "prompt": prompt,
                "input_data": {
                    "test_type": scenario["test_type"],
                    "metadata": scenario.get("metadata", {})
                },
                "raw_response": raw_response_text if isinstance(raw_response, str) else json.dumps(raw_response),
                "parsed_results": parsed.model_dump() if hasattr(parsed, 'model_dump') else parsed,
                "ground_truth": ground_truth,
                "evaluation": evaluation,
                "latency_seconds": latency_seconds,
            }
            
            logger.info(f"Completed: {model_name} | {prompt_type} | {scenario['test_type']} | Accuracy: {evaluation['overall_accuracy']:.2f}")
            
            return result
            
        except Exception as e:
            import traceback
            logger.error(f"Error evaluating {model_name} on {scenario['test_type']}: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return {
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "prompt_type": prompt_type,
                "input_data": {"test_type": scenario.get("test_type", "unknown")},
                "error": str(e),
                "error_type": type(e).__name__,
                "evaluation": {"overall_accuracy": 0.0},
                "latency_seconds": latency_seconds,
            }
    
    async def evaluate_batch(self,
                            models: Dict[str, List[str]],
                            prompt_types: List[str],
                            test_types: List[str],
                            sample_sizes: Optional[List[int]] = None,
                            max_scenarios_per_type: int = 3) -> List[Dict[str, Any]]:
        """Evaluate multiple models on multiple scenarios"""
        
        if sample_sizes is None:
            sample_sizes = config.SAMPLE_SIZES[:2]  # Limit for faster testing
        
        # Generate scenarios
        logger.info("Generating test scenarios...")
        scenarios = []
        for test_type in test_types:
            for sample_size in sample_sizes:
                for _ in range(max_scenarios_per_type):
                    try:
                        # Generate scenario with appropriate parameters for each test type
                        if test_type in ["one_sample_t_test", "one_sample_z_test"]:
                            scenario = self.data_generator.generate_scenario(
                                test_type,
                                sample_size=sample_size
                            )
                        elif test_type in ["two_sample_t_test", "two_sample_z_test"]:
                            scenario = self.data_generator.generate_scenario(
                                test_type,
                                sample_size1=sample_size,
                                sample_size2=sample_size
                            )
                        elif test_type == "paired_t_test":
                            scenario = self.data_generator.generate_scenario(
                                test_type,
                                sample_size1=sample_size,
                                sample_size2=sample_size
                            )
                        elif test_type == "anova":
                            scenario = self.data_generator.generate_scenario(
                                test_type,
                                samples_per_group=sample_size
                            )
                        elif test_type in ["chi_square_goodness_of_fit", "chi_square_independence"]:
                            scenario = self.data_generator.generate_scenario(
                                test_type,
                                n_samples=sample_size
                            )
                        else:
                            continue
                        
                        scenarios.append(scenario)
                    except Exception as e:
                        logger.warning(f"Error generating {test_type}: {e}")
        
        logger.info(f"Generated {len(scenarios)} scenarios")
        
        # Create evaluation tasks
        tasks = []
        for provider, model_list in models.items():
            for model_name in model_list:
                try:
                    client = get_client(provider, model_name)
                    
                    for prompt_type in prompt_types:
                        for scenario in scenarios:
                            tasks.append(self.evaluate_single(
                                client, model_name, prompt_type, scenario
                            ))
                except Exception as e:
                    logger.error(f"Error creating client for {provider}/{model_name}: {e}")
        
        logger.info(f"Running {len(tasks)} evaluation tasks...")
        
        # Run evaluations with progress bar
        results = []
        semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
        
        async def run_with_semaphore(task):
            async with semaphore:
                return await task
        
        # Use tqdm for progress tracking
        for coro in tqdm.as_completed([run_with_semaphore(task) for task in tasks], 
                                      total=len(tasks), desc="Evaluating"):
            result = await coro
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def save_results(self, filename: Optional[str] = None):
        """Save results to JSON file"""
        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
        return filepath
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not self.results:
            return {}
        
        from evaluator import BenchmarkAggregator
        
        summary = {
            "total_evaluations": len(self.results),
            "by_model": BenchmarkAggregator.aggregate_by_model(self.results),
            "by_prompt_type": BenchmarkAggregator.aggregate_by_prompt_type(self.results),
            "by_test_type": BenchmarkAggregator.aggregate_by_test_type(self.results),
            "comparison_matrix": BenchmarkAggregator.create_comparison_matrix(self.results)
        }
        
        return summary
    
    def print_summary(self):
        """Print summary to console"""
        summary = self.generate_summary()
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Check if we have any evaluations
        if not self.results or summary.get('total_evaluations', 0) == 0:
            print("\n⚠️  No evaluations completed. Check logs for errors.")
            print("\nPossible issues:")
            print("  - API keys not configured")
            print("  - Data generation errors")
            print("  - Model availability issues")
            return
        
        print(f"\nTotal Evaluations: {summary['total_evaluations']}")
        
        print("\n--- Performance by Model ---")
        for model, stats in summary['by_model'].items():
            print(f"\n{model}:")
            print(f"  Overall Accuracy: {stats['overall_accuracy']['mean']:.2%} ± {stats['overall_accuracy']['std']:.2%}")
            print(f"  Decision Accuracy: {stats['decision_accuracy']['correct_rate']:.2%}")
            print(f"  Test Method Match: {stats['test_method_accuracy']['exact_match_rate']:.2%}")
            print(f"  Hallucination Rate: {stats['hallucination_rate']:.2%}")
            latency_stats = stats.get('latency')
            if latency_stats and latency_stats.get('mean') is not None:
                print(f"  Avg Latency: {latency_stats['mean']:.2f}s")
        
        print("\n--- Performance by Prompt Type ---")
        for prompt_type, stats in summary['by_prompt_type'].items():
            print(f"\n{prompt_type}:")
            print(f"  Overall Accuracy: {stats['overall_accuracy']['mean']:.2%}")
        
        print("\n" + "="*80)


async def run_quick_test():
    """Run a quick test with a subset of models and scenarios"""
    logger.info("Starting quick test...")
    
    benchmark = HypothesisTestingBenchmark()
    
    # Test with a small subset
    models = {
        "openai": ["gpt-4o-mini"],
    }
    
    prompt_types = ["zero_shot", "chain_of_thought"]
    test_types = ["one_sample_t_test", "two_sample_t_test"]
    
    await benchmark.evaluate_batch(
        models=models,
        prompt_types=prompt_types,
        test_types=test_types,
        sample_sizes=[50],
        max_scenarios_per_type=2
    )
    
    benchmark.save_results("quick_test_results.json")
    benchmark.print_summary()


async def run_full_benchmark():
    """Run comprehensive benchmark across all models and scenarios"""
    logger.info("Starting full benchmark...")
    
    benchmark = HypothesisTestingBenchmark()
    
    # Target model set for full benchmark (skip Google, exclude Ollama)
    full_models = config.FULL_MODE_MODEL_MAP

    # Filter out providers without configured API keys
    models = {}
    for provider, model_list in full_models.items():
        if config.API_KEYS.get(provider):
            models[provider] = model_list
        else:
            logger.warning(f"Skipping {provider}: missing API key")
    
    # Skip ANOVA, z-tests, and chi-square variants in full mode
    excluded_tests = {
        "anova",
        "one_sample_z_test",
        "two_sample_z_test",
        "chi_square_goodness_of_fit",
        "chi_square_independence",
    }
    test_types = [t for t in config.TEST_TYPES if t not in excluded_tests]

    prompt_types = config.PROMPT_TYPES

    await benchmark.evaluate_batch(
        models=models,
        prompt_types=prompt_types,
        test_types=test_types,
        sample_sizes=config.SAMPLE_SIZES,
        max_scenarios_per_type=5
    )
    
    benchmark.save_results()
    benchmark.print_summary()
    
    # Save summary separately
    summary = benchmark.generate_summary()
    summary_path = config.RESULTS_DIR / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Summary saved to {summary_path}")


async def run_custom_benchmark(
    providers: List[str],
    models_per_provider: Dict[str, List[str]],
    prompt_types: List[str],
    test_types: List[str],
    sample_sizes: Optional[List[int]] = None,
    scenarios_per_type: int = 3
):
    """Run custom benchmark with specified parameters"""
    logger.info("Starting custom benchmark...")
    
    benchmark = HypothesisTestingBenchmark()
    
    await benchmark.evaluate_batch(
        models=models_per_provider,
        prompt_types=prompt_types,
        test_types=test_types,
        sample_sizes=sample_sizes,
        max_scenarios_per_type=scenarios_per_type
    )
    
    filepath = benchmark.save_results()
    benchmark.print_summary()
    
    return filepath


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Hypothesis Testing Benchmark")
    parser.add_argument("--mode", choices=["quick", "full", "custom"], default="quick",
                       help="Benchmark mode")
    parser.add_argument("--models", nargs="+", help="Models to test (format: provider/model)")
    parser.add_argument("--prompts", nargs="+", choices=config.PROMPT_TYPES,
                       help="Prompt types to test (default: all)")
    parser.add_argument("--tests", nargs="+", choices=config.TEST_TYPES,
                       help="Test types to evaluate (default: all)")
    parser.add_argument("--sample-sizes", nargs="+", type=int,
                       help="Sample sizes to use (default: config.SAMPLE_SIZES)")
    parser.add_argument("--scenarios", type=int,
                       help="Number of scenarios per test type (default: 5)")
    parser.add_argument("--evaluations", type=int,
                       help="Alias for --scenarios")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        asyncio.run(run_quick_test())
    elif args.mode == "full":
        asyncio.run(run_full_benchmark())
    else:  # custom
        if args.models:
            # Parse models
            models_dict = {}
            for model_spec in args.models:
                provider, model = model_spec.split("/")
                if provider not in models_dict:
                    models_dict[provider] = []
                models_dict[provider].append(model)
            
            prompt_types = args.prompts if args.prompts else config.PROMPT_TYPES
            test_types = args.tests if args.tests else config.TEST_TYPES
            sample_sizes = args.sample_sizes if args.sample_sizes else config.SAMPLE_SIZES
            scenarios_per_type = (args.evaluations if args.evaluations is not None else args.scenarios)
            if scenarios_per_type is None:
                scenarios_per_type = 5

            asyncio.run(run_custom_benchmark(
                providers=list(models_dict.keys()),
                models_per_provider=models_dict,
                prompt_types=prompt_types,
                test_types=test_types,
                sample_sizes=sample_sizes,
                scenarios_per_type=scenarios_per_type
            ))
        else:
            logger.error("Custom mode requires --models argument")


if __name__ == "__main__":
    main()
