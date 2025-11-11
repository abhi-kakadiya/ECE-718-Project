"""
Main benchmark runner script.

Orchestrates the complete experimental pipeline:
1. Run baseline benchmarks (NumPy/MKL)
2. Run all DSPy optimizers
3. Collect and save results
4. Generate visualizations
5. Create summary reports
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import dspy

from benchmarks.baseline_implementations import BaselineBenchmarks
from dspy_program.training_examples import get_training_examples
from dspy_program.baseline_optimizer import run_baseline_optimizer
from tools.generate_tables import TableGenerator
from dspy_program.bootstrap_optimizer import run_bootstrap_optimizer
from dspy_program.miprov2_optimizer import run_miprov2_optimizer
from tools.logger import get_logger, log_section, log_subsection
from tools.visualize_results import ResultsVisualizer
from tools.code_executor import CodeExecutor

logger = get_logger(__name__)


class BenchmarkRunner:
    """Main orchestrator for running all benchmarks."""

    def __init__(self, config_path: str = "configs/benchmark_config.yaml"):
        """
        Initialize benchmark runner.

        Args:
            config_path: Path to benchmark configuration file
        """
        self.config_path = config_path
        self.config = self.load_config()

        # Create output directories
        self.results_dir = Path(os.getenv('RESULTS_DIR', 'results'))  # For CSV tables
        self.figures_dir = Path(os.getenv('FIGURES_DIR', 'figures'))  # For visualizations
        self.logs_dir = Path(os.getenv('LOGS_DIR', 'logs'))  # For JSON results

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"\n{'=' * 80}\n"
            f"BENCHMARK RUNNER INITIALIZED\n"
            f"{'=' * 80}\n"
            f"  Config file:   {config_path}\n"
            f"  Results dir:   {self.results_dir} (CSV tables)\n"
            f"  Logs dir:      {self.logs_dir} (JSON results)\n"
            f"  Figures dir:   {self.figures_dir} (visualizations)\n"
            f"{'=' * 80}"
        )

    def load_config(self) -> Dict:
        """Load benchmark configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✓ Configuration loaded: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"✗ Failed to load config: {e}")
            logger.warning("⚠ Using default configuration")
            return self.get_default_config()

    def get_default_config(self) -> Dict:
        """Get default configuration if file not found."""
        return {
            'benchmarks': {
                'matrix_multiply': {
                    'sizes': [128, 256, 512],
                    'iterations': 10,
                    'warmup': 3
                },
                'cholesky': {
                    'sizes': [128, 256, 512],
                    'iterations': 10,
                    'warmup': 3
                },
                'fft': {
                    'sizes': [1024, 4096, 16384],
                    'iterations': 10,
                    'warmup': 3
                }
            },
            'optimizers': {
                'baseline': {'enabled': True},
                'bootstrap': {'enabled': True, 'max_bootstrapped_demos': 4},
                'miprov2': {'enabled': True, 'num_candidates': 10},
                'gepa': {'enabled': True, 'population_size': 20, 'generations': 10}
            }
        }

    def configure_dspy(self):
        """Configure DSPy with LLM settings."""
        import shutil

        logger.info("Configuring DSPy...")

        # Clear DSPy cache for fresh LLM responses
        cache_dir = Path(os.getenv('DSP_CACHE_DIR', './cache'))
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info(f"Cleared DSPy cache directory: {cache_dir}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        model = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
        api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment!")
            logger.error("Please set it in your .env file")
            raise ValueError("Missing OPENAI_API_KEY")

        lm = dspy.LM(
            model=model,
            api_key=api_key,
            temperature=float(os.getenv('LLM_TEMPERATURE', 0.7)),
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', 2048)),
            cache=False  # Disable caching for fresh results
        )

        dspy.settings.configure(lm=lm)

        logger.info(f"DSPy configured with model: {model}")
        logger.info("DSPy caching disabled for fresh results")

    def clear_old_results(self, benchmarks: List[str] = None):
        """
        Clear old results for fresh benchmark runs.

        Args:
            benchmarks: List of specific benchmarks to clear. If None, clears all.
        """
        import shutil

        logger.info("Clearing old results for fresh benchmark run...")

        # Determine which benchmarks to clear
        if benchmarks:
            to_clear = benchmarks
        else:
            # Clear all benchmark directories from results
            if self.results_dir.exists():
                to_clear_results = [d.name for d in self.results_dir.iterdir() if d.is_dir()]
            else:
                to_clear_results = []

            # Clear all benchmark directories from logs
            if self.logs_dir.exists():
                to_clear_logs = [d.name for d in self.logs_dir.iterdir() if d.is_dir()]
            else:
                to_clear_logs = []

            to_clear = list(set(to_clear_results + to_clear_logs))

        # Remove benchmark CSV/table directories from results
        for benchmark_name in to_clear:
            benchmark_dir = self.results_dir / benchmark_name
            if benchmark_dir.exists():
                shutil.rmtree(benchmark_dir)
                logger.info(f"  Cleared CSV tables for: {benchmark_name}")

        # Remove benchmark JSON directories from logs
        for benchmark_name in to_clear:
            benchmark_dir = self.logs_dir / benchmark_name
            if benchmark_dir.exists():
                shutil.rmtree(benchmark_dir)
                logger.info(f"  Cleared JSON results for: {benchmark_name}")

        # Clear cross-benchmark comparison if it exists
        cross_benchmark_file = self.results_dir / "cross_benchmark_comparison.csv"
        if cross_benchmark_file.exists():
            cross_benchmark_file.unlink()
            logger.info("  Cleared cross-benchmark comparison")

        # Clear figures directory
        if self.figures_dir.exists():
            shutil.rmtree(self.figures_dir)
            logger.info("  Cleared figures directory")

        # Recreate directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Old results cleared successfully")

    def run_baseline_benchmarks(self, benchmarks: List[str] = None) -> Dict[str, List[Dict]]:
        """
        Run baseline benchmarks (NumPy).

        Args:
            benchmarks: List of specific benchmarks to run. If None, runs all.

        Returns:
            Dictionary mapping benchmark names to results
        """
        log_section(logger, "RUNNING BASELINE BENCHMARKS (NumPy)")

        all_benchmarks = self.config.get('benchmarks', {})

        # Filter to specific benchmarks if provided
        if benchmarks:
            all_benchmarks = {k: v for k, v in all_benchmarks.items() if k in benchmarks}

        baseline_results = {}

        for benchmark_name, bench_config in all_benchmarks.items():
            log_subsection(logger, f"Benchmark: {benchmark_name}")

            sizes = bench_config.get('sizes', [])
            iterations = bench_config.get('iterations', 10)
            warmup = bench_config.get('warmup', 3)

            # Use only the first size to match DSPy optimizer test size
            # This ensures direct comparison in tables
            test_size = sizes[0] if sizes else 128
            logger.info(f"Running NumPy baseline at size: {test_size} (matching DSPy test size)")

            benchmark_results = []

            # Run at single test size only (not all sizes)
            result = BaselineBenchmarks.run_benchmark(
                benchmark_name=benchmark_name,
                method="numpy",
                size=test_size,
                iterations=iterations,
                warmup=warmup
            )

            benchmark_results.append(result.to_dict())

            baseline_results[benchmark_name] = benchmark_results

            # Save baseline results
            self.save_results(benchmark_name, "numpy", benchmark_results)

        return baseline_results

    def run_dspy_optimizers(self, benchmark_name: str, size: int) -> Dict[str, Dict]:
        """
        Run all DSPy optimizers for a specific benchmark and size.

        This now includes:
        1. Code generation via DSPy optimizers
        2. Code execution and performance measurement
        3. Correctness verification

        Args:
            benchmark_name: Name of the benchmark
            size: Problem size

        Returns:
            Dictionary mapping optimizer names to results (with performance metrics)
        """
        optimizers_config = self.config.get('optimizers', {})
        trainset = get_training_examples(benchmark_name)

        # Get benchmark config for iterations/warmup
        benchmark_config = self.config.get('benchmarks', {}).get(benchmark_name, {})
        iterations = benchmark_config.get('iterations', 10)
        warmup = benchmark_config.get('warmup', 3)

        # Create code executor
        code_executor = CodeExecutor(iterations=iterations, warmup=warmup)

        results = {}

        # Baseline (no optimization)
        if optimizers_config.get('baseline', {}).get('enabled', True):
            log_subsection(logger, f"Running Baseline Optimizer (size={size})")
            try:
                result = run_baseline_optimizer(
                    benchmark_name=benchmark_name,
                    size=size,
                    trainset=trainset
                )

                # Execute and measure generated code
                exec_result = code_executor.execute_code(
                    result['generated_code'],
                    benchmark_name,
                    size
                )

                # Add performance metrics to result
                result.update({
                    'mean_time_ms': exec_result.mean_time_ms,
                    'std_time_ms': exec_result.std_time_ms,
                    'min_time_ms': exec_result.min_time_ms,
                    'max_time_ms': exec_result.max_time_ms,
                    'execution_success': exec_result.success,
                    'correctness_verified': exec_result.correctness_verified,
                    'execution_error': exec_result.error_message if not exec_result.success else None
                })

                # Log execution summary
                summary = (
                    f"\n{'─' * 80}\n"
                    f"BASELINE OPTIMIZER - EXECUTION SUMMARY\n"
                    f"{'─' * 80}\n"
                    f"  ✓ Code generated successfully\n"
                    f"  {'✓' if exec_result.success else '✗'} Code execution: {'SUCCESS' if exec_result.success else 'FAILED'}\n"
                    f"  {'✓' if exec_result.correctness_verified else '✗'} Correctness verification: {'PASSED' if exec_result.correctness_verified else 'FAILED'}\n"
                    f"  ⏱  Performance: {exec_result.mean_time_ms:.4f} ± {exec_result.std_time_ms:.4f} ms\n"
                )

                if not exec_result.success and exec_result.error_message:
                    # Show first line of error message in summary
                    error_first_line = exec_result.error_message.split('\n')[0]
                    summary += f"  ⚠  Error: {error_first_line}\n"

                summary += f"{'─' * 80}"
                logger.info(summary)

                results['baseline'] = result
            except Exception as e:
                logger.error(f"Baseline optimizer failed: {e}")

        # Bootstrap
        if optimizers_config.get('bootstrap', {}).get('enabled', True):
            log_subsection(logger, f"Running Bootstrap Optimizer (size={size})")
            try:
                max_demos = optimizers_config['bootstrap'].get('max_bootstrapped_demos', 4)
                result = run_bootstrap_optimizer(
                    benchmark_name=benchmark_name,
                    size=size,
                    trainset=trainset,
                    max_bootstrapped_demos=max_demos,
                    use_performance_metric=True,  # Enable performance-aware optimization
                    performance_weight=0.5  # Balance between quality and performance
                )

                # Execute and measure generated code
                exec_result = code_executor.execute_code(
                    result['generated_code'],
                    benchmark_name,
                    size
                )

                # Add performance metrics to result
                result.update({
                    'mean_time_ms': exec_result.mean_time_ms,
                    'std_time_ms': exec_result.std_time_ms,
                    'min_time_ms': exec_result.min_time_ms,
                    'max_time_ms': exec_result.max_time_ms,
                    'execution_success': exec_result.success,
                    'correctness_verified': exec_result.correctness_verified,
                    'execution_error': exec_result.error_message if not exec_result.success else None
                })

                # Log execution summary
                summary = (
                    f"\n{'─' * 80}\n"
                    f"BOOTSTRAP OPTIMIZER - EXECUTION SUMMARY\n"
                    f"{'─' * 80}\n"
                    f"  ✓ Code generated successfully\n"
                    f"  {'✓' if exec_result.success else '✗'} Code execution: {'SUCCESS' if exec_result.success else 'FAILED'}\n"
                    f"  {'✓' if exec_result.correctness_verified else '✗'} Correctness verification: {'PASSED' if exec_result.correctness_verified else 'FAILED'}\n"
                    f"  ⏱  Performance: {exec_result.mean_time_ms:.4f} ± {exec_result.std_time_ms:.4f} ms\n"
                )

                if not exec_result.success and exec_result.error_message:
                    # Show first line of error message in summary
                    error_first_line = exec_result.error_message.split('\n')[0]
                    summary += f"  ⚠  Error: {error_first_line}\n"

                summary += f"{'─' * 80}"
                logger.info(summary)

                results['bootstrap'] = result
            except Exception as e:
                logger.error(f"Bootstrap optimizer failed: {e}")

        # MIPROv2
        if optimizers_config.get('miprov2', {}).get('enabled', True):
            log_subsection(logger, f"Running MIPROv2 Optimizer (size={size})")
            try:
                num_candidates = optimizers_config['miprov2'].get('num_candidates', 10)
                result = run_miprov2_optimizer(
                    benchmark_name=benchmark_name,
                    size=size,
                    trainset=trainset,
                    num_candidates=num_candidates,
                    num_trials=10,
                    use_performance_metric=True,  # Enable performance-aware optimization
                    performance_weight=0.5  # Balance between quality and performance
                )

                # Execute and measure generated code
                exec_result = code_executor.execute_code(
                    result['generated_code'],
                    benchmark_name,
                    size
                )

                # Add performance metrics to result
                result.update({
                    'mean_time_ms': exec_result.mean_time_ms,
                    'std_time_ms': exec_result.std_time_ms,
                    'min_time_ms': exec_result.min_time_ms,
                    'max_time_ms': exec_result.max_time_ms,
                    'execution_success': exec_result.success,
                    'correctness_verified': exec_result.correctness_verified,
                    'execution_error': exec_result.error_message if not exec_result.success else None
                })

                # Log execution summary
                summary = (
                    f"\n{'─' * 80}\n"
                    f"MIPROV2 OPTIMIZER - EXECUTION SUMMARY\n"
                    f"{'─' * 80}\n"
                    f"  ✓ Code generated successfully\n"
                    f"  {'✓' if exec_result.success else '✗'} Code execution: {'SUCCESS' if exec_result.success else 'FAILED'}\n"
                    f"  {'✓' if exec_result.correctness_verified else '✗'} Correctness verification: {'PASSED' if exec_result.correctness_verified else 'FAILED'}\n"
                    f"  ⏱  Performance: {exec_result.mean_time_ms:.4f} ± {exec_result.std_time_ms:.4f} ms\n"
                )

                if not exec_result.success and exec_result.error_message:
                    # Show first line of error message in summary
                    error_first_line = exec_result.error_message.split('\n')[0]
                    summary += f"  ⚠  Error: {error_first_line}\n"

                summary += f"{'─' * 80}"
                logger.info(summary)

                results['miprov2'] = result
            except Exception as e:
                logger.error(f"MIPROv2 optimizer failed: {e}")

        return results

    def save_results(self, benchmark_name: str, optimizer_name: str, results: Any):
        """
        Save results to JSON file in logs directory.

        Args:
            benchmark_name: Name of the benchmark
            optimizer_name: Name of the optimizer
            results: Results data to save
        """
        # Save JSON results to logs/<benchmark>/
        output_dir = self.logs_dir / benchmark_name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{optimizer_name}_results.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved JSON results to {output_file}")

    def run_all(self,
                benchmarks: List[str] = None,
                optimizers: List[str] = None,
                skip_baseline: bool = False,
                skip_visualization: bool = False,
                skip_clear: bool = False):
        """
        Run complete experimental pipeline.

        Args:
            benchmarks: List of benchmark names to run (None = all)
            optimizers: List of optimizers to run (None = all)
            skip_baseline: Skip baseline benchmarks
            skip_visualization: Skip visualization generation
            skip_clear: Skip clearing old results (for when already cleared)
        """
        start_time = time.time()

        log_section(logger, "STARTING COMPLETE BENCHMARK RUN")
        logger.info(f"Start time: {datetime.now().isoformat()}")

        # Step 0: Clear old results for fresh run
        if not skip_clear:
            self.clear_old_results(benchmarks=benchmarks)

        # Step 1: Run baseline benchmarks
        if not skip_baseline:
            baseline_results = self.run_baseline_benchmarks()  # noqa: F841
        else:
            logger.info("Skipping baseline benchmarks")

        # Step 2: Configure DSPy
        self.configure_dspy()

        # Step 3: Run DSPy optimizers for each benchmark
        benchmark_configs = self.config.get('benchmarks', {})

        if benchmarks:
            # Filter to specified benchmarks
            benchmark_configs = {k: v for k, v in benchmark_configs.items() if k in benchmarks}

        for benchmark_name, bench_config in benchmark_configs.items():
            log_section(logger, f"BENCHMARK: {benchmark_name.upper()}")

            sizes = bench_config.get('sizes', [])

            # For full experiments, run all sizes
            # For quick testing, you might want to run just one size
            test_size = sizes[0] if sizes else 128

            logger.info(f"Running with test size: {test_size}")

            # Run all optimizers
            dspy_results = self.run_dspy_optimizers(benchmark_name, test_size)

            # Save individual optimizer results
            for optimizer_name, result in dspy_results.items():
                self.save_results(benchmark_name, optimizer_name, result)

        # Step 4: Generate visualizations
        if not skip_visualization:
            log_section(logger, "GENERATING VISUALIZATIONS")
            visualizer = ResultsVisualizer(
                results_dir=str(self.logs_dir),  # Read JSON from logs
                figures_dir=str(self.figures_dir)
            )
            visualizer.visualize_all(list(benchmark_configs.keys()))
        else:
            logger.info("Skipping visualization generation")

        # Step 5: Generate comparison tables
        log_section(logger, "GENERATING COMPARISON TABLES")
        table_generator = TableGenerator(results_dir=str(self.results_dir))

        benchmark_list = list(benchmark_configs.keys())

        # Generate individual tables for each benchmark
        for benchmark_name in benchmark_list:
            logger.info(f"Generating tables for {benchmark_name}")
            table_generator.generate_all_tables(benchmark_name)

        # Generate cross-benchmark comparison if multiple benchmarks
        if len(benchmark_list) > 1:
            logger.info("Generating cross-benchmark comparison table")
            table_generator.generate_cross_benchmark_table(benchmark_list)

        # Completion
        elapsed_time = time.time() - start_time
        log_section(logger, "BENCHMARK RUN COMPLETED")
        logger.info(f"Total time: {elapsed_time/60:.2f} minutes")
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info(f"Figures saved to: {self.figures_dir}")
        logger.info(f"Tables saved to: {self.results_dir}/<benchmark_name>/")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run DSPy HPC benchmarks")
    parser.add_argument('--benchmarks', nargs='+',
                       help='Benchmarks to run (default: all)',
                       choices=['matrix_multiply', 'cholesky', 'fft'])
    parser.add_argument('--optimizers', nargs='+',
                       help='Optimizers to run (default: all)',
                       choices=['baseline', 'bootstrap', 'miprov2', 'gepa'])
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline benchmarks')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--config', default='configs/benchmark_config.yaml',
                       help='Path to configuration file')

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Create and run benchmark runner
    runner = BenchmarkRunner(config_path=args.config)
    runner.run_all(
        benchmarks=args.benchmarks,
        optimizers=args.optimizers,
        skip_baseline=args.skip_baseline,
        skip_visualization=args.skip_visualization
    )


if __name__ == "__main__":
    main()
