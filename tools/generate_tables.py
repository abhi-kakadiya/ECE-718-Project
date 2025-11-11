#!/usr/bin/env python3
"""
Generate comprehensive comparison tables from benchmark results.

Creates multiple tables for different insights:
1. Performance comparison table
2. Token usage comparison table
3. Cost-benefit analysis table
4. Speedup analysis table
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


class TableGenerator:
    """Generate insightful tables from benchmark results."""

    def __init__(self, results_dir: str = "results", logs_dir: str = "logs"):
        self.results_dir = Path(results_dir)  # For CSV output
        self.logs_dir = Path(logs_dir)  # For JSON input

        # Model pricing (per 1K tokens) - matches token_tracker.py
        self.pricing = {
            # OpenAI GPT-4.1 Family
            "gpt-4.1": {
                "prompt": 0.002,      # $2.00 / 1M tokens
                "completion": 0.008   # $8.00 / 1M tokens
            },
            "gpt-4.1-mini": {
                "prompt": 0.0004,     # $0.40 / 1M tokens
                "completion": 0.0016  # $1.60 / 1M tokens
            },
            "gpt-4.1-nano": {
                "prompt": 0.0001,     # $0.10 / 1M tokens
                "completion": 0.0004  # $0.40 / 1M tokens
            },
            # OpenAI GPT-5 Mini
            "gpt-5-mini": {
                "prompt": 0.00025,    # $0.25 / 1M tokens
                "completion": 0.002   # $2.00 / 1M tokens
            },
            # Anthropic Claude Sonnet 4 Family
            "claude-sonnet-4.5": {
                "prompt": 0.003,      # $3.00 / 1M tokens
                "completion": 0.015   # $15.00 / 1M tokens
            },
            "claude-sonnet-4": {
                "prompt": 0.003,      # $3.00 / 1M tokens
                "completion": 0.015   # $15.00 / 1M tokens
            },
            # Legacy models
            "gpt-3.5-turbo": {
                "prompt": 0.0005,     # $0.50 / 1M tokens
                "completion": 0.0015  # $1.50 / 1M tokens
            },
            "gpt-4": {
                "prompt": 0.03,       # $30.00 / 1M tokens
                "completion": 0.06    # $60.00 / 1M tokens
            },
        }

    def get_model_pricing(self, model: str) -> dict:
        """
        Get pricing for a specific model.

        Args:
            model: Model name

        Returns:
            Dictionary with prompt and completion pricing per 1K tokens
        """
        if model in self.pricing:
            return self.pricing[model]
        else:
            # Default to gpt-4.1-mini if unknown
            print(f"Warning: Unknown model '{model}', using gpt-4.1-mini pricing")
            return self.pricing["gpt-4.1-mini"]

    def load_results(self, benchmark_name: str) -> Dict[str, Any]:
        """Load all results for a benchmark from logs directory."""
        # Read JSON files from logs/<benchmark>/
        benchmark_dir = self.logs_dir / benchmark_name
        if not benchmark_dir.exists():
            print(f"No results found for {benchmark_name} in {benchmark_dir}")
            return {}

        results = {}
        for result_file in benchmark_dir.glob("*_results.json"):
            optimizer_name = result_file.stem.replace("_results", "")
            with open(result_file) as f:
                results[optimizer_name] = json.load(f)

        return results

    def create_performance_table(self, benchmark_name: str) -> pd.DataFrame:
        """Create performance comparison table."""
        results = self.load_results(benchmark_name)

        data = []
        for optimizer, result in results.items():
            if optimizer == "numpy":
                # NumPy is a list of results for different sizes
                for size_result in result:
                    mean_time = size_result.get('mean_time_ms', 0)
                    std_dev = size_result.get('std_time_ms', 0)

                    # Check for high variance (coefficient of variation > 50%)
                    cv = (std_dev / mean_time * 100) if mean_time > 0 else 0
                    variance_warning = ' ⚠️' if cv > 50 else ''

                    data.append({
                        'Optimizer': 'NumPy (baseline)',
                        'Size': size_result.get('size', 'N/A'),
                        'Mean Time (ms)': f"{mean_time:.4f}",
                        'Std Dev (ms)': f"{std_dev:.4f}{variance_warning}",
                        'Min Time (ms)': f"{size_result.get('min_time_ms', 0):.4f}",
                        'Max Time (ms)': f"{size_result.get('max_time_ms', 0):.4f}",
                        'Correctness': '✓',
                        'Success': '✓'
                    })
            else:
                # DSPy optimizers
                mean_time = result.get('mean_time_ms', 0)
                std_dev = result.get('std_time_ms', 0)

                # Check for high variance (coefficient of variation > 50%)
                cv = (std_dev / mean_time * 100) if mean_time > 0 else 0
                variance_warning = ' ⚠️' if cv > 50 else ''

                data.append({
                    'Optimizer': optimizer.capitalize(),
                    'Size': 'N/A',
                    'Mean Time (ms)': f"{mean_time:.4f}" if mean_time else 'N/A',
                    'Std Dev (ms)': f"{std_dev:.4f}{variance_warning}" if std_dev else 'N/A',
                    'Min Time (ms)': f"{result.get('min_time_ms', 0):.4f}" if result.get('min_time_ms') else 'N/A',
                    'Max Time (ms)': f"{result.get('max_time_ms', 0):.4f}" if result.get('max_time_ms') else 'N/A',
                    'Correctness': '✓' if result.get('correctness_verified') else '✗',
                    'Success': '✓' if result.get('execution_success') else '✗'
                })

        df = pd.DataFrame(data)
        return df

    def create_token_usage_table(self, benchmark_name: str) -> pd.DataFrame:
        """Create token usage comparison table."""
        results = self.load_results(benchmark_name)

        data = []
        for optimizer, result in results.items():
            if optimizer == "numpy":
                continue  # Skip NumPy for token usage

            token_usage = result.get('token_usage', {})

            # Get token breakdown from token_usage
            # token_usage has structure: {compilation: {prompt_tokens, completion_tokens, ...}, inference: {...}, ...}
            compilation_data = token_usage.get('compilation', {})
            inference_data = token_usage.get('inference', {})

            compilation_prompt = compilation_data.get('prompt_tokens', 0)
            compilation_completion = compilation_data.get('completion_tokens', 0)
            compilation_tokens = compilation_data.get('total_tokens', 0)

            inference_prompt = inference_data.get('prompt_tokens', 0)
            inference_completion = inference_data.get('completion_tokens', 0)
            inference_tokens = inference_data.get('total_tokens', 0)

            total_tokens = token_usage.get('total_tokens', 0)
            total_api_calls = token_usage.get('total_calls', 0)

            # Get model-specific pricing
            model = token_usage.get('model', 'gpt-4.1-mini')
            pricing = self.get_model_pricing(model)

            # Calculate cost using model-specific pricing (pricing is per 1K tokens)
            compilation_cost = (compilation_prompt * pricing['prompt'] +
                               compilation_completion * pricing['completion']) / 1000
            inference_cost = (inference_prompt * pricing['prompt'] +
                             inference_completion * pricing['completion']) / 1000
            total_cost = compilation_cost + inference_cost

            data.append({
                'Optimizer': optimizer.capitalize(),
                'Model': model,  # Show which model was used
                'Compilation Tokens': f"{compilation_tokens:,}",
                'Inference Tokens': f"{inference_tokens:,}",
                'Total Tokens': f"{total_tokens:,}",
                'Compilation Cost': f"${compilation_cost:.6f}",
                'Inference Cost': f"${inference_cost:.6f}",
                'Total Cost': f"${total_cost:.6f}",
                'API Calls': total_api_calls,
                'Compile Time (s)': f"{result.get('compilation_time_seconds', 0):.2f}"
            })

        df = pd.DataFrame(data)
        return df

    def create_cost_benefit_table(self, benchmark_name: str) -> pd.DataFrame:
        """Create cost-benefit analysis table."""
        results = self.load_results(benchmark_name)

        # Get NumPy baseline performance
        numpy_time = None
        if 'numpy' in results and len(results['numpy']) > 0:
            numpy_time = results['numpy'][0].get('mean_time_ms', 1.0)

        data = []
        for optimizer, result in results.items():
            if optimizer == "numpy":
                continue

            mean_time = result.get('mean_time_ms', 0)
            token_usage = result.get('token_usage', {})
            total_tokens = token_usage.get('total_tokens', 0)

            # Calculate speedup vs NumPy
            speedup = numpy_time / mean_time if mean_time > 0 and numpy_time else 0

            # Calculate actual cost using proper token breakdown and model-specific pricing
            compilation_data = token_usage.get('compilation', {})
            inference_data = token_usage.get('inference', {})

            compilation_prompt = compilation_data.get('prompt_tokens', 0)
            compilation_completion = compilation_data.get('completion_tokens', 0)
            inference_prompt = inference_data.get('prompt_tokens', 0)
            inference_completion = inference_data.get('completion_tokens', 0)

            # Get model-specific pricing
            model = token_usage.get('model', 'gpt-4.1-mini')
            pricing = self.get_model_pricing(model)

            # Calculate cost using model-specific pricing (pricing is per 1K tokens)
            compilation_cost = (compilation_prompt * pricing['prompt'] +
                               compilation_completion * pricing['completion']) / 1000
            inference_cost = (inference_prompt * pricing['prompt'] +
                             inference_completion * pricing['completion']) / 1000
            estimated_cost = compilation_cost + inference_cost

            data.append({
                'Optimizer': optimizer.capitalize(),
                'Model': model,  # Show which model was used
                'Performance (ms)': f"{mean_time:.4f}" if mean_time > 0 else 'N/A',
                'Speedup vs NumPy': f"{speedup:.2f}x" if speedup > 0 else 'N/A',
                'Total Tokens': f"{total_tokens:,}",
                'Est. Cost ($)': f"${estimated_cost:.6f}",
                'Success': '✓' if result.get('execution_success') and result.get('correctness_verified') else '✗'
            })

        df = pd.DataFrame(data)
        return df

    def create_speedup_table(self, benchmark_name: str) -> pd.DataFrame:
        """Create speedup analysis table."""
        results = self.load_results(benchmark_name)

        # Get NumPy baseline performance
        numpy_time = None
        if 'numpy' in results and len(results['numpy']) > 0:
            numpy_time = results['numpy'][0].get('mean_time_ms', 1.0)

        data = []
        for optimizer, result in results.items():
            if optimizer == "numpy":
                data.append({
                    'Optimizer': 'NumPy (baseline)',
                    'Mean Time (ms)': f"{numpy_time:.4f}",
                    'Speedup': '1.00x (baseline)',
                    'Performance': '100%',
                    'Status': 'Reference'
                })
                continue

            mean_time = result.get('mean_time_ms', 0)
            if mean_time > 0 and numpy_time:
                speedup = numpy_time / mean_time
                performance_pct = speedup * 100

                if speedup >= 0.8:
                    status = '🟢 Excellent'
                elif speedup >= 0.5:
                    status = '🟡 Good'
                else:
                    status = '🔴 Needs Work'
            else:
                speedup = 0
                performance_pct = 0
                status = '⚫ Failed'

            data.append({
                'Optimizer': optimizer.capitalize(),
                'Mean Time (ms)': f"{mean_time:.4f}" if mean_time > 0 else 'N/A',
                'Speedup': f"{speedup:.2f}x" if speedup > 0 else 'N/A',
                'Performance': f"{performance_pct:.1f}%" if speedup > 0 else 'N/A',
                'Status': status
            })

        df = pd.DataFrame(data)
        return df

    def create_comparison_summary(self, benchmark_name: str) -> pd.DataFrame:
        """Create overall comparison summary."""
        results = self.load_results(benchmark_name)

        # Get NumPy baseline
        numpy_time = None
        if 'numpy' in results and len(results['numpy']) > 0:
            numpy_time = results['numpy'][0].get('mean_time_ms', 1.0)

        data = []

        # Add NumPy first (reference baseline)
        if 'numpy' in results and len(results['numpy']) > 0:
            data.append({
                'Optimizer': 'NUMPY',
                'Execution': '✓',
                'Correct': '✓',
                'Time (ms)': f"{numpy_time:.4f}",
                'Speedup': '1.00x (baseline)',
                'Tokens': 'N/A',
                'Compile (s)': 'N/A',
                'Overall': '⚪ Reference'
            })

        # Add DSPy optimizers
        for optimizer, result in results.items():
            if optimizer == "numpy":
                continue

            mean_time = result.get('mean_time_ms', 0)
            token_usage = result.get('token_usage', {})
            total_tokens = token_usage.get('total_tokens', 0)
            compilation_time = result.get('compilation_time_seconds', 0)

            speedup = numpy_time / mean_time if mean_time > 0 and numpy_time else 0

            data.append({
                'Optimizer': optimizer.upper(),
                'Execution': '✓' if result.get('execution_success') else '✗',
                'Correct': '✓' if result.get('correctness_verified') else '✗',
                'Time (ms)': f"{mean_time:.4f}" if mean_time > 0 else 'N/A',
                'Speedup': f"{speedup:.2f}x" if speedup > 0 else 'N/A',
                'Tokens': f"{total_tokens:,}",
                'Compile (s)': f"{compilation_time:.2f}",
                'Overall': '🟢' if result.get('execution_success') and result.get('correctness_verified') and speedup >= 0.8 else
                           '🟡' if result.get('execution_success') and result.get('correctness_verified') and speedup >= 0.5 else
                           '🔴'
            })

        df = pd.DataFrame(data)
        return df

    def generate_all_tables(self, benchmark_name: str, output_dir: str = None):
        """Generate all tables for a benchmark."""
        if output_dir is None:
            output_dir = self.results_dir / benchmark_name

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 80}")
        print(f"GENERATING TABLES FOR: {benchmark_name.upper()}")
        print(f"{'=' * 80}\n")

        # 1. Comparison Summary
        print("1. COMPARISON SUMMARY")
        print("-" * 80)
        summary_df = self.create_comparison_summary(benchmark_name)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(output_path / "comparison_summary.csv", index=False)
        print(f"\n✓ Saved to: {output_path / 'comparison_summary.csv'}\n")

        # 2. Performance Table
        print("2. PERFORMANCE DETAILS")
        print("-" * 80)
        perf_df = self.create_performance_table(benchmark_name)
        print(perf_df.to_string(index=False))
        perf_df.to_csv(output_path / "performance_details.csv", index=False)
        print(f"\n✓ Saved to: {output_path / 'performance_details.csv'}\n")

        # 3. Token Usage Table
        print("3. TOKEN USAGE")
        print("-" * 80)
        token_df = self.create_token_usage_table(benchmark_name)
        print(token_df.to_string(index=False))
        token_df.to_csv(output_path / "token_usage.csv", index=False)
        print(f"\n✓ Saved to: {output_path / 'token_usage.csv'}\n")

        # 4. Cost-Benefit Table
        print("4. COST-BENEFIT ANALYSIS")
        print("-" * 80)
        cost_df = self.create_cost_benefit_table(benchmark_name)
        print(cost_df.to_string(index=False))
        cost_df.to_csv(output_path / "cost_benefit.csv", index=False)
        print(f"\n✓ Saved to: {output_path / 'cost_benefit.csv'}\n")

        # 5. Speedup Table
        print("5. SPEEDUP ANALYSIS")
        print("-" * 80)
        speedup_df = self.create_speedup_table(benchmark_name)
        print(speedup_df.to_string(index=False))
        speedup_df.to_csv(output_path / "speedup_analysis.csv", index=False)
        print(f"\n✓ Saved to: {output_path / 'speedup_analysis.csv'}\n")

        print(f"{'=' * 80}")
        print("ALL TABLES GENERATED SUCCESSFULLY")
        print(f"{'=' * 80}\n")

    def create_cross_benchmark_comparison(self, benchmarks: List[str]) -> pd.DataFrame:
        """Create master comparison table across all benchmarks."""
        data = []

        for benchmark_name in benchmarks:
            results = self.load_results(benchmark_name)
            if not results:
                continue

            row = {'Benchmark': benchmark_name.replace('_', ' ').title()}

            # Get NumPy baseline
            if 'numpy' in results and len(results['numpy']) > 0:
                numpy_time = results['numpy'][0].get('mean_time_ms', 0)
                row['NumPy (ms)'] = f"{numpy_time:.4f}"
            else:
                row['NumPy (ms)'] = 'N/A'
                numpy_time = None

            # Get DSPy optimizer results
            optimizer_times = {}
            for optimizer in ['baseline', 'bootstrap', 'miprov2']:
                if optimizer in results:
                    mean_time = results[optimizer].get('mean_time_ms', 0)
                    success = results[optimizer].get('execution_success', False)
                    correct = results[optimizer].get('correctness_verified', False)

                    if success and correct and mean_time > 0:
                        row[f'{optimizer.capitalize()} (ms)'] = f"{mean_time:.4f}"
                        optimizer_times[optimizer] = mean_time
                    else:
                        row[f'{optimizer.capitalize()} (ms)'] = '✗ Failed'
                        optimizer_times[optimizer] = float('inf')
                else:
                    row[f'{optimizer.capitalize()} (ms)'] = 'N/A'
                    optimizer_times[optimizer] = float('inf')

            # Determine best optimizer (including NumPy)
            all_times = {}
            if numpy_time is not None:
                all_times['NumPy'] = numpy_time
            all_times.update(optimizer_times)

            if all_times:
                best_optimizer = min(all_times.items(), key=lambda x: x[1])
                if best_optimizer[1] != float('inf'):
                    row['Best'] = best_optimizer[0].capitalize()

                    # Calculate speedup vs NumPy
                    if numpy_time and best_optimizer[1] > 0:
                        speedup = numpy_time / best_optimizer[1]
                        row['Speedup vs NumPy'] = f"{speedup:.2f}x"
                    else:
                        row['Speedup vs NumPy'] = 'N/A'
                else:
                    row['Best'] = 'None'
                    row['Speedup vs NumPy'] = 'N/A'
            else:
                row['Best'] = 'N/A'
                row['Speedup vs NumPy'] = 'N/A'

            data.append(row)

        df = pd.DataFrame(data)
        return df

    def generate_cross_benchmark_table(self, benchmarks: List[str], output_dir: str = "results"):
        """Generate cross-benchmark comparison table."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 80}")
        print("GENERATING CROSS-BENCHMARK COMPARISON")
        print(f"{'=' * 80}\n")

        df = self.create_cross_benchmark_comparison(benchmarks)

        print("CROSS-BENCHMARK COMPARISON")
        print("-" * 80)
        print(df.to_string(index=False))

        # Save to CSV
        csv_path = output_path / "cross_benchmark_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved to: {csv_path}\n")

        print(f"{'=' * 80}")
        print("CROSS-BENCHMARK TABLE GENERATED SUCCESSFULLY")
        print(f"{'=' * 80}\n")


def main():
    """Generate tables for all available benchmarks."""
    import sys

    generator = TableGenerator()

    # Check if benchmark name provided
    if len(sys.argv) > 1:
        benchmark_name = sys.argv[1]
        print(f"\nGenerating tables for single benchmark: {benchmark_name}\n")
        generator.generate_all_tables(benchmark_name)
    else:
        # Generate for all benchmarks - look in logs directory for JSON results
        logs_dir = Path("logs")
        if not logs_dir.exists():
            print("No logs directory found!")
            return

        benchmarks = [d.name for d in logs_dir.iterdir() if d.is_dir()]

        if not benchmarks:
            print("No benchmark results found in logs directory!")
            return

        print(f"\nGenerating tables for {len(benchmarks)} benchmarks\n")

        # Generate individual tables for each benchmark
        for benchmark in benchmarks:
            generator.generate_all_tables(benchmark)
            print("\n")

        # Generate cross-benchmark comparison table
        if len(benchmarks) > 1:
            generator.generate_cross_benchmark_table(benchmarks)


if __name__ == "__main__":
    main()
