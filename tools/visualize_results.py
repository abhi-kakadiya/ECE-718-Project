"""
Visualization tools for benchmark results.

Creates comprehensive visualizations including:
- Performance comparison charts
- Token usage analysis
- Speedup graphs
- Summary tables
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict, List
import seaborn as sns

from tools.logger import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.size'] = 10


class ResultsVisualizer:
    """Visualization system for benchmark results."""

    def __init__(self, results_dir: str = "logs", figures_dir: str = "figures"):
        """
        Initialize visualizer.

        Args:
            results_dir: Directory containing result JSON files (default: logs)
            figures_dir: Directory to save generated figures
        """
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"\n{'=' * 80}\n"
            f"RESULTS VISUALIZER INITIALIZED\n"
            f"{'=' * 80}\n"
            f"  JSON Results:  {self.results_dir}\n"
            f"  Figures dir:   {self.figures_dir}\n"
            f"{'=' * 80}"
        )

    def load_results(self, benchmark_name: str) -> Dict[str, Dict]:
        """
        Load all results for a benchmark.

        Args:
            benchmark_name: Name of the benchmark

        Returns:
            Dictionary mapping optimizer names to results

        Note:
            This loads TWO types of results:
            - NumPy baseline: List of performance measurements
            - DSPy optimizers: Dict with generated code and token usage
        """
        logger.info(f"Loading results for {benchmark_name}...")

        benchmark_dir = self.results_dir / benchmark_name
        if not benchmark_dir.exists():
            logger.warning(f"No results directory found for {benchmark_name}")
            return {}

        results = {}
        for json_file in benchmark_dir.glob("*.json"):
            optimizer_name = json_file.stem.replace("_results", "")
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    results[optimizer_name] = data
                logger.debug(f"  Loaded {optimizer_name} results")
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")

        logger.info(f"Loaded {len(results)} result files")
        return results

    def plot_performance_comparison(self,
                                   benchmark_name: str,
                                   results: Dict[str, Dict],
                                   baseline_time: float = None):
        """
        Plot performance comparison chart (ALL optimizers + NumPy).

        Args:
            benchmark_name: Name of the benchmark
            results: Dictionary of results
            baseline_time: Baseline execution time (NumPy)

        Note:
            Now compares DSPy-generated code performance against NumPy baseline!
        """
        logger.info(f"Creating performance comparison plot for {benchmark_name}")

        fig, ax = plt.subplots(figsize=(12, 7))

        # Get all DSPy optimizer results with performance metrics
        dspy_results = {
            name: data for name, data in results.items()
            if isinstance(data, dict) and 'mean_time_ms' in data and data.get('execution_success', False)
        }

        # Get NumPy baseline (use first size for comparison)
        numpy_results = results.get('numpy', [])
        numpy_time = None
        if numpy_results and isinstance(numpy_results, list) and len(numpy_results) > 0:
            numpy_time = numpy_results[0]['mean_time_ms']

        if not dspy_results and not numpy_time:
            logger.warning(f"No performance results found for {benchmark_name}, skipping plot")
            return

        # Prepare data for plotting
        optimizers = list(dspy_results.keys())
        if numpy_time:
            optimizers.append('numpy')

        times = [dspy_results[opt]['mean_time_ms'] for opt in dspy_results.keys()]
        stds = [dspy_results[opt].get('std_time_ms', 0) for opt in dspy_results.keys()]

        if numpy_time:
            times.append(numpy_time)
            numpy_std = numpy_results[0].get('std_time_ms', 0) if numpy_results else 0
            stds.append(numpy_std)

        # Create bar plot
        bars = ax.bar(optimizers, times, yerr=stds, capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Color bars
        colors = {
            'baseline': '#e74c3c',
            'bootstrap': '#3498db',
            'miprov2': '#2ecc71',
            'numpy': '#95a5a6'
        }

        for bar, opt in zip(bars, optimizers):
            bar.set_color(colors.get(opt, '#34495e'))

        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_val:.2f}ms',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Labels and formatting
        ax.set_ylabel('Execution Time (ms)', fontsize=13, fontweight='bold')
        ax.set_title(f'{benchmark_name.replace("_", " ").title()} - Performance Comparison\n(All Optimizers vs NumPy)',
                    fontsize=15, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()

        # Save figure
        output_file = self.figures_dir / f"{benchmark_name}_performance_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved performance comparison plot to {output_file}")

    def plot_token_usage(self, benchmark_name: str, results: Dict[str, Dict]):
        """
        Plot token usage comparison (DSPy optimizers only).

        Args:
            benchmark_name: Name of the benchmark
            results: Dictionary of results

        Note:
            Only plots DSPy optimizer token usage.
            NumPy baseline doesn't use LLM (no tokens).
        """
        logger.info(f"Creating token usage plot for {benchmark_name}")

        # Filter: Only process DSPy results (dict with token_usage key)
        dspy_results = {
            name: data for name, data in results.items()
            if isinstance(data, dict) and 'token_usage' in data
        }

        if not dspy_results:
            logger.warning(f"No DSPy results with token usage found for {benchmark_name}, skipping token plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        optimizers = list(dspy_results.keys())
        compilation_tokens = []
        inference_tokens = []

        for opt in optimizers:
            token_data = dspy_results[opt].get('token_usage', {})
            comp = token_data.get('compilation', {}).get('total_tokens', 0)
            inf = token_data.get('inference', {}).get('total_tokens', 0)
            compilation_tokens.append(comp)
            inference_tokens.append(inf)

        # Stacked bar chart
        x = np.arange(len(optimizers))
        width = 0.6

        ax1.bar(x, compilation_tokens, width, label='Compilation', alpha=0.8)
        ax1.bar(x, inference_tokens, width, bottom=compilation_tokens,
               label='Inference', alpha=0.8)

        ax1.set_ylabel('Total Tokens')
        ax1.set_title('Token Usage by Phase')
        ax1.set_xticks(x)
        ax1.set_xticklabels(optimizers, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Total tokens comparison
        total_tokens = [c + i for c, i in zip(compilation_tokens, inference_tokens)]
        bars = ax2.bar(optimizers, total_tokens, alpha=0.7)

        # Color bars
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        for bar, color in zip(bars, colors[:len(bars)]):
            bar.set_color(color)

        ax2.set_ylabel('Total Tokens')
        ax2.set_title('Total Token Usage')
        ax2.set_xticklabels(optimizers, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)

        # Save figure
        output_path = self.figures_dir / f"{benchmark_name}_tokens.png"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Saved token usage plot to {output_path}")

    def plot_speedup_chart(self, benchmark_name: str, results: Dict[str, Dict],
                          baseline_name: str = 'numpy'):
        """
        Plot speedup relative to baseline (NOW ENABLED!).

        Args:
            benchmark_name: Name of the benchmark
            results: Dictionary of results
            baseline_name: Name of baseline to compare against

        Note:
            Shows how generated code performs relative to NumPy baseline.
            Speedup > 1.0 means faster than NumPy (rare but possible!)
            Speedup < 1.0 means slower than NumPy (common for LLM-generated code)
        """
        logger.info(f"Creating speedup chart for {benchmark_name}")

        # Get NumPy baseline time (use first size)
        numpy_results = results.get('numpy', [])
        if not numpy_results or not isinstance(numpy_results, list) or len(numpy_results) == 0:
            logger.warning(f"No NumPy baseline found for {benchmark_name}, skipping speedup chart")
            return

        baseline_time = numpy_results[0]['mean_time_ms']

        # Get DSPy results with performance metrics
        dspy_results = {
            name: data for name, data in results.items()
            if isinstance(data, dict) and 'mean_time_ms' in data and data.get('execution_success', False)
        }

        if not dspy_results:
            logger.warning(f"No DSPy performance results for {benchmark_name}, skipping speedup chart")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate speedups (NumPy time / Generated code time)
        optimizers = list(dspy_results.keys())
        speedups = [baseline_time / dspy_results[opt]['mean_time_ms'] for opt in optimizers]

        # Create bar plot
        bars = ax.bar(optimizers, speedups, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Color based on speedup (green if >= 0.8, yellow if >= 0.5, red otherwise)
        for bar, speedup in zip(bars, speedups):
            if speedup >= 0.8:
                bar.set_color('#2ecc71')  # Green - good!
            elif speedup >= 0.5:
                bar.set_color('#f39c12')  # Orange - okay
            else:
                bar.set_color('#e74c3c')  # Red - slow

        # Add horizontal line at 1.0 (same as NumPy)
        ax.axhline(y=1.0, color='#95a5a6', linestyle='--', linewidth=2, label='NumPy Baseline', zorder=0)

        # Add value labels on bars
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{speedup:.2f}x',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Labels and formatting
        ax.set_ylabel('Speedup (relative to NumPy)', fontsize=12, fontweight='bold')
        ax.set_title(f'{benchmark_name.replace("_", " ").title()} - Speedup Analysis\n(Higher is Better, 1.0 = Same as NumPy)',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.legend()

        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()

        # Save figure
        output_file = self.figures_dir / f"{benchmark_name}_speedup.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved speedup chart to {output_file}")

        if baseline_name not in results:
            logger.warning(f"Baseline '{baseline_name}' not found in results")
            return

        # Handle both list (NumPy) and dict (DSPy) results
        numpy_results = results[baseline_name]
        if isinstance(numpy_results, list):
            if len(numpy_results) == 0:
                logger.warning(f"No NumPy results found for {benchmark_name}")
                return
            baseline_time = numpy_results[0].get('mean_time_ms', 1.0)
        else:
            baseline_time = numpy_results.get('mean_time_ms', 1.0)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Only include DSPy optimizers with valid performance data
        optimizers = []
        speedups = []

        for opt in results.keys():
            if opt == baseline_name:
                continue

            opt_data = results[opt]
            # Skip if it's a list or doesn't have performance data
            if isinstance(opt_data, list) or 'mean_time_ms' not in opt_data:
                continue

            opt_time = opt_data.get('mean_time_ms', baseline_time)
            if opt_time > 0:  # Avoid division by zero
                speedup = baseline_time / opt_time
                optimizers.append(opt)
                speedups.append(speedup)

        # Create bar plot
        bars = ax.bar(optimizers, speedups, alpha=0.7)

        # Color bars (green if > 1.0, red if < 1.0)
        for bar, speedup in zip(bars, speedups):
            if speedup >= 1.0:
                bar.set_color('#2ecc71')
            else:
                bar.set_color('#e74c3c')

        # Add 1.0 line
        ax.axhline(y=1.0, color='gray', linestyle='--', label='Baseline')

        ax.set_ylabel('Speedup vs NumPy')
        ax.set_title(f'Speedup Comparison: {benchmark_name.replace("_", " ").title()}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')

        # Save figure
        output_path = self.figures_dir / f"{benchmark_name}_speedup.png"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Saved speedup chart to {output_path}")

    def create_summary_table(self, benchmark_name: str, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Create summary table of DSPy optimizer results (token usage only).

        Args:
            benchmark_name: Name of the benchmark
            results: Dictionary of results

        Returns:
            Pandas DataFrame with token usage summary

        Note:
            Only shows DSPy optimizer token usage.
            NumPy baseline doesn't use LLM (no tokens).
        """
        logger.info(f"Creating token usage summary table for {benchmark_name}")

        # Filter: Only process DSPy results
        dspy_results = {
            name: data for name, data in results.items()
            if isinstance(data, dict) and 'token_usage' in data
        }

        if not dspy_results:
            logger.warning(f"No DSPy results found for {benchmark_name}")
            return pd.DataFrame()

        data = []

        for optimizer, result in dspy_results.items():
            token_data = result.get('token_usage', {})
            row = {
                'Optimizer': optimizer,
                'Mean Time (ms)': f"{result.get('mean_time_ms', 0):.2f}" if result.get('mean_time_ms') else 'N/A',
                'Std Dev (ms)': f"{result.get('std_time_ms', 0):.2f}" if result.get('std_time_ms') else 'N/A',
                'Correctness': '✓' if result.get('correctness_verified', False) else '✗',
                'Compilation Tokens': token_data.get('compilation', {}).get('total_tokens', 0),
                'Inference Tokens': token_data.get('inference', {}).get('total_tokens', 0),
                'Total Tokens': token_data.get('total_tokens', 0),
                'Compilation Time (s)': f"{result.get('compilation_time_seconds', 0):.2f}",
            }
            data.append(row)

        df = pd.DataFrame(data)

        # Save to CSV
        output_path = self.results_dir / benchmark_name / "summary_table.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved summary table to {output_path}")

        return df

    def create_combined_visualization(self, benchmarks: List[str]):
        """
        Create simple combined visualization - skip if problematic.

        Args:
            benchmarks: List of benchmark names
        """
        logger.info("Skipping combined visualization (individual charts are sufficient)")
        return

    def visualize_all(self, benchmarks: List[str] = None):
        """
        Create all visualizations for all benchmarks.

        Args:
            benchmarks: List of benchmark names (None = auto-detect)
        """
        if benchmarks is None:
            # Auto-detect benchmarks from results directory
            benchmarks = [d.name for d in self.results_dir.iterdir() if d.is_dir()]

        logger.info(f"\n{'='*80}")
        logger.info(f"Creating visualizations for {len(benchmarks)} benchmarks")
        logger.info(f"{'='*80}\n")

        for benchmark in benchmarks:
            logger.info(f"\nProcessing {benchmark}...")

            # Load results
            results = self.load_results(benchmark)

            if not results:
                logger.warning(f"No results found for {benchmark}, skipping")
                continue

            # Create individual plots
            self.plot_performance_comparison(benchmark, results)
            self.plot_token_usage(benchmark, results)
            self.plot_speedup_chart(benchmark, results)

            # Create summary table
            summary_df = self.create_summary_table(benchmark, results)
            logger.info(f"\nSummary for {benchmark}:")
            logger.info(f"\n{summary_df.to_string(index=False)}\n")

        # Create combined visualization
        if len(benchmarks) > 1:
            self.create_combined_visualization(benchmarks)

        logger.info(f"\n{'='*80}")
        logger.info("All visualizations created successfully!")
        logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    # Test visualizations with mock data
    logger.info("Testing visualization system...")

    # Create mock results
    mock_results_dir = Path("results/test_benchmark")
    mock_results_dir.mkdir(parents=True, exist_ok=True)

    mock_data = {
        "baseline": {"mean_time_ms": 85.4, "std_time_ms": 3.1, "min_time_ms": 82.0, "max_time_ms": 90.0,
                    "token_usage": {"compilation": {"total_tokens": 0}, "inference": {"total_tokens": 850}, "total_tokens": 850}},
        "bootstrap": {"mean_time_ms": 52.1, "std_time_ms": 2.3, "min_time_ms": 49.0, "max_time_ms": 55.0,
                     "token_usage": {"compilation": {"total_tokens": 2340}, "inference": {"total_tokens": 1420}, "total_tokens": 3760}},
        "miprov2": {"mean_time_ms": 45.2, "std_time_ms": 2.1, "min_time_ms": 42.0, "max_time_ms": 48.0,
                   "token_usage": {"compilation": {"total_tokens": 8920}, "inference": {"total_tokens": 1270}, "total_tokens": 10190}},
        "numpy": {"mean_time_ms": 38.7, "std_time_ms": 1.2, "min_time_ms": 37.0, "max_time_ms": 40.0}
    }

    for opt, data in mock_data.items():
        filepath = mock_results_dir / f"{opt}_results.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    # Create visualizations
    viz = ResultsVisualizer()
    viz.visualize_all(["test_benchmark"])

    logger.info("Test visualizations created!")
