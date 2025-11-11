"""
Main entry point for DSPy HPC Benchmarking Project.

This provides a user-friendly CLI interface with pretty formatting.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

from tools.pretty_output import (
    print_header, print_section, print_subsection,
    print_success, print_error, print_warning, print_info,
    print_step, print_metric, print_key_value, print_divider
)
from tools.logger import get_logger

logger = get_logger(__name__)


def check_environment():
    """Check if the environment is properly configured."""
    print_subsection("Checking Environment")

    issues = []

    # Check for .env file
    if not Path('.env').exists():
        issues.append(".env file not found (copy from .env.example)")
        print_warning(".env file not found")
    else:
        print_success(".env file found")

    # Check for API key
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        issues.append("No API key found in .env")
        print_error("No API key configured")
    else:
        print_success("API key configured")

    # Check for required directories
    required_dirs = ['results', 'figures', 'logs', 'cache']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print_info(f"Created {dir_name}/ directory")
        else:
            print_success(f"{dir_name}/ directory exists")

    return len(issues) == 0, issues


def show_configuration():
    """Display current configuration."""
    print_subsection("Current Configuration")

    load_dotenv()

    config_items = [
        ("Model", os.getenv('LLM_MODEL', 'not set')),
        ("Temperature", os.getenv('LLM_TEMPERATURE', 'not set')),
        ("Cache Enabled", os.getenv('DSP_CACHEBOOL', 'not set')),
        ("Results Directory", os.getenv('RESULTS_DIR', './results')),
        ("Figures Directory", os.getenv('FIGURES_DIR', './figures')),
    ]

    for key, value in config_items:
        print_key_value(key, value, indent=1)


def run_quick_test():
    """Run a quick test with baseline optimizer."""
    print_subsection("Running Quick Test")

    print_step(1, 3, "Configuring DSPy...")

    try:
        import dspy
        load_dotenv()

        model = os.getenv('LLM_MODEL', 'gpt-4.1-mini')
        api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            print_error("OPENAI_API_KEY not found in .env")
            return False

        lm = dspy.LM(model=model, api_key=api_key)
        dspy.settings.configure(lm=lm)

        print_success(f"DSPy configured with {model}")

    except Exception as e:
        print_error(f"DSPy configuration failed: {e}")
        return False

    print_step(2, 3, "Running baseline benchmark...")

    try:
        from benchmarks.baseline_implementations import BaselineBenchmarks

        result = BaselineBenchmarks.run_benchmark(
            benchmark_name="matrix_multiply",
            method="numpy",
            size=128,
            iterations=3,
            warmup=1
        )

        print_success("Baseline benchmark completed")
        print_metric("Mean Time", f"{result.mean_time_ms:.2f}", "ms")
        print_metric("Std Dev", f"{result.std_time_ms:.2f}", "ms")

    except Exception as e:
        print_error(f"Baseline benchmark failed: {e}")
        return False

    print_step(3, 3, "Testing baseline optimizer...")

    try:
        from dspy_program.baseline_optimizer import run_baseline_optimizer

        optimizer_result = run_baseline_optimizer(
            benchmark_name="matrix_multiply",
            size=128
        )

        print_success("Baseline optimizer completed")
        print_metric("Code Length", f"{len(optimizer_result['generated_code'])}", "chars")
        print_metric("Total Tokens", optimizer_result['token_usage']['total_tokens'])

    except Exception as e:
        print_error(f"Baseline optimizer failed: {e}")
        return False

    print_success("Quick test completed successfully!")
    return True


def run_single_benchmark(benchmark: str):
    """Run a single benchmark with all optimizers."""
    print_subsection(f"Running Benchmark: {benchmark}")

    try:
        from tools.run_all_benchmarks import BenchmarkRunner

        runner = BenchmarkRunner()

        print_info("This may take 10-20 minutes...")
        print_warning("API costs: approximately $0.03-0.10")

        # Clear old results for fresh run
        print_step(0, 3, "Clearing old results...")
        runner.clear_old_results(benchmarks=[benchmark])

        # Configure DSPy
        runner.configure_dspy()

        # Run NumPy baseline for selected benchmark only
        print_step(1, 3, f"Running baseline benchmarks for {benchmark}...")
        runner.run_baseline_benchmarks(benchmarks=[benchmark])

        # Run DSPy optimizers
        print_step(2, 3, f"Running DSPy optimizers for {benchmark}...")
        runner.run_all(
            benchmarks=[benchmark],
            skip_baseline=True,
            skip_visualization=False,
            skip_clear=True  # Already cleared above
        )

        print_success(f"Benchmark {benchmark} completed!")
        print_info(f"Results saved to: results/{benchmark}/")
        print_info("Figures saved to: figures/")

        return True

    except Exception as e:
        print_error(f"Benchmark failed: {e}")
        logger.exception("Benchmark error:")
        return False


def run_all_benchmarks():
    """Run all benchmarks."""
    print_subsection("Running All Benchmarks")

    print_warning("This will take 1-2 hours")
    print_warning("API costs: approximately $0.40-1.00")

    response = input("\n  Continue? (y/n): ")
    if response.lower() != 'y':
        print_info("Cancelled by user")
        return False

    try:
        from tools.run_all_benchmarks import BenchmarkRunner

        runner = BenchmarkRunner()
        runner.run_all()

        print_success("All benchmarks completed!")
        return True

    except Exception as e:
        print_error(f"Benchmark suite failed: {e}")
        logger.exception("Benchmark error:")
        return False


def show_menu():
    """Display interactive menu."""
    print_header("DSPy HPC Benchmarking Project")

    print_info("Choose an option:")
    print()
    print("  1. Check environment setup")
    print("  2. Show current configuration")
    print("  3. Run quick test (< 1 minute, < $0.01)")
    print("  4. Run single benchmark (10-20 min, $0.03-0.10)")
    print("  5. Run all benchmarks (1-2 hours, $0.40-1.00)")
    print("  6. Generate visualizations from existing results")
    print("  0. Exit")
    print()

    choice = input("  Enter your choice: ")
    return choice


def generate_visualizations():
    """Generate visualizations from existing results."""
    print_subsection("Generating Visualizations")

    try:
        from tools.visualize_results import ResultsVisualizer

        viz = ResultsVisualizer()
        viz.visualize_all()

        print_success("Visualizations generated!")
        print_info("Figures saved to: figures/")
        return True

    except Exception as e:
        print_error(f"Visualization failed: {e}")
        logger.exception("Visualization error:")
        return False


def interactive_mode():
    """Run in interactive mode with menu."""
    while True:
        choice = show_menu()

        if choice == '1':
            print_section("Environment Check")
            is_ok, issues = check_environment()
            if is_ok:
                print_success("Environment is properly configured!")
            else:
                print_error("Environment has issues:")
                for issue in issues:
                    print_warning(f"  • {issue}")

        elif choice == '2':
            print_section("Configuration")
            show_configuration()

        elif choice == '3':
            print_section("Quick Test")
            if run_quick_test():
                print_divider()
                print_success("Quick test passed! Your setup is working.")
            else:
                print_error("Quick test failed. Check the logs above.")

        elif choice == '4':
            print_section("Single Benchmark")
            print("  Available benchmarks:")
            print("    1. matrix_multiply")
            print("    2. cholesky")
            print("    3. fft")
            bench_choice = input("\n  Choose benchmark (1-3): ")

            benchmark_map = {'1': 'matrix_multiply', '2': 'cholesky', '3': 'fft'}
            if bench_choice in benchmark_map:
                run_single_benchmark(benchmark_map[bench_choice])
            else:
                print_error("Invalid benchmark choice")

        elif choice == '5':
            print_section("All Benchmarks")
            run_all_benchmarks()

        elif choice == '6':
            print_section("Visualizations")
            generate_visualizations()

        elif choice == '0':
            print_divider()
            print_success("Goodbye!")
            break

        else:
            print_error("Invalid choice. Please try again.")

        print_divider()
        input("\nPress Enter to continue...")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DSPy HPC Benchmarking Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive mode
  python main.py --quick-test             # Run quick test
  python main.py --benchmark matrix_multiply  # Run single benchmark
  python main.py --all                    # Run all benchmarks
  python main.py --visualize              # Generate visualizations
        """
    )

    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test')
    parser.add_argument('--benchmark', type=str,
                       help='Run specific benchmark',
                       choices=['matrix_multiply', 'cholesky', 'fft'])
    parser.add_argument('--all', action='store_true',
                       help='Run all benchmarks')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--check', action='store_true',
                       help='Check environment')
    parser.add_argument('--config', action='store_true',
                       help='Show configuration')

    args = parser.parse_args()

    # Non-interactive modes
    if args.check:
        print_header("Environment Check")
        is_ok, issues = check_environment()
        sys.exit(0 if is_ok else 1)

    elif args.config:
        print_header("Configuration")
        show_configuration()

    elif args.quick_test:
        print_header("Quick Test")
        success = run_quick_test()
        sys.exit(0 if success else 1)

    elif args.benchmark:
        print_header(f"Benchmark: {args.benchmark}")
        success = run_single_benchmark(args.benchmark)
        sys.exit(0 if success else 1)

    elif args.all:
        print_header("All Benchmarks")
        success = run_all_benchmarks()
        sys.exit(0 if success else 1)

    elif args.visualize:
        print_header("Visualizations")
        success = generate_visualizations()
        sys.exit(0 if success else 1)

    else:
        # Interactive mode (default)
        interactive_mode()


if __name__ == "__main__":
    main()
