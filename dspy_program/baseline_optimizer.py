"""
Baseline optimizer: No DSPy optimization (control group).

This uses raw DSPy prediction without any optimization,
establishing the baseline performance and token usage.
"""

import dspy
import time
from typing import Dict, Any

from dspy_program.modules import get_module_for_benchmark
from tools.logger import get_logger
from tools.token_tracker import TokenTracker


logger = get_logger(__name__)


class BaselineOptimizer:
    """
    Baseline optimizer that performs no optimization.

    This serves as the control group to measure the impact of
    DSPy optimization techniques.
    """

    def __init__(self, benchmark_name: str):
        """
        Initialize baseline optimizer.

        Args:
            benchmark_name: Name of the benchmark
        """
        self.benchmark_name = benchmark_name
        self.module = get_module_for_benchmark(benchmark_name)
        self.token_tracker = TokenTracker(benchmark_name, "baseline")

        logger.info(
            f"Initialized BaselineOptimizer for {benchmark_name}\n"
            f"  → This is the control group with zero optimization"
        )

    def compile(self, trainset=None) -> 'BaselineOptimizer':
        """
        'Compile' the optimizer (no-op for baseline).

        Args:
            trainset: Training examples (unused for baseline)

        Returns:
            Self (unmodified)
        """
        logger.info("✓ Baseline optimizer ready (no compilation needed)")

        # No compilation for baseline - that's the point!
        # Token usage is zero for compilation phase

        return self

    def _clean_generated_code(self, code: str) -> str:
        """
        Clean up LLM-generated code by removing markdown blocks and artifacts.

        Args:
            code: Raw code from LLM

        Returns:
            Cleaned Python code
        """
        import re

        code = str(code).strip()

        # Remove markdown code blocks
        if '```python' in code:
            # Extract code between ```python and ```
            match = re.search(r'```python\s*\n(.*?)\n```', code, re.DOTALL)
            if match:
                code = match.group(1).strip()
        elif '```' in code:
            # Extract code between ``` and ```
            match = re.search(r'```\s*\n(.*?)\n```', code, re.DOTALL)
            if match:
                code = match.group(1).strip()

        # Remove common LLM prefixes
        prefixes_to_remove = [
            'Here is',
            'Here\'s',
            'Sure, here is',
            'Here you go',
            'Below is',
        ]

        for prefix in prefixes_to_remove:
            if code.lower().startswith(prefix.lower()):
                # Remove the prefix line
                lines = code.split('\n')
                code = '\n'.join(lines[1:]).strip()
                break

        return code

    def generate_code(self, **kwargs) -> Dict[str, Any]:
        """
        Generate code using unoptimized module.

        Args:
            **kwargs: Input arguments for the module

        Returns:
            Dictionary with generated code and metadata
        """
        logger.debug("Generating code with baseline (no optimization)")

        # Track start time
        start_time = time.time()

        # Enable DSPy's LM call tracking
        with dspy.context(lm=dspy.settings.lm):
            # Generate code
            prediction = self.module(**kwargs)

        # Track end time
        elapsed_time = time.time() - start_time

        # Extract token usage from DSPy's history
        # NOTE: This requires DSPy to track usage
        if hasattr(dspy.settings, 'lm') and hasattr(dspy.settings.lm, 'history'):
            history = dspy.settings.lm.history
            if history:
                last_call = history[-1]
                if 'usage' in last_call:
                    usage = last_call['usage']
                    self.token_tracker.track_call(
                        prompt_tokens=usage.get('prompt_tokens', 0),
                        completion_tokens=usage.get('completion_tokens', 0),
                        call_type='inference',
                        model=last_call.get('model', 'unknown')
                    )

        # Extract generated code
        if hasattr(prediction, 'optimized_code'):
            generated_code = prediction.optimized_code
        elif hasattr(prediction, 'generated_code'):
            generated_code = prediction.generated_code
        else:
            generated_code = str(prediction)

        # Clean up LLM output - remove markdown code blocks
        generated_code = self._clean_generated_code(generated_code)

        result = {
            'generated_code': generated_code,
            'elapsed_time_seconds': elapsed_time,
            'optimizer': 'baseline',
            'benchmark': self.benchmark_name
        }

        logger.debug(f"Code generated in {elapsed_time:.2f} seconds")

        return result

    def get_token_summary(self) -> Dict:
        """Get token usage summary."""
        return self.token_tracker.get_summary()


def run_baseline_optimizer(
    benchmark_name: str,
    size: int,
    trainset=None
) -> Dict[str, Any]:
    """
    Run the baseline optimizer for a benchmark.

    Args:
        benchmark_name: Name of the benchmark
        size: Problem size
        trainset: Training examples (unused)

    Returns:
        Dictionary with results
    """
    logger.info(
        f"\n{'='*80}\n"
        f"Running BASELINE optimizer: {benchmark_name} (size={size})\n"
        f"{'='*80}\n"
    )

    # Create optimizer
    optimizer = BaselineOptimizer(benchmark_name)

    # Compile (no-op for baseline)
    optimizer.compile(trainset=trainset)

    # Generate code with appropriate inputs
    if benchmark_name == "matrix_multiply":
        result = optimizer.generate_code(
            matrix_size=f"{size}x{size}",
            optimization_techniques="Minimize execution time using NumPy optimizations"
        )
    elif benchmark_name == "cholesky":
        result = optimizer.generate_code(
            matrix_size=f"{size}x{size}",
            optimization_techniques="Use NumPy's optimized LAPACK-based implementation"
        )
    elif benchmark_name == "fft":
        result = optimizer.generate_code(
            signal_length=str(size),
            optimization_techniques="Use NumPy's optimized FFT implementation"
        )
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    # Add token usage to result
    result['token_usage'] = optimizer.get_token_summary()

    logger.info(f"\nBaseline optimizer completed for {benchmark_name}\n")
    optimizer.token_tracker.log_summary()

    return result


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Configure DSPy with your LLM
    lm = dspy.LM(
        model=os.getenv('LLM_MODEL', 'gpt-3.5-turbo'),
        api_key=os.getenv('OPENAI_API_KEY'),
        temperature=float(os.getenv('LLM_TEMPERATURE', 0.7)),
        max_tokens=int(os.getenv('LLM_MAX_TOKENS', 2048))
    )
    dspy.settings.configure(lm=lm)

    # Test baseline optimizer
    logger.info("Testing Baseline Optimizer...")

    result = run_baseline_optimizer(
        benchmark_name="matrix_multiply",
        size=128
    )

    print("\n" + "="*80)
    print("Generated Code:")
    print("="*80)
    print(result['generated_code'])
    print("="*80)

    print("\nToken Usage:")
    print(f"  Total tokens: {result['token_usage']['total_tokens']}")
