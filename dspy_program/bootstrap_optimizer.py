"""
BootstrapFewShot optimizer implementation.

This optimizer learns from training examples by bootstrapping demonstrations.
It selects the most effective examples to include as few-shot prompts.
"""

import dspy
import time
from typing import Dict, Any, List

from dspy.teleprompt import BootstrapFewShot
from dspy_program.modules import get_module_for_benchmark
from dspy_program.metrics import get_metric_for_benchmark
from dspy_program.training_examples import get_training_examples
from tools.logger import get_logger
from tools.token_tracker import TokenTracker


logger = get_logger(__name__)


class BootstrapOptimizer:
    """
    Bootstrap Few-Shot optimizer.

    This optimizer:
    1. Takes training examples
    2. Runs the program on training inputs
    3. Selects successful demonstrations
    4. Includes them as few-shot examples in future prompts
    """

    def __init__(self, benchmark_name: str, max_bootstrapped_demos: int = 4,
                 use_performance_metric: bool = True, performance_weight: float = 0.5,
                 test_size: int = 128):
        """
        Initialize Bootstrap optimizer.

        Args:
            benchmark_name: Name of the benchmark
            max_bootstrapped_demos: Maximum number of demonstrations to bootstrap
            use_performance_metric: Whether to use performance-aware metric during compilation
            performance_weight: Weight for performance component (0.0-1.0)
            test_size: Problem size for performance testing during compilation
        """
        self.benchmark_name = benchmark_name
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.module = get_module_for_benchmark(benchmark_name)
        self.metric = get_metric_for_benchmark(
            benchmark_name,
            use_performance=use_performance_metric,
            performance_weight=performance_weight,
            test_size=test_size
        )
        self.token_tracker = TokenTracker(benchmark_name, "bootstrap")

        self.optimized_module = None
        self.compilation_time = 0.0

        metric_type = "performance-aware" if use_performance_metric else "quality-only"
        logger.info(
            f"Initialized BootstrapOptimizer for {benchmark_name}\n"
            f"  → Max bootstrapped demos: {max_bootstrapped_demos}\n"
            f"  → Metric type: {metric_type} (weight={performance_weight if use_performance_metric else 'N/A'})"
        )

    def compile(self, trainset: List[dspy.Example] = None) -> 'BootstrapOptimizer':
        """
        Compile the optimizer using Bootstrap Few-Shot.

        Args:
            trainset: Training examples (if None, will load default examples)

        Returns:
            Self with optimized module
        """
        # Load training examples if not provided
        if trainset is None:
            trainset = get_training_examples(self.benchmark_name)

        logger.info(
            f"\n{'='*80}\n"
            f"COMPILING BOOTSTRAP OPTIMIZER: {self.benchmark_name}\n"
            f"{'='*80}\n"
            f"  Training examples: {len(trainset)}\n"
            f"  Max demos:         {self.max_bootstrapped_demos}\n"
            f"\n"
            f"  Compilation steps:\n"
            f"    1. Running program on training examples\n"
            f"    2. Selecting successful demonstrations\n"
            f"    3. Building optimized prompts\n"
            f"{'='*80}"
        )

        # Track compilation start time
        start_time = time.time()

        # Create and compile Bootstrap optimizer
        teleprompter = BootstrapFewShot(
            metric=self.metric,
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_labeled_demos=self.max_bootstrapped_demos * 2
        )

        try:
            self.optimized_module = teleprompter.compile(
                student=self.module,
                trainset=trainset
            )

            # Track compilation time
            self.compilation_time = time.time() - start_time

            logger.info(f"\nCompilation completed in {self.compilation_time:.2f} seconds")

            # Extract token usage from compilation
            # DSPy's compilation may make multiple LLM calls
            if hasattr(dspy.settings, 'lm') and hasattr(dspy.settings.lm, 'history'):
                history = dspy.settings.lm.history
                logger.info(f"Processing {len(history)} LLM calls from compilation...")

                for call in history:
                    if 'usage' in call:
                        usage = call['usage']
                        self.token_tracker.track_call(
                            prompt_tokens=usage.get('prompt_tokens', 0),
                            completion_tokens=usage.get('completion_tokens', 0),
                            call_type='compilation',
                            model=call.get('model', 'unknown')
                        )

                # Clear history for inference phase
                dspy.settings.lm.history.clear()

            logger.info("✓ Bootstrap optimization successful!")

        except Exception as e:
            logger.error(f"Compilation failed: {e}")
            logger.warning("Falling back to unoptimized module")
            self.optimized_module = self.module
            self.compilation_time = time.time() - start_time

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
        Generate code using optimized module.

        Args:
            **kwargs: Input arguments for the module

        Returns:
            Dictionary with generated code and metadata
        """
        if self.optimized_module is None:
            raise RuntimeError("Must call compile() before generate_code()")

        logger.debug("Generating code with Bootstrap-optimized module")

        # Track start time
        start_time = time.time()

        # Generate code
        with dspy.context(lm=dspy.settings.lm):
            prediction = self.optimized_module(**kwargs)

        # Track end time
        elapsed_time = time.time() - start_time

        # Extract token usage from inference
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
            'compilation_time_seconds': self.compilation_time,
            'optimizer': 'bootstrap',
            'benchmark': self.benchmark_name
        }

        logger.debug(f"Code generated in {elapsed_time:.2f} seconds")

        return result

    def get_token_summary(self) -> Dict:
        """Get token usage summary."""
        return self.token_tracker.get_summary()


def run_bootstrap_optimizer(
    benchmark_name: str,
    size: int,
    trainset: List[dspy.Example] = None,
    max_bootstrapped_demos: int = 4,
    use_performance_metric: bool = True,
    performance_weight: float = 0.5
) -> Dict[str, Any]:
    """
    Run the Bootstrap optimizer for a benchmark.

    Args:
        benchmark_name: Name of the benchmark
        size: Problem size
        trainset: Training examples
        max_bootstrapped_demos: Max demonstrations to bootstrap
        use_performance_metric: Whether to use performance-aware metric during compilation
        performance_weight: Weight for performance component (0.0-1.0)

    Returns:
        Dictionary with results
    """
    logger.info(
        f"\n{'='*80}\n"
        f"Running BOOTSTRAP optimizer: {benchmark_name} (size={size})\n"
        f"{'='*80}\n"
    )

    # Use smaller test size for compilation (faster)
    # Typically 1/4 to 1/2 of actual test size
    compilation_test_size = max(64, size // 4)

    # Create optimizer
    optimizer = BootstrapOptimizer(
        benchmark_name=benchmark_name,
        max_bootstrapped_demos=max_bootstrapped_demos,
        use_performance_metric=use_performance_metric,
        performance_weight=performance_weight,
        test_size=compilation_test_size
    )

    # Compile with training examples
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

    logger.info(f"\nBootstrap optimizer completed for {benchmark_name}\n")
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

    # Test Bootstrap optimizer
    logger.info("Testing Bootstrap Optimizer...")

    result = run_bootstrap_optimizer(
        benchmark_name="matrix_multiply",
        size=128,
        max_bootstrapped_demos=2  # Small number for testing
    )

    print("\n" + "="*80)
    print("Generated Code:")
    print("="*80)
    print(result['generated_code'])
    print("="*80)

    print("\nToken Usage:")
    print(f"  Compilation tokens: {result['token_usage']['compilation']['total_tokens']}")
    print(f"  Inference tokens: {result['token_usage']['inference']['total_tokens']}")
    print(f"  Total tokens: {result['token_usage']['total_tokens']}")
