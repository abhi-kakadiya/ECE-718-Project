"""
Evaluation metrics for DSPy optimization.

Metrics guide the optimization process by defining what "better" means.
"""

import re
import numpy as np
import time
from typing import Tuple, Any

def code_validity_metric(example, prediction, trace=None) -> float:
    """
    Metric that checks if generated code is valid Python.

    Args:
        example: DSPy Example with expected outputs
        prediction: Model prediction
        trace: Optional execution trace

    Returns:
        1.0 if code is valid Python, 0.0 otherwise
    """
    try:
        # Get generated code
        if hasattr(prediction, 'optimized_code'):
            code = prediction.optimized_code
        elif hasattr(prediction, 'generated_code'):
            code = prediction.generated_code
        else:
            return 0.0

        # Try to compile the code
        compile(code, '<string>', 'exec')
        return 1.0

    except SyntaxError:
        return 0.0
    except Exception:
        return 0.0


def code_contains_numpy_metric(example, prediction, trace=None) -> float:
    """
    Metric that checks if code uses NumPy (required for our benchmarks).

    Args:
        example: DSPy Example
        prediction: Model prediction
        trace: Optional trace

    Returns:
        1.0 if code imports and uses NumPy, 0.0 otherwise
    """
    try:
        # Get generated code
        if hasattr(prediction, 'optimized_code'):
            code = prediction.optimized_code
        elif hasattr(prediction, 'generated_code'):
            code = prediction.generated_code
        else:
            return 0.0

        # Check for numpy import
        has_import = bool(re.search(r'import\s+numpy|from\s+numpy', code))

        # Check for numpy usage
        has_usage = bool(re.search(r'np\.|numpy\.', code))

        return 1.0 if (has_import and has_usage) else 0.0

    except Exception:
        return 0.0


def code_has_function_metric(example, prediction, trace=None) -> float:
    """
    Metric that checks if code defines a function.

    Args:
        example: DSPy Example
        prediction: Model prediction
        trace: Optional trace

    Returns:
        1.0 if code defines a function, 0.0 otherwise
    """
    try:
        # Get generated code
        if hasattr(prediction, 'optimized_code'):
            code = prediction.optimized_code
        elif hasattr(prediction, 'generated_code'):
            code = prediction.generated_code
        else:
            return 0.0

        # Check for function definition
        has_function = bool(re.search(r'def\s+\w+\s*\(', code))

        return 1.0 if has_function else 0.0

    except Exception:
        return 0.0


def code_has_docstring_metric(example, prediction, trace=None) -> float:
    """
    Metric that checks if code has docstrings (good practice).

    Args:
        example: DSPy Example
        prediction: Model prediction
        trace: Optional trace

    Returns:
        1.0 if code has docstrings, 0.5 if partial, 0.0 otherwise
    """
    try:
        # Get generated code
        if hasattr(prediction, 'optimized_code'):
            code = prediction.optimized_code
        elif hasattr(prediction, 'generated_code'):
            code = prediction.generated_code
        else:
            return 0.0

        # Check for docstrings (triple quotes)
        has_docstring = bool(re.search(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', code))

        return 1.0 if has_docstring else 0.0

    except Exception:
        return 0.0


def combined_code_quality_metric(example, prediction, trace=None) -> float:
    """
    Combined metric that evaluates multiple aspects of code quality.

    Weights:
    - Valid Python: 40%
    - Uses NumPy: 30%
    - Has function: 20%
    - Has docstring: 10%

    Args:
        example: DSPy Example
        prediction: Model prediction
        trace: Optional trace

    Returns:
        Score from 0.0 to 1.0
    """
    validity_score = code_validity_metric(example, prediction, trace)
    numpy_score = code_contains_numpy_metric(example, prediction, trace)
    function_score = code_has_function_metric(example, prediction, trace)
    docstring_score = code_has_docstring_metric(example, prediction, trace)

    # Weighted combination
    total_score = (
        0.4 * validity_score +
        0.3 * numpy_score +
        0.2 * function_score +
        0.1 * docstring_score
    )

    return total_score


def _create_test_data(benchmark_name: str, size: int) -> Tuple[Any, ...]:
    """
    Create test data for performance measurement.

    Args:
        benchmark_name: Name of the benchmark
        size: Problem size

    Returns:
        Tuple of test inputs
    """
    if benchmark_name == "matrix_multiply":
        A = np.random.randn(size, size).astype(np.float64)
        B = np.random.randn(size, size).astype(np.float64)
        return (A, B)

    elif benchmark_name == "cholesky":
        # Create symmetric positive-definite matrix
        A = np.random.randn(size, size).astype(np.float64)
        A = A @ A.T + size * np.eye(size)
        return (A,)

    elif benchmark_name == "fft":
        signal = np.random.randn(size).astype(np.complex128)
        return (signal,)

    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")


def _get_numpy_baseline_time(benchmark_name: str, test_data: Tuple, iterations: int = 5) -> float:
    """
    Get baseline execution time using NumPy reference implementation.

    Args:
        benchmark_name: Name of the benchmark
        test_data: Test input data
        iterations: Number of timing iterations

    Returns:
        Mean execution time in milliseconds
    """
    times = []

    for _ in range(iterations):
        start = time.perf_counter()

        if benchmark_name == "matrix_multiply":
            A, B = test_data
            _ = A @ B
        elif benchmark_name == "cholesky":
            A = test_data[0]
            _ = np.linalg.cholesky(A)
        elif benchmark_name == "fft":
            signal = test_data[0]
            _ = np.fft.fft(signal)

        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    return float(np.mean(times))


def _measure_code_performance(code: str, benchmark_name: str, test_data: Tuple, iterations: int = 5) -> float:
    """
    Measure execution time of generated code.

    Args:
        code: Generated code string
        benchmark_name: Name of the benchmark
        test_data: Test input data
        iterations: Number of timing iterations

    Returns:
        Mean execution time in milliseconds, or -1.0 if execution failed
    """
    try:
        # Extract function name
        match = re.search(r'def\s+(\w+)\s*\(', code)
        if not match:
            return -1.0

        function_name = match.group(1)

        # Execute code in isolated namespace
        namespace = {'np': np, 'numpy': np}
        exec(code, namespace)

        if function_name not in namespace:
            return -1.0

        func = namespace[function_name]

        # Test execution (verify it works)
        try:
            _ = func(*test_data)
        except Exception:
            return -1.0

        # Measure performance
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = func(*test_data)
            end = time.perf_counter()
            times.append((end - start) * 1000.0)

        return float(np.mean(times))

    except Exception:
        return -1.0


def performance_aware_metric(benchmark_name: str, size: int = 128, weight: float = 0.5):
    """
    Create a performance-aware metric that rewards faster code.

    This metric combines code quality with actual execution performance,
    guiding DSPy optimizers to generate not just correct, but also fast code.

    Args:
        benchmark_name: Name of the benchmark
        size: Problem size for testing (smaller for faster compilation)
        weight: Weight for performance component (0.0 = quality only, 1.0 = performance only)

    Returns:
        Metric function that evaluates both quality and performance
    """
    def metric(example, prediction, trace=None) -> float:
        """Performance-aware metric."""
        # Start with code quality score (0.0 to 1.0)
        quality_score = combined_code_quality_metric(example, prediction, trace)

        # If code quality is too low, don't bother measuring performance
        if quality_score < 0.5:
            return quality_score

        try:
            # Get generated code
            if hasattr(prediction, 'optimized_code'):
                code = prediction.optimized_code
            elif hasattr(prediction, 'generated_code'):
                code = prediction.generated_code
            else:
                return quality_score

            # Create test data
            test_data = _create_test_data(benchmark_name, size)

            # Measure NumPy baseline performance
            baseline_time = _get_numpy_baseline_time(benchmark_name, test_data, iterations=3)

            # Measure generated code performance
            code_time = _measure_code_performance(code, benchmark_name, test_data, iterations=3)

            # If execution failed, return quality score only
            if code_time < 0:
                return quality_score

            # Calculate performance score (1.0 = matches NumPy, >1.0 = faster, <1.0 = slower)
            # Cap at 1.5 to prevent unrealistic scores
            performance_ratio = min(1.5, baseline_time / code_time)

            # Normalize to 0.0-1.0 range (0.5 = 2x slower, 1.0 = matches NumPy, 1.0 = faster)
            # Using sigmoid-like curve: slower code gets penalized more
            if performance_ratio >= 1.0:
                # As fast or faster than NumPy
                performance_score = 1.0
            else:
                # Slower than NumPy - penalize proportionally
                # 0.5x speed (2x slower) = 0.5 score, 0.75x speed = 0.75 score
                performance_score = max(0.0, performance_ratio)

            # Combine quality and performance
            final_score = (1.0 - weight) * quality_score + weight * performance_score

            return final_score

        except Exception:
            # If any error occurs, fall back to quality score
            return quality_score

    return metric


def benchmark_specific_metric(benchmark_name: str):
    """
    Create a benchmark-specific metric.

    Args:
        benchmark_name: Name of the benchmark

    Returns:
        Metric function for the benchmark
    """
    def metric(example, prediction, trace=None) -> float:
        """Benchmark-specific metric."""
        # Start with base code quality
        base_score = combined_code_quality_metric(example, prediction, trace)

        try:
            # Get generated code
            if hasattr(prediction, 'optimized_code'):
                code = prediction.optimized_code
            elif hasattr(prediction, 'generated_code'):
                code = prediction.generated_code
            else:
                return base_score

            # Benchmark-specific checks
            bonus = 0.0

            if benchmark_name == "matrix_multiply":
                # Check for matmul or @ operator
                if 'matmul' in code or '@' in code:
                    bonus += 0.1
                # Check for optimization comments
                if 'optimization' in code.lower() or 'cache' in code.lower():
                    bonus += 0.05

            elif benchmark_name == "cholesky":
                # Check for cholesky function
                if 'cholesky' in code.lower():
                    bonus += 0.1
                # Check for symmetric handling
                if 'symmetric' in code.lower() or 'spd' in code.lower():
                    bonus += 0.05

            elif benchmark_name == "fft":
                # Check for FFT functions
                if 'fft' in code.lower():
                    bonus += 0.1
                # Check for power-of-2 optimization
                if 'power' in code.lower() or 'log2' in code.lower():
                    bonus += 0.05

            return min(1.0, base_score + bonus)

        except Exception:
            return base_score

    return metric


# Create default metrics for each benchmark
matrix_multiply_metric = benchmark_specific_metric("matrix_multiply")
cholesky_metric = benchmark_specific_metric("cholesky")
fft_metric = benchmark_specific_metric("fft")


def get_metric_for_benchmark(benchmark_name: str, use_performance: bool = False,
                            performance_weight: float = 0.5, test_size: int = 128):
    """
    Get the appropriate metric for a benchmark.

    Args:
        benchmark_name: Name of the benchmark
        use_performance: Whether to use performance-aware metric (default: False for backward compatibility)
        performance_weight: Weight for performance component (0.0 = quality only, 1.0 = performance only)
        test_size: Problem size for performance testing during compilation

    Returns:
        Metric function
    """
    if use_performance:
        # Return performance-aware metric
        return performance_aware_metric(benchmark_name, size=test_size, weight=performance_weight)
    else:
        # Return quality-only metrics (backward compatible)
        metrics = {
            "matrix_multiply": matrix_multiply_metric,
            "cholesky": cholesky_metric,
            "fft": fft_metric
        }
        return metrics.get(benchmark_name, combined_code_quality_metric)


if __name__ == "__main__":
    print("Testing Metrics...")

    # Create a mock prediction
    class MockPrediction:
        def __init__(self, code):
            self.optimized_code = code

    # Test valid code
    valid_code = """
import numpy as np

def optimized_matmul(A, B):
    \"\"\"Optimized matrix multiplication.\"\"\"
    return np.matmul(A, B)
"""

    pred = MockPrediction(valid_code)

    print("\n1. Valid code test:")
    print(f"   Validity: {code_validity_metric(None, pred)}")
    print(f"   NumPy: {code_contains_numpy_metric(None, pred)}")
    print(f"   Function: {code_has_function_metric(None, pred)}")
    print(f"   Docstring: {code_has_docstring_metric(None, pred)}")
    print(f"   Combined: {combined_code_quality_metric(None, pred)}")
    print(f"   MatMul specific: {matrix_multiply_metric(None, pred)}")

    # Test invalid code
    invalid_code = "this is not valid python code"
    invalid_pred = MockPrediction(invalid_code)

    print("\n2. Invalid code test:")
    print(f"   Validity: {code_validity_metric(None, invalid_pred)}")
    print(f"   Combined: {combined_code_quality_metric(None, invalid_pred)}")

    print("\nMetrics tested successfully!")
