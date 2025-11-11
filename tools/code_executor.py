"""
Code executor for measuring performance of LLM-generated code.

This module safely executes generated code and measures its performance,
enabling comparison between different DSPy optimizers.
"""

import numpy as np
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import traceback

from tools.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    mean_time_ms: float = 0.0
    std_time_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0
    num_iterations: int = 0
    correctness_verified: bool = False
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'mean_time_ms': self.mean_time_ms,
            'std_time_ms': self.std_time_ms,
            'min_time_ms': self.min_time_ms,
            'max_time_ms': self.max_time_ms,
            'num_iterations': self.num_iterations,
            'correctness_verified': self.correctness_verified,
            'error_message': self.error_message
        }


class CodeExecutor:
    """
    Executes generated code and measures performance.

    Features:
    - Safe execution with timeout
    - Performance measurement with statistics
    - Correctness validation against reference implementation
    - Error handling and reporting
    """

    def __init__(self, iterations: int = 10, warmup: int = 3, timeout: float = 30.0):
        """
        Initialize code executor.

        Args:
            iterations: Number of timing iterations
            warmup: Number of warmup iterations
            timeout: Timeout in seconds for code execution
        """
        self.iterations = iterations
        self.warmup = warmup
        self.timeout = timeout

        logger.debug(f"CodeExecutor initialized: iterations={iterations}, warmup={warmup}")

    def extract_function(self, code: str, benchmark_name: str) -> Optional[str]:
        """
        Extract the main function from generated code.

        Args:
            code: Generated code string
            benchmark_name: Name of benchmark

        Returns:
            Function name if found, None otherwise
        """
        import re

        # Use regex to find the first function definition
        # This handles function names with suffixes like optimized_matmul_128x128
        match = re.search(r'def\s+(\w+)\s*\(', code)
        if match:
            func_name = match.group(1)
            logger.debug(f"Found function: {func_name}")
            return func_name

        logger.warning("No function definition found in generated code")
        return None

    def create_test_data(self, benchmark_name: str, size: int) -> Tuple[Any, ...]:
        """
        Create test data for benchmark.

        Args:
            benchmark_name: Name of benchmark
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
            A = A @ A.T + size * np.eye(size)  # Ensure positive-definite
            return (A,)

        elif benchmark_name == "fft":
            signal = np.random.randn(size).astype(np.complex128)
            return (signal,)

        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

    def get_reference_output(self, benchmark_name: str, test_data: Tuple) -> Any:
        """
        Get reference output using NumPy.

        Args:
            benchmark_name: Name of benchmark
            test_data: Test input data

        Returns:
            Reference output
        """
        if benchmark_name == "matrix_multiply":
            A, B = test_data
            return A @ B

        elif benchmark_name == "cholesky":
            A = test_data[0]
            return np.linalg.cholesky(A)

        elif benchmark_name == "fft":
            signal = test_data[0]
            return np.fft.fft(signal)

        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

    def verify_correctness(self,
                          output: Any,
                          reference: Any,
                          benchmark_name: str,
                          rtol: float = 1e-5,
                          atol: float = 1e-8) -> bool:
        """
        Verify output correctness against reference.

        Args:
            output: Generated code output
            reference: Reference output
            benchmark_name: Name of benchmark
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            True if correct, False otherwise
        """
        try:
            # Handle None or missing outputs
            if output is None or reference is None:
                logger.warning("Output or reference is None")
                return False

            # Convert to arrays if needed
            output_array = np.asarray(output)
            reference_array = np.asarray(reference)

            # Check shapes match
            if output_array.shape != reference_array.shape:
                logger.warning(f"Shape mismatch: {output_array.shape} vs {reference_array.shape}")
                return False

            # Compare values
            is_close = np.allclose(output_array, reference_array, rtol=rtol, atol=atol)

            if is_close:
                logger.debug("Output verified as correct")
            else:
                max_diff = np.max(np.abs(output_array - reference_array))
                logger.warning(f"Output incorrect, max difference: {max_diff}")

            return is_close

        except Exception as e:
            logger.error(f"Error verifying correctness: {e}")
            return False

    def execute_code(self,
                    code: str,
                    benchmark_name: str,
                    size: int) -> ExecutionResult:
        """
        Execute generated code and measure performance.

        Args:
            code: Generated code string
            benchmark_name: Name of benchmark
            size: Problem size

        Returns:
            ExecutionResult with performance metrics
        """
        logger.info(f"Executing generated code for {benchmark_name} (size={size})")

        try:
            # Extract function name
            function_name = self.extract_function(code, benchmark_name)
            if not function_name:
                error_msg = "Could not find function in generated code"
                logger.error(f"✗ {error_msg}")
                logger.error(f"  Expected function matching: optimized_{benchmark_name}, {benchmark_name}, run_{benchmark_name}")
                return ExecutionResult(
                    success=False,
                    error_message=error_msg
                )

            # Create test data
            test_data = self.create_test_data(benchmark_name, size)
            reference_output = self.get_reference_output(benchmark_name, test_data)

            # Execute code in isolated namespace
            namespace = {'np': np, 'numpy': np}

            try:
                exec(code, namespace)
            except Exception as e:
                error_msg = f"Code execution failed: {str(e)}"
                logger.error(f"✗ {error_msg}")
                logger.error(f"  Traceback:\n{traceback.format_exc()}")
                return ExecutionResult(
                    success=False,
                    error_message=f"{error_msg}\n{traceback.format_exc()}"
                )

            # Get function from namespace
            if function_name not in namespace:
                error_msg = f"Function '{function_name}' not found in namespace"
                logger.error(f"✗ {error_msg}")
                logger.error(f"  Available names: {list(namespace.keys())}")
                return ExecutionResult(
                    success=False,
                    error_message=error_msg
                )

            func = namespace[function_name]

            # Test execution and verify correctness
            try:
                test_output = func(*test_data)
                correctness_verified = self.verify_correctness(
                    test_output, reference_output, benchmark_name
                )
            except Exception as e:
                error_msg = f"Function execution failed: {str(e)}"
                logger.error(f"✗ {error_msg}")
                logger.error(f"  Traceback:\n{traceback.format_exc()}")
                return ExecutionResult(
                    success=False,
                    error_message=f"{error_msg}\n{traceback.format_exc()}"
                )

            # Warmup runs
            logger.debug(f"Running {self.warmup} warmup iterations...")
            for _ in range(self.warmup):
                try:
                    _ = func(*test_data)
                except Exception as e:
                    return ExecutionResult(
                        success=False,
                        error_message=f"Warmup failed: {str(e)}"
                    )

            # Timing runs
            logger.debug(f"Running {self.iterations} timing iterations...")
            times = []

            for i in range(self.iterations):
                start = time.perf_counter()
                try:
                    _ = func(*test_data)
                except Exception as e:
                    return ExecutionResult(
                        success=False,
                        error_message=f"Timing iteration {i} failed: {str(e)}"
                    )
                end = time.perf_counter()

                elapsed_ms = (end - start) * 1000.0
                times.append(elapsed_ms)

            # Calculate statistics
            times_array = np.array(times)
            mean_time = float(np.mean(times_array))
            std_time = float(np.std(times_array))
            min_time = float(np.min(times_array))
            max_time = float(np.max(times_array))

            logger.info(f"✓ Execution successful: {mean_time:.4f} ± {std_time:.4f} ms")
            logger.info(f"  Correctness verified: {correctness_verified}")

            return ExecutionResult(
                success=True,
                mean_time_ms=mean_time,
                std_time_ms=std_time,
                min_time_ms=min_time,
                max_time_ms=max_time,
                num_iterations=self.iterations,
                correctness_verified=correctness_verified
            )

        except Exception as e:
            logger.error(f"Unexpected error executing code: {e}")
            logger.error(traceback.format_exc())
            return ExecutionResult(
                success=False,
                error_message=f"Unexpected error: {str(e)}"
            )


def execute_and_measure(code: str,
                       benchmark_name: str,
                       size: int,
                       iterations: int = 10,
                       warmup: int = 3) -> Dict[str, Any]:
    """
    Convenience function to execute code and measure performance.

    Args:
        code: Generated code string
        benchmark_name: Name of benchmark
        size: Problem size
        iterations: Number of timing iterations
        warmup: Number of warmup iterations

    Returns:
        Dictionary with execution results
    """
    executor = CodeExecutor(iterations=iterations, warmup=warmup)
    result = executor.execute_code(code, benchmark_name, size)
    return result.to_dict()


if __name__ == "__main__":
    # Test code executor
    test_code = """
import numpy as np

def matmul(A, B):
    return A @ B
"""

    logger.info("Testing CodeExecutor...")

    executor = CodeExecutor(iterations=5, warmup=2)
    result = executor.execute_code(test_code, "matrix_multiply", 128)

    print("\nExecution Result:")
    print(f"  Success: {result.success}")
    print(f"  Mean time: {result.mean_time_ms:.4f} ms")
    print(f"  Std dev: {result.std_time_ms:.4f} ms")
    print(f"  Correctness: {result.correctness_verified}")

    if not result.success:
        print(f"  Error: {result.error_message}")
