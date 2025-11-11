"""
Baseline implementations using NumPy and Intel MKL.
These serve as performance baselines for comparison.
"""

import numpy as np
import time
from typing import Tuple
from dataclasses import dataclass

from tools.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    benchmark_name: str
    method: str
    size: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    iterations: int
    dtype: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "method": self.method,
            "size": self.size,
            "mean_time_ms": round(self.mean_time_ms, 4),
            "std_time_ms": round(self.std_time_ms, 4),
            "min_time_ms": round(self.min_time_ms, 4),
            "max_time_ms": round(self.max_time_ms, 4),
            "iterations": self.iterations,
            "dtype": self.dtype
        }


class BaselineBenchmarks:
    """Collection of baseline benchmark implementations."""

    @staticmethod
    def matrix_multiply_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication using NumPy (BLAS backend).

        Args:
            A: First matrix (NxN)
            B: Second matrix (NxN)

        Returns:
            Result matrix C = A @ B
        """
        return np.matmul(A, B)

    @staticmethod
    def cholesky_numpy(A: np.ndarray) -> np.ndarray:
        """
        Cholesky factorization using NumPy.

        Args:
            A: Symmetric positive-definite matrix (NxN)

        Returns:
            Lower triangular matrix L where A = L @ L.T
        """
        return np.linalg.cholesky(A)

    @staticmethod
    def fft_numpy(signal: np.ndarray) -> np.ndarray:
        """
        Fast Fourier Transform using NumPy.

        Args:
            signal: Input signal (1D array)

        Returns:
            FFT of the signal
        """
        return np.fft.fft(signal)

    @staticmethod
    def generate_test_data(benchmark: str, size: int, seed: int = 42) -> Tuple:
        """
        Generate test data for benchmarks.

        Args:
            benchmark: Benchmark name ("matrix_multiply", "cholesky", "fft")
            size: Problem size
            seed: Random seed for reproducibility

        Returns:
            Tuple of input arrays for the benchmark
        """
        np.random.seed(seed)

        if benchmark == "matrix_multiply":
            A = np.random.randn(size, size).astype(np.float64)
            B = np.random.randn(size, size).astype(np.float64)
            return (A, B)

        elif benchmark == "cholesky":
            # Generate symmetric positive-definite matrix
            A = np.random.randn(size, size).astype(np.float64)
            A_spd = A.T @ A + size * np.eye(size)
            return (A_spd,)

        elif benchmark == "fft":
            # Generate complex signal
            signal = (np.random.randn(size) + 1j * np.random.randn(size)).astype(np.complex128)
            return (signal,)

        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

    @staticmethod
    def run_benchmark(
        benchmark_name: str,
        method: str,
        size: int,
        iterations: int = 10,
        warmup: int = 3,
        seed: int = 42
    ) -> BenchmarkResult:
        """
        Run a benchmark and measure performance.

        Args:
            benchmark_name: Name of benchmark
            method: Method to use ("numpy", "mkl", or custom)
            size: Problem size
            iterations: Number of measurement iterations
            warmup: Number of warmup iterations
            seed: Random seed

        Returns:
            BenchmarkResult with timing statistics
        """
        logger.info(f"Running {benchmark_name} benchmark: method={method}, size={size}")

        # Generate test data
        test_data = BaselineBenchmarks.generate_test_data(benchmark_name, size, seed)

        # Select function
        if method == "numpy":
            if benchmark_name == "matrix_multiply":
                func = BaselineBenchmarks.matrix_multiply_numpy
            elif benchmark_name == "cholesky":
                func = BaselineBenchmarks.cholesky_numpy
            elif benchmark_name == "fft":
                func = BaselineBenchmarks.fft_numpy
            else:
                raise ValueError(f"Unknown benchmark: {benchmark_name}")
        else:
            raise ValueError(f"Unknown method: {method}")

        # Warmup runs
        logger.debug(f"Running {warmup} warmup iterations...")
        for _ in range(warmup):
            _ = func(*test_data)

        # Measurement runs
        logger.debug(f"Running {iterations} measurement iterations...")
        times_ms = []

        for i in range(iterations):
            start_time = time.perf_counter()
            result = func(*test_data)
            end_time = time.perf_counter()

            elapsed_ms = (end_time - start_time) * 1000
            times_ms.append(elapsed_ms)

            logger.debug(f"  Iteration {i+1}/{iterations}: {elapsed_ms:.4f} ms")

        # Compute statistics
        times_array = np.array(times_ms)
        mean_time = float(np.mean(times_array))
        std_time = float(np.std(times_array))
        min_time = float(np.min(times_array))
        max_time = float(np.max(times_array))

        # Determine dtype
        if benchmark_name == "fft":
            dtype = "complex128"
        else:
            dtype = "float64"

        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            method=method,
            size=size,
            mean_time_ms=mean_time,
            std_time_ms=std_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            iterations=iterations,
            dtype=dtype
        )

        logger.info(f"  Mean time: {mean_time:.4f} ± {std_time:.4f} ms")

        return result


if __name__ == "__main__":
    # Test baseline implementations
    logger.info("Testing baseline implementations...")

    # Test matrix multiplication
    logger.info("\n" + "="*80)
    logger.info("Matrix Multiplication Test")
    logger.info("="*80)

    sizes_to_test = [128, 256]

    for size in sizes_to_test:
        result = BaselineBenchmarks.run_benchmark(
            benchmark_name="matrix_multiply",
            method="numpy",
            size=size,
            iterations=5,
            warmup=2
        )
        logger.info(f"\nSize {size}: {result.mean_time_ms:.4f} ± {result.std_time_ms:.4f} ms")

    # Test Cholesky
    logger.info("\n" + "="*80)
    logger.info("Cholesky Factorization Test")
    logger.info("="*80)

    for size in sizes_to_test:
        result = BaselineBenchmarks.run_benchmark(
            benchmark_name="cholesky",
            method="numpy",
            size=size,
            iterations=5,
            warmup=2
        )
        logger.info(f"\nSize {size}: {result.mean_time_ms:.4f} ± {result.std_time_ms:.4f} ms")

    # Test FFT
    logger.info("\n" + "="*80)
    logger.info("FFT Test")
    logger.info("="*80)

    fft_sizes = [1024, 4096]

    for size in fft_sizes:
        result = BaselineBenchmarks.run_benchmark(
            benchmark_name="fft",
            method="numpy",
            size=size,
            iterations=5,
            warmup=2
        )
        logger.info(f"\nSize {size}: {result.mean_time_ms:.4f} ± {result.std_time_ms:.4f} ms")

    logger.info("\n" + "="*80)
    logger.info("Baseline benchmarking complete!")
    logger.info("="*80)
