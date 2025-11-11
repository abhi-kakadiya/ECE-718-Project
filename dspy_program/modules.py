"""
DSPy Module definitions for code generation.

Modules are the building blocks that execute signatures.
They can be optimized by DSPy's teleprompters.
"""

import dspy
from dspy_program.signatures import (
    CodeOptimizationSignature,
    MatrixMultiplyOptimization,
    CholeskyOptimization,
    FFTOptimization
)


class CodeGenerator(dspy.Module):
    """
    Basic code generator module using Chain of Thought.

    This module generates optimized code with step-by-step reasoning.
    """

    def __init__(self, signature_class=CodeOptimizationSignature):
        super().__init__()
        self.signature_class = signature_class
        # Use ChainOfThought for better reasoning
        self.generate = dspy.ChainOfThought(signature_class)

    def forward(self, **kwargs):
        """Generate code given inputs."""
        return self.generate(**kwargs)


class MatrixMultiplyGenerator(dspy.Module):
    """Specialized module for matrix multiplication code generation."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(MatrixMultiplyOptimization)

    def forward(self, matrix_size: str, optimization_techniques: str):
        """Generate optimized matrix multiplication code."""
        return self.generate(
            matrix_size=matrix_size,
            optimization_techniques=optimization_techniques
        )


class CholeskyGenerator(dspy.Module):
    """Specialized module for Cholesky factorization code generation."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(CholeskyOptimization)

    def forward(self, matrix_size: str, optimization_techniques: str):
        """Generate optimized Cholesky factorization code."""
        return self.generate(
            matrix_size=matrix_size,
            optimization_techniques=optimization_techniques
        )


class FFTGenerator(dspy.Module):
    """Specialized module for FFT code generation."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(FFTOptimization)

    def forward(self, signal_length: str, optimization_techniques: str):
        """Generate optimized FFT code."""
        return self.generate(
            signal_length=signal_length,
            optimization_techniques=optimization_techniques
        )


def get_module_for_benchmark(benchmark_name: str) -> dspy.Module:
    """
    Get the appropriate module for a benchmark.

    Args:
        benchmark_name: Name of the benchmark

    Returns:
        DSPy Module instance for the benchmark
    """
    modules = {
        "matrix_multiply": MatrixMultiplyGenerator(),
        "cholesky": CholeskyGenerator(),
        "fft": FFTGenerator()
    }

    return modules.get(benchmark_name, CodeGenerator())


if __name__ == "__main__":
    print("Testing DSPy Modules...")

    # Test module creation
    matmul_module = MatrixMultiplyGenerator()
    print("\n1. MatrixMultiplyGenerator created")
    print(f"   Type: {type(matmul_module)}")

    cholesky_module = CholeskyGenerator()
    print("\n2. CholeskyGenerator created")
    print(f"   Type: {type(cholesky_module)}")

    fft_module = FFTGenerator()
    print("\n3. FFTGenerator created")
    print(f"   Type: {type(fft_module)}")

    print("\nModules created successfully!")
