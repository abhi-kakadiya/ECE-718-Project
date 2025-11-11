"""
DSPy Signature definitions for HPC code optimization.

Signatures are declarative specifications that define what a module should do,
without specifying how to do it. They act like type hints for LLM interactions.
"""

import dspy


class CodeOptimizationSignature(dspy.Signature):
    """
    Generate optimized numerical computing code.

    This signature defines the task of taking a problem description
    and generating high-performance code with specific optimizations.
    """

    problem_description = dspy.InputField(
        desc="Description of the numerical computation problem to solve"
    )

    optimization_goals = dspy.InputField(
        desc="Specific optimization objectives (e.g., minimize execution time, cache efficiency)"
    )

    constraints = dspy.InputField(
        desc="Constraints and requirements (e.g., language, libraries, data types)"
    )

    optimized_code = dspy.OutputField(
        desc="Complete, optimized, executable Python code with comments explaining optimizations"
    )

    optimization_rationale = dspy.OutputField(
        desc="Explanation of optimization techniques applied and why they improve performance"
    )


class MatrixMultiplyOptimization(dspy.Signature):
    """Generate optimized matrix multiplication code."""

    matrix_size = dspy.InputField(
        desc="Size of the square matrices (NxN)"
    )

    optimization_techniques = dspy.InputField(
        desc="Optimization techniques to apply (e.g., blocking, vectorization, parallelization)"
    )

    optimized_code = dspy.OutputField(
        desc="Complete Python function that takes two NxN matrices (A, B) as parameters and returns their product using NumPy optimizations. The function signature must be: def function_name(A, B): ... return result"
    )


class CholeskyOptimization(dspy.Signature):
    """Generate optimized Cholesky factorization code."""

    matrix_size = dspy.InputField(
        desc="Size of the symmetric positive-definite matrix (NxN)"
    )

    optimization_techniques = dspy.InputField(
        desc="Optimization techniques to apply (e.g., block algorithm, in-place computation)"
    )

    optimized_code = dspy.OutputField(
        desc="Complete Python function that takes a symmetric positive-definite matrix A as parameter and returns its Cholesky factorization using NumPy. The function signature must be: def function_name(A): ... return L"
    )


class FFTOptimization(dspy.Signature):
    """Generate optimized FFT code."""

    signal_length = dspy.InputField(
        desc="Length of the signal (number of points)"
    )

    optimization_techniques = dspy.InputField(
        desc="Optimization techniques to apply (e.g., radix-2 algorithm, cache-friendly access)"
    )

    optimized_code = dspy.OutputField(
        desc="Complete Python function that takes a signal array x as parameter and returns its FFT using NumPy. The function signature must be: def function_name(x): ... return fft_result"
    )


# Simpler signature for basic predictions
class SimpleCodeGeneration(dspy.Signature):
    """Simple code generation signature."""

    task_description: str = dspy.InputField()
    generated_code: str = dspy.OutputField()


# Signature for code quality evaluation (used in metrics)
class CodeQualityEvaluation(dspy.Signature):
    """Evaluate the quality of generated code."""

    code = dspy.InputField(desc="The code to evaluate")
    criteria = dspy.InputField(desc="Evaluation criteria")
    score = dspy.OutputField(desc="Quality score from 0-10")
    feedback = dspy.OutputField(desc="Specific feedback on code quality")


def get_signature_for_benchmark(benchmark_name: str) -> type:
    """
    Get the appropriate signature class for a benchmark.

    Args:
        benchmark_name: Name of the benchmark

    Returns:
        DSPy Signature class for the benchmark
    """
    signatures = {
        "matrix_multiply": MatrixMultiplyOptimization,
        "cholesky": CholeskyOptimization,
        "fft": FFTOptimization,
        "general": CodeOptimizationSignature
    }

    return signatures.get(benchmark_name, CodeOptimizationSignature)


if __name__ == "__main__":
    # Test signature definitions
    print("Testing DSPy Signatures...")

    print("\n1. CodeOptimizationSignature:")
    print(f"   Inputs: {list(CodeOptimizationSignature.input_fields.keys())}")
    print(f"   Outputs: {list(CodeOptimizationSignature.output_fields.keys())}")

    print("\n2. MatrixMultiplyOptimization:")
    print(f"   Inputs: {list(MatrixMultiplyOptimization.input_fields.keys())}")
    print(f"   Outputs: {list(MatrixMultiplyOptimization.output_fields.keys())}")

    print("\n3. CholeskyOptimization:")
    print(f"   Inputs: {list(CholeskyOptimization.input_fields.keys())}")
    print(f"   Outputs: {list(CholeskyOptimization.output_fields.keys())}")

    print("\n4. FFTOptimization:")
    print(f"   Inputs: {list(FFTOptimization.input_fields.keys())}")
    print(f"   Outputs: {list(FFTOptimization.output_fields.keys())}")

    print("\nSignatures defined successfully!")
