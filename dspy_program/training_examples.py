"""
Training examples for DSPy optimizers.

This module creates high-quality training examples that teach DSPy
how to generate optimized numerical computing code.

Each example consists of:
- Input: Problem description with optimization goals
- Output: Expected optimized code with explanations

These examples are used by Bootstrap, MIPROv2, and GEPA optimizers
to learn effective prompting strategies.
"""

import dspy
from typing import List, Dict


class TrainingExample:
    """Container for a single training example."""

    def __init__(self,
                 problem_description: str,
                 optimization_goals: str,
                 expected_code: str,
                 optimization_notes: str = ""):
        self.problem_description = problem_description
        self.optimization_goals = optimization_goals
        self.expected_code = expected_code
        self.optimization_notes = optimization_notes

    def to_dspy_example(self, benchmark_type: str = "general") -> dspy.Example:
        """Convert to DSPy Example format."""
        if benchmark_type == "matrix_multiply":
            return dspy.Example(
                matrix_size="NxN",
                optimization_techniques=self.optimization_goals,
                optimized_code=self.expected_code
            ).with_inputs("matrix_size", "optimization_techniques")

        elif benchmark_type == "cholesky":
            return dspy.Example(
                matrix_size="NxN",
                optimization_techniques=self.optimization_goals,
                optimized_code=self.expected_code
            ).with_inputs("matrix_size", "optimization_techniques")

        elif benchmark_type == "fft":
            return dspy.Example(
                signal_length="N",
                optimization_techniques=self.optimization_goals,
                optimized_code=self.expected_code
            ).with_inputs("signal_length", "optimization_techniques")

        else:  # general
            return dspy.Example(
                problem_description=self.problem_description,
                optimization_goals=self.optimization_goals,
                constraints="Python with NumPy, executable code",
                optimized_code=self.expected_code,
                optimization_rationale=self.optimization_notes
            ).with_inputs("problem_description", "optimization_goals", "constraints")


# Matrix Multiplication Training Examples
MATRIX_MULTIPLY_EXAMPLES = [
    TrainingExample(
        problem_description="Implement matrix multiplication C = A @ B for dense NxN matrices",
        optimization_goals="Minimize execution time using NumPy optimizations",
        expected_code="""import numpy as np

def optimized_matmul(A, B):
    \"\"\"
    Optimized matrix multiplication using NumPy's highly optimized matmul.

    NumPy's matmul uses:
    - BLAS backend (OpenBLAS, MKL, etc.) for optimal performance
    - Automatic vectorization (SIMD instructions)
    - Multi-threading when beneficial
    - Cache-efficient algorithms
    \"\"\"
    # Use @ operator or np.matmul for best performance
    return A @ B
""",
        optimization_notes="NumPy's matmul is already highly optimized and uses BLAS libraries."
    ),

    TrainingExample(
        problem_description="Matrix multiplication with explicit loop structure for understanding",
        optimization_goals="Use NumPy vectorization instead of Python loops",
        expected_code="""import numpy as np

def optimized_matmul_vectorized(A, B):
    \"\"\"
    Vectorized matrix multiplication.

    Key optimizations:
    - Avoid explicit Python loops (slow)
    - Use NumPy's vectorized operations
    - Leverage BLAS for performance
    \"\"\"
    # Bad: Python loops
    # for i in range(n):
    #     for j in range(n):
    #         for k in range(n):
    #             C[i,j] += A[i,k] * B[k,j]

    # Good: NumPy vectorization
    return np.matmul(A, B)
""",
        optimization_notes="Vectorization is critical - Python loops are 100x+ slower"
    ),

    TrainingExample(
        problem_description="Matrix multiplication with memory layout optimization",
        optimization_goals="Optimize for cache efficiency and memory access patterns",
        expected_code="""import numpy as np

def optimized_matmul_cache_friendly(A, B):
    \"\"\"
    Cache-friendly matrix multiplication.

    Optimizations:
    - Ensure C-contiguous arrays (row-major) for NumPy
    - Minimize cache misses through proper memory layout
    - Use in-place operations where possible
    \"\"\"
    # Ensure proper memory layout
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    if not B.flags['C_CONTIGUOUS']:
        B = np.ascontiguousarray(B)

    # NumPy matmul handles cache optimization internally
    return np.matmul(A, B)
""",
        optimization_notes="Memory layout affects cache performance significantly"
    ),

    TrainingExample(
        problem_description="Matrix multiplication with output pre-allocation",
        optimization_goals="Minimize memory allocations for better performance",
        expected_code="""import numpy as np

def optimized_matmul_preallocated(A, B, C=None):
    \"\"\"
    Matrix multiplication with output pre-allocation.

    Optimization: Reuse output array to avoid allocation overhead.
    \"\"\"
    n = A.shape[0]

    # Pre-allocate output if not provided
    if C is None:
        C = np.empty((n, n), dtype=A.dtype)

    # Use out parameter for in-place result
    np.matmul(A, B, out=C)

    return C
""",
        optimization_notes="Pre-allocation avoids repeated memory allocation in loops"
    ),

    TrainingExample(
        problem_description="Matrix multiplication with type optimization",
        optimization_goals="Use appropriate data types for optimal performance",
        expected_code="""import numpy as np

def optimized_matmul_typed(A, B, dtype=np.float64):
    \"\"\"
    Type-optimized matrix multiplication.

    Key points:
    - float64 for precision (standard in scientific computing)
    - Consistent types avoid conversion overhead
    - BLAS optimized for float32 and float64
    \"\"\"
    # Ensure consistent types
    A = A.astype(dtype, copy=False)
    B = B.astype(dtype, copy=False)

    return np.matmul(A, B)
""",
        optimization_notes="Consistent types avoid costly conversions during computation"
    )
]


# Cholesky Factorization Training Examples
CHOLESKY_EXAMPLES = [
    TrainingExample(
        problem_description="Implement Cholesky factorization for symmetric positive-definite matrix",
        optimization_goals="Use NumPy's optimized LAPACK-based implementation",
        expected_code="""import numpy as np

def optimized_cholesky(A):
    \"\"\"
    Optimized Cholesky factorization using NumPy.

    NumPy's cholesky uses:
    - LAPACK library (dpotrf routine)
    - Cache-blocking algorithms
    - Numerically stable implementation
    \"\"\"
    # NumPy's cholesky is already highly optimized
    L = np.linalg.cholesky(A)
    return L
""",
        optimization_notes="NumPy's cholesky uses optimized LAPACK routines"
    ),

    TrainingExample(
        problem_description="Cholesky factorization with input validation",
        optimization_goals="Ensure numerical stability and symmetric positive-definite input",
        expected_code="""import numpy as np

def optimized_cholesky_validated(A):
    \"\"\"
    Cholesky factorization with validation.

    Optimizations:
    - Check symmetry (Cholesky requires symmetric input)
    - Add small diagonal term for numerical stability if needed
    - Use lower=True for consistency
    \"\"\"
    # Check if symmetric (within tolerance)
    if not np.allclose(A, A.T):
        # Make symmetric if close
        A = (A + A.T) / 2

    # Perform Cholesky factorization
    L = np.linalg.cholesky(A)

    return L
""",
        optimization_notes="Validation prevents failures and improves robustness"
    ),

    TrainingExample(
        problem_description="Cholesky with memory efficiency",
        optimization_goals="Minimize memory usage through in-place operations",
        expected_code="""import numpy as np
from scipy import linalg

def optimized_cholesky_inplace(A, overwrite_a=True):
    \"\"\"
    Memory-efficient Cholesky factorization.

    Optimization: Overwrite input to save memory.
    Note: Input matrix is destroyed!
    \"\"\"
    # SciPy allows overwriting input to save memory
    L = linalg.cholesky(A, lower=True, overwrite_a=overwrite_a)

    return L
""",
        optimization_notes="In-place operations reduce memory footprint for large matrices"
    ),

    TrainingExample(
        problem_description="Cholesky factorization for multiple matrices",
        optimization_goals="Vectorize over batch dimension for efficiency",
        expected_code="""import numpy as np

def optimized_cholesky_batch(matrices):
    \"\"\"
    Batch Cholesky factorization.

    Optimization: Process multiple matrices efficiently.
    \"\"\"
    # Use list comprehension for clarity
    # NumPy will vectorize internally
    results = [np.linalg.cholesky(A) for A in matrices]

    return results
""",
        optimization_notes="Batching amortizes overhead and improves cache usage"
    )
]


# FFT Training Examples
FFT_EXAMPLES = [
    TrainingExample(
        problem_description="Implement Fast Fourier Transform for 1D signal",
        optimization_goals="Use NumPy's optimized FFT implementation",
        expected_code="""import numpy as np

def optimized_fft(signal):
    \"\"\"
    Optimized FFT using NumPy.

    NumPy's FFT uses:
    - Cooley-Tukey algorithm (O(n log n))
    - Optimized for power-of-2 lengths
    - FFTW library backend when available
    - Cache-efficient implementation
    \"\"\"
    return np.fft.fft(signal)
""",
        optimization_notes="NumPy's FFT is already highly optimized"
    ),

    TrainingExample(
        problem_description="FFT with power-of-2 optimization",
        optimization_goals="Ensure input length is power of 2 for optimal performance",
        expected_code="""import numpy as np

def optimized_fft_power_of_2(signal):
    \"\"\"
    FFT optimized for power-of-2 lengths.

    Optimization: Pad to nearest power of 2 for best performance.
    \"\"\"
    n = len(signal)

    # Find next power of 2
    n_fft = 2 ** int(np.ceil(np.log2(n)))

    # Pad with zeros if necessary
    if n < n_fft:
        signal_padded = np.pad(signal, (0, n_fft - n), mode='constant')
    else:
        signal_padded = signal

    # Compute FFT (fastest for power-of-2 lengths)
    spectrum = np.fft.fft(signal_padded)

    return spectrum[:n]  # Return only original length
""",
        optimization_notes="Power-of-2 lengths enable radix-2 algorithm (fastest variant)"
    ),

    TrainingExample(
        problem_description="FFT with real-valued signal optimization",
        optimization_goals="Use rfft for real signals to save computation",
        expected_code="""import numpy as np

def optimized_fft_real(signal):
    \"\"\"
    Optimized FFT for real-valued signals.

    Optimization: Use rfft instead of fft for real signals.
    - Computes only positive frequencies
    - Saves 50% computation (uses conjugate symmetry)
    - Saves 50% memory
    \"\"\"
    # For real signals, use rfft (real FFT)
    if np.isrealobj(signal):
        spectrum = np.fft.rfft(signal)
    else:
        spectrum = np.fft.fft(signal)

    return spectrum
""",
        optimization_notes="rfft exploits conjugate symmetry of real signals for 2x speedup"
    ),

    TrainingExample(
        problem_description="FFT with normalization optimization",
        optimization_goals="Choose efficient normalization strategy",
        expected_code="""import numpy as np

def optimized_fft_normalized(signal, norm='ortho'):
    \"\"\"
    FFT with normalization.

    Normalization modes:
    - None: No normalization (fastest)
    - 'ortho': Orthonormal scaling (1/sqrt(n))
    - 'forward': Scale by 1/n on forward transform
    \"\"\"
    # Use norm parameter for efficiency
    spectrum = np.fft.fft(signal, norm=norm)

    return spectrum
""",
        optimization_notes="Choose normalization based on application needs"
    )
]


def get_training_examples(benchmark_name: str, num_examples: int = None) -> List[dspy.Example]:
    """
    Get training examples for a specific benchmark.

    Args:
        benchmark_name: Name of the benchmark
        num_examples: Number of examples to return (None = all)

    Returns:
        List of DSPy Example objects
    """
    examples_map = {
        "matrix_multiply": MATRIX_MULTIPLY_EXAMPLES,
        "cholesky": CHOLESKY_EXAMPLES,
        "fft": FFT_EXAMPLES
    }

    training_examples = examples_map.get(benchmark_name, [])

    if num_examples is not None:
        training_examples = training_examples[:num_examples]

    # Convert to DSPy Example format
    dspy_examples = [ex.to_dspy_example(benchmark_name) for ex in training_examples]

    return dspy_examples


def get_all_training_examples() -> Dict[str, List[dspy.Example]]:
    """Get all training examples for all benchmarks."""
    return {
        "matrix_multiply": get_training_examples("matrix_multiply"),
        "cholesky": get_training_examples("cholesky"),
        "fft": get_training_examples("fft")
    }


if __name__ == "__main__":
    print("Testing Training Examples...")

    for benchmark in ["matrix_multiply", "cholesky", "fft"]:
        examples = get_training_examples(benchmark)
        print(f"\n{benchmark}: {len(examples)} examples")

        if examples:
            print(f"  First example inputs: {list(examples[0].inputs().keys())}")
            print(f"  First example outputs: {list(examples[0].labels().keys())}")

    print("\nTraining examples created successfully!")
