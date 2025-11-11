# DSPy HPC Benchmarking Project

**Prompt Engineering for Numerical Datasets using DSPy**

ECE 718 - Compiler Design for High Performance Computing

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [What is DSPy?](#what-is-dspy)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Usage Guide](#usage-guide)
7. [Benchmarks](#benchmarks)
8. [DSPy Optimizers](#dspy-optimizers)
9. [Results and Tables](#results-and-tables)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

---

## Overview

This project demonstrates **systematic prompt optimization** for high-performance numerical computing code generation using **DSPy** (Stanford NLP framework). We evaluate three DSPy optimization strategies on classic HPC benchmarks:

- **Matrix Multiplication** (BLAS Level 3)
- **Cholesky Factorization** (Linear Algebra)
- **Fast Fourier Transform** (Signal Processing)

### Key Innovation

Instead of manual prompt engineering, DSPy treats prompts as **compiled artifacts** that can be automatically optimized using compiler-inspired techniques like:
- Few-shot learning (Bootstrap)
- Bayesian optimization (MIPROv2)
- Instruction tuning

---

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key (for GPT models)
- 2GB disk space for dependencies

### 5-Minute Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ECE-718-Project.git
cd ECE-718-Project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Run quick test (< 1 minute, < $0.01)
python main.py --quick-test
```

If the quick test passes, you're ready to go!

### Run Your First Benchmark

```bash
# Interactive mode (recommended for first time)
python main.py

# Choose option 4: Run single benchmark
# Select benchmark: 1 (matrix_multiply)
# Wait 10-20 minutes
# Results saved to: results/matrix_multiply/
```

### Command Line Usage

```bash
# Single benchmark
python main.py --benchmark matrix_multiply

# All benchmarks (1-2 hours, $0.40-1.00)
python main.py --all

# Check environment
python main.py --check

# Generate visualizations only
python main.py --visualize
```

---

## What is DSPy?

**DSPy** is a framework for programming with language models developed by Stanford NLP.

### Traditional vs DSPy Approach

**Traditional Prompt Engineering:**
```
You write: "Optimize this matrix multiplication code..."
LLM generates: Code (quality depends on your prompt)
Result: Manual trial-and-error, brittle prompts
```

**DSPy Approach:**
```python
# 1. Define signature (what to do)
class CodeOptimization(dspy.Signature):
    problem_description = dspy.InputField()
    optimized_code = dspy.OutputField()

# 2. Create module (how to do it)
optimizer = dspy.ChainOfThought(CodeOptimization)

# 3. Compile with examples (automatic optimization)
compiled_optimizer = dspy.BootstrapFewShot().compile(
    student=optimizer,
    trainset=examples
)
```

**Result:** Systematic optimization, robust prompts, measurable improvements

### DSPy Key Concepts

1. **Signatures**: Declarative specifications of LLM tasks (like type hints)
2. **Modules**: Reusable LLM operations (ChainOfThought, ReAct, etc.)
3. **Optimizers/Teleprompters**: Automatic prompt compilation strategies
4. **Metrics**: Quantitative evaluation of generated outputs

---

## Project Structure

```
ECE-718-Project/
├── main.py                      # Main entry point (CLI interface)
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment configuration template
│
├── benchmarks/                  # NumPy baseline implementations
│   └── baseline_implementations.py
│
├── dspy_program/                # DSPy components
│   ├── signatures.py            # Task specifications
│   ├── modules.py               # DSPy modules
│   ├── baseline_optimizer.py   # No optimization (control)
│   ├── bootstrap_optimizer.py  # Few-shot learning
│   ├── miprov2_optimizer.py    # Bayesian optimization
│   ├── training_examples.py    # Few-shot examples
│   └── metrics.py               # Evaluation metrics
│
├── tools/                       # Utilities
│   ├── run_all_benchmarks.py   # Orchestration
│   ├── code_executor.py        # Safe code execution
│   ├── token_tracker.py        # LLM cost tracking
│   ├── visualize_results.py    # Chart generation
│   ├── generate_tables.py      # Table generation
│   ├── logger.py               # Logging utilities
│   └── pretty_output.py        # CLI formatting
│
├── configs/                     # Configuration files
│   └── benchmarks.yaml
│
├── docs/                        # Documentation
│   ├── TABLE_GENERATION_GUIDE.md
│   └── USAGE_GUIDE.md
│
├── results/                     # Generated results (created on run)
│   ├── matrix_multiply/
│   ├── cholesky/
│   ├── fft/
│   └── cross_benchmark_comparison.csv
│
├── figures/                     # Generated visualizations
│   ├── token_usage_comparison.png
│   ├── performance_comparison.png
│   └── speedup_analysis.png
│
├── logs/                        # Execution logs
└── cache/                       # DSPy cache (cleared on each run)
```

---

## Installation

### System Requirements

- **OS**: Linux, macOS, or Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 2GB for dependencies + space for results
- **Internet**: Required for LLM API calls

### Step-by-Step Installation

#### 1. Clone Repository

```bash
git clone https://github.com/your-username/ECE-718-Project.git
cd ECE-718-Project
```

#### 2. Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Dependencies include:**
- `dspy-ai==2.6.5` - DSPy framework
- `numpy` - Numerical computing
- `pandas` - Data analysis
- `matplotlib` - Visualization
- `pyyaml` - Configuration
- `python-dotenv` - Environment management
- `openai` - OpenAI API client

#### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```bash
# OpenAI API key (required)
OPENAI_API_KEY=sk-your-api-key-here

# Model selection (optional)
LLM_MODEL=gpt-4.1-mini

# Temperature (optional)
LLM_TEMPERATURE=0.7
```

**Getting an OpenAI API Key:**
1. Go to https://platform.openai.com/
2. Sign up or log in
3. Navigate to API Keys section
4. Create new secret key
5. Copy and paste into `.env`

#### 5. Verify Installation

```bash
python main.py --check
```

Expected output:
```
Checking Environment
✓ .env file found
✓ API key configured
✓ results/ directory exists
✓ figures/ directory exists
✓ logs/ directory exists
✓ cache/ directory exists

Environment is properly configured!
```

#### 6. Run Quick Test

```bash
python main.py --quick-test
```

This runs a minimal benchmark to verify everything works (< 1 minute, < $0.01).

---

## Usage Guide

### Interactive Mode (Recommended for Beginners)

```bash
python main.py
```

**Menu Options:**
```
1. Check environment setup
2. Show current configuration
3. Run quick test (< 1 minute, < $0.01)
4. Run single benchmark (10-20 min, $0.03-0.10)
5. Run all benchmarks (1-2 hours, $0.40-1.00)
6. Generate visualizations from existing results
0. Exit
```

### Command Line Mode

#### Single Benchmark

```bash
python main.py --benchmark matrix_multiply
```

**What happens:**
1. Clears old results for fresh run
2. Runs NumPy baseline (reference performance)
3. Runs DSPy optimizers:
   - Baseline (no optimization)
   - Bootstrap (few-shot learning)
   - MIPROv2 (Bayesian optimization)
4. Generates visualizations
5. Creates 5 comparison tables

**Time:** 10-20 minutes
**Cost:** $0.03-0.10
**Output:**
- `results/matrix_multiply/*.json` - Raw results
- `results/matrix_multiply/*.csv` - Comparison tables
- `figures/*.png` - Visualizations

#### All Benchmarks

```bash
python main.py --all
```

**What happens:**
1. Runs all 3 benchmarks (matrix_multiply, cholesky, fft)
2. Individual tables for each benchmark
3. Cross-benchmark comparison table
4. Combined visualizations

**Time:** 1-2 hours
**Cost:** $0.40-1.00
**Output:**
- Individual results for each benchmark
- `results/cross_benchmark_comparison.csv`
- Combined visualizations

#### Other Commands

```bash
# Check environment
python main.py --check

# Show configuration
python main.py --config

# Generate tables from existing results
python tools/generate_tables.py matrix_multiply

# Generate all tables
python tools/generate_tables.py
```

### Understanding Results

After running benchmarks, you'll find:

**1. JSON Results** (`results/<benchmark>/*_results.json`)
- Raw execution data
- Token usage
- Generated code
- Error messages (if any)

**2. CSV Tables** (`results/<benchmark>/*.csv`)
- `comparison_summary.csv` - Quick overview
- `performance_details.csv` - Execution statistics
- `token_usage.csv` - LLM costs
- `cost_benefit.csv` - ROI analysis
- `speedup_analysis.csv` - Performance vs NumPy

**3. Visualizations** (`figures/*.png`)
- Token usage comparison
- Performance comparison
- Speedup analysis

**4. Logs** (`logs/experiment.log`)
- Detailed execution trace
- Debugging information

---

## Benchmarks

### 1. Matrix Multiplication

**Problem:** Compute C = A × B for N×N matrices

**NumPy Baseline:**
```python
def matrix_multiply(A, B):
    return np.dot(A, B)  # Uses optimized BLAS
```

**DSPy Task:** Generate optimized NumPy code for matrix multiplication

**Test Size:** 128×128 matrices
**Iterations:** 10 (with 3 warmup runs)
**Correctness:** Verified with `np.allclose(result, expected)`

### 2. Cholesky Factorization

**Problem:** Decompose symmetric positive-definite matrix A = L L^T

**NumPy Baseline:**
```python
def cholesky_factorization(A):
    return np.linalg.cholesky(A)  # Uses LAPACK
```

**DSPy Task:** Generate optimized Cholesky decomposition code

**Test Size:** 128×128 matrix
**Iterations:** 10 (with 3 warmup runs)
**Correctness:** Verified by reconstructing A from L

### 3. Fast Fourier Transform (FFT)

**Problem:** Compute discrete Fourier transform of signal

**NumPy Baseline:**
```python
def fft_transform(x):
    return np.fft.fft(x)  # Uses FFTW
```

**DSPy Task:** Generate optimized FFT code

**Test Size:** 128 points
**Iterations:** 10 (with 3 warmup runs)
**Correctness:** Verified with `np.allclose(result, expected)`

---

## DSPy Optimizers

We evaluate three DSPy optimization strategies:

### 1. Baseline (Control Group)

**Strategy:** No optimization - direct LLM invocation

**Approach:**
```python
optimizer = dspy.ChainOfThought(CodeOptimizationSignature)
result = optimizer(problem_description="...")
```

**Characteristics:**
- ✓ Fast compilation (< 1 second)
- ✓ Low token usage (~600 tokens)
- ✓ Low cost ($0.0003)
- ✗ No learning from examples
- ✗ Inconsistent quality

**Use Case:** Baseline for comparison

### 2. Bootstrap Few-Shot

**Strategy:** Learn from successful examples

**Approach:**
```python
optimizer = dspy.BootstrapFewShot(
    metric=code_quality_metric,
    max_bootstrapped_demos=3
)
compiled = optimizer.compile(
    student=program,
    trainset=training_examples
)
```

**Characteristics:**
- ✓ Learns from successful examples
- ✓ Moderate cost ($0.004)
- ✓ More consistent output
- ✗ Slower compilation (~1 second)
- ✗ Higher token usage (~9,000 tokens)

**Use Case:** Good balance of cost and quality

### 3. MIPROv2 (Bayesian Optimization)

**Strategy:** Optimize instructions AND demonstrations using Bayesian search

**Approach:**
```python
optimizer = dspy.MIPROv2(
    metric=code_quality_metric,
    num_candidates=10,
    init_temperature=1.0
)
compiled = optimizer.compile(
    student=program,
    trainset=training_examples,
    num_trials=25
)
```

**Characteristics:**
- ✓ Sophisticated optimization
- ✓ Potentially better quality
- ✗ Expensive ($0.08-0.10 per benchmark)
- ✗ Slow compilation (10-20 seconds)
- ✗ Very high token usage (~108,000 tokens)

**Use Case:** Research, finding optimal prompts

---

## Results and Tables

### Single Benchmark Results

When you run a single benchmark, you get **5 comprehensive tables**:

#### 1. Comparison Summary
Quick overview of all optimizers:
```
Optimizer | Execution | Correct | Time (ms) | Speedup  | Tokens | Compile (s) | Overall
----------|-----------|---------|-----------|----------|--------|-------------|--------
NUMPY     | ✓         | ✓       | 0.1234    | baseline | N/A    | N/A         | ⚪
BASELINE  | ✓         | ✓       | 0.2000    | 0.62x    | 610    | 0.00        | 🔴
BOOTSTRAP | ✓         | ✓       | 0.3200    | 0.39x    | 8,965  | 0.01        | 🔴
MIPROV2   | ✓         | ✓       | 0.1900    | 0.65x    | 108,566| 0.19        | 🔴
```

#### 2. Performance Details
Statistical analysis with variance:
```
Optimizer | Mean (ms) | Std Dev | Min    | Max    | Correctness | Success
----------|-----------|---------|--------|--------|-------------|--------
NumPy     | 0.1234    | 0.0050  | 0.1180 | 0.1300 | ✓           | ✓
Baseline  | 0.2000    | 0.0100  | 0.1850 | 0.2150 | ✓           | ✓
```

#### 3. Token Usage
LLM consumption and costs:
```
Optimizer | Compilation | Inference | Total   | Total Cost | API Calls | Compile Time
----------|-------------|-----------|---------|------------|-----------|-------------
Baseline  | 0           | 610       | 610     | $0.000366  | 1         | 0.00s
Bootstrap | 6,574       | 2,391     | 8,965   | $0.003845  | 8         | 0.01s
MIPROv2   | 107,295     | 1,271     | 108,566 | $0.081234  | 25        | 0.19s
```

#### 4. Cost-Benefit Analysis
Performance vs LLM costs:
```
Optimizer | Time (ms) | Speedup | Total Tokens | Cost      | Tokens/ms saved
----------|-----------|---------|--------------|-----------|----------------
Baseline  | 0.2000    | 0.62x   | 610          | $0.000366 | 7,922
Bootstrap | 0.3200    | 0.39x   | 8,965        | $0.003845 | N/A
MIPROv2   | 0.1900    | 0.65x   | 108,566      | $0.081234 | 1,588,529
```

#### 5. Speedup Analysis
Performance relative to NumPy:
```
Optimizer | Mean Time | Speedup | Performance | Status
----------|-----------|---------|-------------|-------------
NumPy     | 0.1234    | 1.00x   | 100%        | Reference
Baseline  | 0.2000    | 0.62x   | 62%         | 🔴 Needs Work
```

**Status Indicators:**
- 🟢 Excellent: ≥80% of NumPy performance
- 🟡 Good: ≥50% of NumPy performance
- 🔴 Needs Work: <50% of NumPy performance

### All Benchmarks Results

When you run all benchmarks, you also get a **cross-benchmark comparison table**:

```
Benchmark        | NumPy (ms) | Baseline (ms) | Bootstrap (ms) | MIPROv2 (ms) | Best
-----------------|------------|---------------|----------------|--------------|------
Matrix Multiply  | 0.1234     | 0.2000        | 0.3200         | 0.1900       | NumPy
Cholesky         | 0.1800     | 0.2500        | 0.2800         | 0.2200       | NumPy
FFT              | 0.1200     | 0.1800        | 0.2100         | 0.1600       | NumPy
```

This shows which optimizer performs best on each benchmark.

### Interpreting Results

**Speedup < 1.0x:** Optimizer is slower than NumPy (common)
- NumPy uses decades of hand-optimized C/Fortran code
- Current LLMs struggle to match this level of optimization

**Speedup > 1.0x:** Optimizer beats NumPy (rare but possible)
- Worth investigating what optimizations LLM discovered
- May apply to specific problem sizes or constraints

**Example Interpretation:**
- Speedup = 0.62x means optimizer takes 1.61x longer (1/0.62)
- Achieves 62% of NumPy's performance
- Status: 🔴 Needs Work

---

## Troubleshooting

### Environment Issues

**Problem:** `.env file not found`
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

**Problem:** `No API key configured`
```bash
# Check .env file
cat .env | grep OPENAI_API_KEY

# Should show: OPENAI_API_KEY=sk-your-key-here
```

**Problem:** `Module not found`
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Execution Issues

**Problem:** Optimizer shows "✗ Failed" or "N/A"

**Solution 1:** Run diagnostic
```bash
python debug_baseline.py
```

**Solution 2:** Check logs
```bash
tail -100 logs/experiment.log
```

**Solution 3:** Check result file
```bash
cat results/matrix_multiply/baseline_results.json
# Look for "execution_error" field
```

**Common Errors:**
- Function signature mismatch (fixed in latest version)
- Invalid Python code generation
- Timeout during execution

### API Issues

**Problem:** Rate limit errors

**Solution:** Add delays or reduce concurrent requests
```bash
# In .env, reduce number of trials
NUM_ITERATIONS=5  # Instead of 10
```

**Problem:** High costs

**Solution:**
- Use quick test first
- Run single benchmarks individually
- Monitor token usage in tables
- Consider using gpt-4.1-mini instead of gpt-4.1

### Results Issues

**Problem:** Missing NumPy baseline

**Solution:** NumPy baseline should run automatically
```bash
# Check that skip_baseline is not set
# Run fresh benchmark:
python main.py --benchmark matrix_multiply
```

**Problem:** Old results showing

**Solution:** Results are automatically cleared on each run
- Check that `clear_old_results()` is being called
- Manually delete: `rm -rf results/ figures/`

### Performance Issues

**Problem:** Benchmarks taking too long

**Solution:**
- Reduce iterations in `configs/benchmarks.yaml`
- Run quick test first to verify setup
- Run single benchmarks instead of all

---

## References

### DSPy

- **Paper:** [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714)
- **GitHub:** https://github.com/stanfordnlp/dspy
- **Documentation:** https://dspy-docs.vercel.app/

### Research Background

- **Few-Shot Learning:** [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3 paper)
- **Chain-of-Thought:** [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- **Optimization:** [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409)

### HPC Benchmarks

- **BLAS:** Basic Linear Algebra Subprograms
- **LAPACK:** Linear Algebra PACKage
- **FFTW:** Fastest Fourier Transform in the West

### Course Materials

- ECE 718: Compiler Design for High Performance Computing
- Instructor: [Your Instructor Name]
- Institution: [Your University]

---

## License

This project is for educational purposes as part of ECE 718 coursework.

---

## Contact

For questions or issues:
1. Check [Troubleshooting](#troubleshooting) section
2. Review logs in `logs/experiment.log`
3. Run diagnostic: `python debug_baseline.py`
4. Contact course staff

---

## Acknowledgments

- **DSPy Team** at Stanford NLP for the framework
- **OpenAI** for GPT models
- **NumPy/SciPy** communities for reference implementations
- ECE 718 course staff and fellow students

---

**Happy Optimizing! 🚀**
