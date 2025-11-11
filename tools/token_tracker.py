"""
Token usage tracking utility for DSPy experiments.
Tracks LLM API calls and provides comprehensive token usage statistics.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import List
from datetime import datetime

from tools.logger import get_logger


logger = get_logger(__name__)


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    call_type: str = "unknown"  # "compilation" or "inference"
    model: str = "unknown"

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class PhaseTokenUsage:
    """Aggregated token usage for a phase (compilation or inference)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    num_calls: int = 0

    def add(self, usage: TokenUsage):
        """Add token usage from a single call."""
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.num_calls += 1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class TokenTracker:
    """
    Centralized token tracking for DSPy experiments.

    Tracks token usage across compilation and inference phases,
    providing detailed statistics and cost estimation.
    """

    def __init__(self, benchmark_name: str = "", optimizer_name: str = ""):
        """
        Initialize token tracker.

        Args:
            benchmark_name: Name of the benchmark being run
            optimizer_name: Name of the optimizer being used
        """
        self.benchmark_name = benchmark_name
        self.optimizer_name = optimizer_name

        self.compilation_usage = PhaseTokenUsage()
        self.inference_usage = PhaseTokenUsage()

        self.all_calls: List[TokenUsage] = []
        self.start_time = time.time()

        logger.debug(f"TokenTracker initialized for {benchmark_name} with {optimizer_name}")

    def track_call(self,
                   prompt_tokens: int,
                   completion_tokens: int,
                   call_type: str = "inference",
                   model: str = "unknown"):
        """
        Track a single LLM API call.

        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            call_type: Type of call ("compilation" or "inference")
            model: Model name used for the call
        """
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            call_type=call_type,
            model=model
        )

        self.all_calls.append(usage)

        if call_type == "compilation":
            self.compilation_usage.add(usage)
            logger.debug(f"Compilation call: {prompt_tokens} prompt + {completion_tokens} completion = {usage.total_tokens} total tokens")
        else:
            self.inference_usage.add(usage)
            logger.debug(f"Inference call: {prompt_tokens} prompt + {completion_tokens} completion = {usage.total_tokens} total tokens")

    def get_total_tokens(self) -> int:
        """Get total tokens used across all calls."""
        return self.compilation_usage.total_tokens + self.inference_usage.total_tokens

    def get_summary(self) -> dict:
        """
        Get comprehensive token usage summary.

        Returns:
            Dictionary with detailed token usage statistics
        """
        elapsed_time = time.time() - self.start_time

        # Get model from first call if available
        model_name = "unknown"
        if self.all_calls:
            model_name = self.all_calls[0].model

        return {
            "benchmark": self.benchmark_name,
            "optimizer": self.optimizer_name,
            "model": model_name,  # Add model information
            "compilation": self.compilation_usage.to_dict(),
            "inference": self.inference_usage.to_dict(),
            "total_tokens": self.get_total_tokens(),
            "total_calls": len(self.all_calls),
            "elapsed_time_seconds": round(elapsed_time, 2),
            "timestamp": datetime.now().isoformat()
        }

    def estimate_cost(self, model: str = "gpt-4.1-mini") -> dict:
        """
        Estimate API cost based on token usage.

        Args:
            model: Model name for cost calculation

        Returns:
            Dictionary with cost breakdown

        Note:
            Pricing updated as of November 2025.
            Sources:
            - OpenAI: https://openai.com/api/pricing/
            - Anthropic: https://www.anthropic.com/pricing
        """
        # Pricing per 1K tokens (updated November 2025)
        pricing = {
            # OpenAI GPT-4.1 Family
            "gpt-4.1": {
                "prompt": 0.002,      # $2.00 / 1M tokens
                "completion": 0.008   # $8.00 / 1M tokens
            },
            "gpt-4.1-mini": {
                "prompt": 0.0004,     # $0.40 / 1M tokens (DEFAULT)
                "completion": 0.0016  # $1.60 / 1M tokens
            },
            "gpt-4.1-nano": {
                "prompt": 0.0001,     # $0.10 / 1M tokens (Cheapest)
                "completion": 0.0004  # $0.40 / 1M tokens
            },

            # OpenAI GPT-5 Mini
            "gpt-5-mini": {
                "prompt": 0.00025,    # $0.25 / 1M tokens
                "completion": 0.002   # $2.00 / 1M tokens
            },

            # Anthropic Claude Sonnet 4 Family
            "claude-sonnet-4.5": {
                "prompt": 0.003,      # $3.00 / 1M tokens
                "completion": 0.015   # $15.00 / 1M tokens
            },
            "claude-sonnet-4": {
                "prompt": 0.003,      # $3.00 / 1M tokens
                "completion": 0.015   # $15.00 / 1M tokens
            },
        }

        if model not in pricing:
            logger.warning(f"Unknown model {model}, using gpt-4.1-mini pricing")
            model = "gpt-4.1-mini"

        rates = pricing[model]

        compilation_cost = (
            (self.compilation_usage.prompt_tokens / 1000) * rates["prompt"] +
            (self.compilation_usage.completion_tokens / 1000) * rates["completion"]
        )

        inference_cost = (
            (self.inference_usage.prompt_tokens / 1000) * rates["prompt"] +
            (self.inference_usage.completion_tokens / 1000) * rates["completion"]
        )

        total_cost = compilation_cost + inference_cost

        return {
            "model": model,
            "compilation_cost_usd": round(compilation_cost, 4),
            "inference_cost_usd": round(inference_cost, 4),
            "total_cost_usd": round(total_cost, 4),
            "cost_per_call_usd": round(total_cost / max(len(self.all_calls), 1), 4)
        }

    def log_summary(self):
        """Log token usage summary to console and file."""
        summary = self.get_summary()

        logger.info(
            f"\n{'=' * 80}\n"
            f"Token Usage Summary: {self.benchmark_name} - {self.optimizer_name}\n"
            f"{'=' * 80}\n"
            f"\n"
            f"Compilation Phase:\n"
            f"  Prompt tokens:     {summary['compilation']['prompt_tokens']:>10,}\n"
            f"  Completion tokens: {summary['compilation']['completion_tokens']:>10,}\n"
            f"  Total tokens:      {summary['compilation']['total_tokens']:>10,}\n"
            f"  API calls:         {summary['compilation']['num_calls']:>10,}\n"
            f"\n"
            f"Inference Phase:\n"
            f"  Prompt tokens:     {summary['inference']['prompt_tokens']:>10,}\n"
            f"  Completion tokens: {summary['inference']['completion_tokens']:>10,}\n"
            f"  Total tokens:      {summary['inference']['total_tokens']:>10,}\n"
            f"  API calls:         {summary['inference']['num_calls']:>10,}\n"
            f"\n"
            f"Overall:\n"
            f"  Total tokens:      {summary['total_tokens']:>10,}\n"
            f"  Total API calls:   {summary['total_calls']:>10,}\n"
            f"  Elapsed time:      {summary['elapsed_time_seconds']:>10.2f} seconds\n"
            f"{'=' * 80}\n"
        )

    @staticmethod
    def extract_from_dspy_history(history: list) -> 'TokenTracker':
        """
        Extract token usage from DSPy history.

        Args:
            history: DSPy's LM call history

        Returns:
            TokenTracker with populated data
        """
        tracker = TokenTracker()

        for entry in history:
            # DSPy history format may vary, adapt as needed
            if hasattr(entry, 'usage'):
                usage = entry.usage
                tracker.track_call(
                    prompt_tokens=getattr(usage, 'prompt_tokens', 0),
                    completion_tokens=getattr(usage, 'completion_tokens', 0),
                    call_type="inference",
                    model=getattr(entry, 'model', 'unknown')
                )

        return tracker


if __name__ == "__main__":
    # Test the token tracker
    tracker = TokenTracker("matrix_multiply", "miprov2")

    # Simulate some API calls
    tracker.track_call(prompt_tokens=850, completion_tokens=420, call_type="compilation")
    tracker.track_call(prompt_tokens=1200, completion_tokens=680, call_type="compilation")
    tracker.track_call(prompt_tokens=950, completion_tokens=320, call_type="inference")

    # Log summary
    tracker.log_summary()

    # Estimate cost
    cost = tracker.estimate_cost("gpt-3.5-turbo")
    logger.info(f"\nEstimated cost: ${cost['total_cost_usd']:.4f}")
