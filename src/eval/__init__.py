"""
Evaluation module for Agentic Memory.

This module provides evaluators for different datasets with a unified interface.
The evaluator handles prompt building, LLM calls, and metric computation.

Usage:
    from src.eval import get_evaluator

    # Get evaluator for a specific dataset
    evaluator = get_evaluator("locomo", args)

    # Evaluate on QA list
    summary = evaluator.evaluate(qa_list, memory_bank)

    # Access results
    print(f"Average F1: {summary.avg_f1}")
"""
from .base import (
    Evaluator,
    EvalResult,
    EvalSummary,
    get_evaluator,
    register_evaluator,
    list_evaluators,
)
from .locomo import LoCoMoEvaluator
from .longmemeval import LongMemEvalEvaluator
from .hotpotqa import HotpotQAEvaluator
from .alfworld import ALFWorldEvaluator


__all__ = [
    # Base classes
    "Evaluator",
    "EvalResult",
    "EvalSummary",
    # Registry functions
    "get_evaluator",
    "register_evaluator",
    "list_evaluators",
    # Concrete evaluators
    "LoCoMoEvaluator",
    "LongMemEvalEvaluator",
    "HotpotQAEvaluator",
    "ALFWorldEvaluator",
]
