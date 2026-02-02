"""
Base classes for evaluation.

This module provides abstract interfaces for evaluation that can be
extended to support different datasets. The evaluator handles:
1. Building prompts from retrieved memories
2. Generating predictions using LLM
3. Computing evaluation metrics
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import copy
import numpy as np


@dataclass
class EvalResult:
    """
    Result of evaluating a single QA item.

    This provides a unified result format across different datasets.
    """
    qa_idx: int
    question: str
    ground_truth: str
    prediction: str
    retrieved_memories: List[str]
    retrieved_indices: List[int]
    f1_score: float = 0.0
    llm_judge_score: float = 0.0
    is_correct: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSummary:
    """Summary statistics from evaluation."""
    avg_f1: float = 0.0
    avg_llm_judge: float = 0.0
    num_samples: int = 0
    num_correct: int = 0
    accuracy: float = 0.0
    results: List[EvalResult] = field(default_factory=list)
    category_scores: Dict[Any, Dict[str, float]] = field(default_factory=dict)


class Evaluator(ABC):
    """
    Abstract base class for evaluation.

    Subclasses should implement methods for:
    1. Filtering QA items (which ones to skip)
    2. Building prompts from retrieved memories
    3. Any dataset-specific evaluation logic
    """

    def __init__(self, args):
        """
        Initialize the evaluator.

        Args:
            args: Arguments containing model config, etc.
        """
        self.args = args

    @abstractmethod
    def filter_qa_list(self, qa_list: List[Dict]) -> List[Tuple[int, Dict]]:
        """
        Filter QA list to get valid items for evaluation.

        Args:
            qa_list: List of QA dicts

        Returns:
            List of (index, qa_dict) tuples for valid QA items
        """
        pass

    @abstractmethod
    def build_prompt(self, question: str, retrieved_memories: List[str],
                     qa_item: Dict) -> str:
        """
        Build evaluation prompt from question and retrieved memories.

        Args:
            question: The question to answer
            retrieved_memories: List of retrieved memory texts
            qa_item: Original QA dict (may contain extra info like date)

        Returns:
            Formatted prompt string
        """
        pass

    def get_ground_truth(self, qa_item: Dict) -> str:
        """
        Extract ground truth answer from QA item.

        Override in subclasses if needed.

        Args:
            qa_item: QA dict

        Returns:
            Ground truth answer string
        """
        return str(qa_item.get('answer', ''))

    def prepare_eval_args(self) -> Any:
        """
        Prepare arguments for evaluation LLM calls.

        Override in subclasses to customize.

        Returns:
            Modified args for evaluation
        """
        eval_args = copy.deepcopy(self.args)
        eval_args.max_new_tokens = 32
        eval_args.temperature = 0.0
        eval_args.batch_size = 32
        return eval_args

    def compute_f1(self, prediction: str, ground_truth: str, qa_item: Dict = None) -> float:
        """
        Compute F1 score between prediction and ground truth.

        Subclasses may override to use category-specific F1 methods.

        Args:
            prediction: Model prediction
            ground_truth: Ground truth answer
            qa_item: Original QA dict (for category-specific handling)

        Returns:
            F1 score (0-1)
        """
        from eval_utils import f1_score
        return f1_score(prediction, ground_truth)

    def build_judge_prompt(self, question: str, ground_truth, prediction: str, qa_item: Dict) -> str:
        """
        Build LLM judge prompt for evaluation.

        Subclasses can override to provide task-specific judging criteria.
        """
        from prompts.prompt_pool import LLM_JUDGE_GENERAL_PROMPT

        if isinstance(ground_truth, list):
            ground_truth_str = ", ".join(str(ans) for ans in ground_truth)
        else:
            ground_truth_str = str(ground_truth)
        return LLM_JUDGE_GENERAL_PROMPT.format(
            question=question,
            ground_truth=ground_truth_str,
            model_answer=prediction
        )

    def evaluate(self, qa_list: List[Dict], memory_bank: Any,
                 conversation_id: str = None,
                 collect_cases: bool = False) -> EvalSummary:
        """
        Evaluate QA performance on a list of QA items.

        This is the main evaluation entry point. It handles:
        1. Filtering valid QA items
        2. Retrieving memories for each question
        3. Building prompts and calling LLM
        4. Computing metrics

        Args:
            qa_list: List of QA dicts
            memory_bank: MemoryBank to retrieve from
            conversation_id: Optional ID for tracking
            collect_cases: Whether to collect detailed cases

        Returns:
            EvalSummary with results and statistics
        """
        from llm_utils import get_llm_response
        from rag_utils import get_embeddings

        # Filter valid QA items
        valid_qa = self.filter_qa_list(qa_list)
        if not valid_qa:
            return EvalSummary()

        eval_args = self.prepare_eval_args()

        # Collect questions for batch embedding
        questions = [qa['question'] for _, qa in valid_qa]

        # Batch compute question embeddings
        q_embeddings = get_embeddings(
            self.args.retriever,
            questions,
            'query'
        )

        # Build task args for parallel LLM calls
        task_args = []
        retrieval_info = {}  # qa_idx -> (retrieved_mems, retrieved_indices)

        for idx, (qa_idx, qa) in enumerate(valid_qa):
            question = qa['question']
            q_embedding = q_embeddings[idx]

            # Retrieve from memory bank
            top_k = getattr(self.args, "mem_top_k_eval", None)
            if top_k is None:
                top_k = getattr(self.args, "mem_top_k", 5)
            retrieved_mems, retrieved_indices = memory_bank.retrieve(
                q_embedding, top_k=top_k, use_state_encoder=False
            )
            retrieval_info[qa_idx] = (retrieved_mems, list(retrieved_indices))

            # Build prompt
            prompt = self.build_prompt(question, retrieved_mems, qa)
            task_args.append((qa_idx, prompt, eval_args))

        if not task_args:
            return EvalSummary()

        # Call LLM in parallel
        ret = get_llm_response(args=eval_args, task_args=task_args)

        # Compute scores
        results = []
        f1_scores = []

        for i, response, _, success in ret:
            qa = qa_list[i]
            ground_truth = self.get_ground_truth(qa)
            prediction = response.strip() if success else ""

            f1 = self.compute_f1(prediction, ground_truth)
            f1_scores.append(f1)

            retrieved_mems, retrieved_indices = retrieval_info.get(i, ([], []))

            result = EvalResult(
                qa_idx=i,
                question=qa['question'],
                ground_truth=ground_truth,
                prediction=prediction,
                retrieved_memories=retrieved_mems,
                retrieved_indices=retrieved_indices,
                f1_score=f1,
                metadata=self._get_result_metadata(qa)
            )
            results.append(result)

        # Compute summary statistics
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0

        return EvalSummary(
            avg_f1=avg_f1,
            num_samples=len(results),
            results=results
        )

    def _get_result_metadata(self, qa: Dict) -> Dict[str, Any]:
        """
        Extract metadata from QA item for result.

        Override in subclasses to add dataset-specific metadata.

        Args:
            qa: QA dict

        Returns:
            Metadata dict
        """
        return {}

    def run_llm_judge(self, results: List[EvalResult]) -> List[float]:
        """
        Run LLM judge on evaluation results.

        Args:
            results: List of EvalResult objects

        Returns:
            List of LLM judge scores (0.0, 0.5, or 1.0)
        """
        from eval_utils import llm_judge
        from prompts.prompt_pool import LLM_JUDGE_GENERAL_PROMPT

        if not results:
            return []

        eval_args = self.prepare_eval_args()

        judge_task_args = []
        for result in results:
            prompt = LLM_JUDGE_GENERAL_PROMPT.format(
                question=result.question,
                ground_truth=result.ground_truth,
                model_answer=result.prediction
            )
            judge_task_args.append((result.qa_idx, prompt, eval_args))

        scores = llm_judge(task_args=judge_task_args, args=eval_args)

        # Update results with LLM judge scores
        for result, score in zip(results, scores):
            result.llm_judge_score = score
            result.is_correct = (score == 1.0)

        return scores


# Global registry for evaluators
_EVALUATOR_REGISTRY: Dict[str, type] = {}


def register_evaluator(name: str):
    """
    Decorator to register an evaluator class.

    Usage:
        @register_evaluator("my_dataset")
        class MyEvaluator(Evaluator):
            ...
    """
    def decorator(cls):
        _EVALUATOR_REGISTRY[name] = cls
        return cls
    return decorator


def get_evaluator(name: str, args) -> Evaluator:
    """
    Get an evaluator instance by name.

    Args:
        name: Registered evaluator name
        args: Arguments to pass to evaluator constructor

    Returns:
        Evaluator instance
    """
    if name not in _EVALUATOR_REGISTRY:
        raise ValueError(f"Unknown evaluator: {name}. Available: {list(_EVALUATOR_REGISTRY.keys())}")
    return _EVALUATOR_REGISTRY[name](args)


def list_evaluators() -> List[str]:
    """List all registered evaluator names."""
    return list(_EVALUATOR_REGISTRY.keys())
