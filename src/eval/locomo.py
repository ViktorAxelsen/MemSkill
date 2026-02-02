"""
LoCoMo dataset evaluator.

Handles evaluation for LoCoMo dataset with its specific QA format and categories.
"""
from typing import List, Dict, Any, Tuple

from .base import Evaluator, EvalResult, register_evaluator


@register_evaluator("locomo")
class LoCoMoEvaluator(Evaluator):
    """
    Evaluator for LoCoMo dataset.

    LoCoMo has 5 question categories:
    - 1: Multi-hop
    - 2: Temporal
    - 3: Open-domain
    - 4: Single-hop
    - 5: Adversarial
    """

    # Categories to skip during evaluation
    SKIP_CATEGORIES = {5}  # Skip adversarial questions

    def prepare_eval_args(self) -> Any:
        """Override default eval args to allow longer generations for HotpotQA."""
        eval_args = super().prepare_eval_args()
        eval_args.max_new_tokens = 32
        return eval_args

    def filter_qa_list(self, qa_list: List[Dict]) -> List[Tuple[int, Dict]]:
        """
        Filter QA list, skipping adversarial questions (category 5).

        Args:
            qa_list: List of QA dicts

        Returns:
            List of (index, qa_dict) tuples for valid QA items
        """
        valid_qa = []
        for i, qa in enumerate(qa_list):
            category = qa.get('category', 1)
            if category not in self.SKIP_CATEGORIES:
                valid_qa.append((i, qa))
        return valid_qa

    def build_prompt(self, question: str, retrieved_memories: List[str],
                     qa_item: Dict) -> str:
        """
        Build evaluation prompt for LoCoMo.

        Args:
            question: The question to answer
            retrieved_memories: List of retrieved memory texts
            qa_item: Original QA dict

        Returns:
            Formatted prompt string
        """
        from prompts.prompt_pool import QA_PROMPT

        if len(retrieved_memories) > 0:
            context = "Below is relevant information from the conversation history:\n\n"
            context += "\n\n".join(retrieved_memories)
        else:
            context = "No relevant information available."

        return context + "\n\n" + QA_PROMPT.format(question)

    def get_ground_truth(self, qa_item: Dict) -> str:
        """
        Extract ground truth answer.

        For open-domain (category 3), use only the first answer (split by ';').

        Args:
            qa_item: QA dict

        Returns:
            Ground truth answer string
        """
        answer = str(qa_item.get('answer', ''))
        category = qa_item.get('category', 1)

        # For open-domain, use only first answer
        if category == 3:
            answer = answer.split(';')[0].strip()

        return answer

    def compute_f1(self, prediction: str, ground_truth: str, qa_item: Dict = None) -> float:
        """
        Compute F1 score based on question category.

        - Category 1 (Multi-hop): Use f1_max (handles comma-separated sub-answers)
        - Category 2, 3, 4: Use standard f1_score

        Args:
            prediction: Model prediction
            ground_truth: Ground truth answer
            qa_item: Original QA dict (needed for category info)

        Returns:
            F1 score (0-1)
        """
        from eval_utils import f1_score, f1_max
        if qa_item is None:
            return f1_score(prediction, ground_truth)

        category = qa_item.get('category', 1)

        # Multi-hop: use f1_max for comma-separated sub-answers
        if category == 1:
            return f1_max(prediction, ground_truth)
        else:
            return f1_score(prediction, ground_truth)

    def _get_result_metadata(self, qa: Dict) -> Dict[str, Any]:
        """
        Extract LoCoMo-specific metadata.

        Args:
            qa: QA dict

        Returns:
            Metadata including category and evidence
        """
        return {
            'category': qa.get('category', 1),
            'evidence': qa.get('evidence', [])
        }

    def compute_category_scores(self, results: List[EvalResult]) -> Dict[int, Dict[str, float]]:
        """
        Compute scores grouped by category.

        Args:
            results: List of EvalResult objects

        Returns:
            Dict mapping category to score statistics
        """
        category_results = {}

        for result in results:
            category = result.metadata.get('category', 1)
            if category not in category_results:
                category_results[category] = {
                    'f1_scores': [],
                    'llm_judge_scores': [],
                    'count': 0
                }

            category_results[category]['f1_scores'].append(result.f1_score)
            category_results[category]['llm_judge_scores'].append(result.llm_judge_score)
            category_results[category]['count'] += 1

        # Compute averages
        category_scores = {}
        for category, data in category_results.items():
            import numpy as np
            category_scores[category] = {
                'avg_f1': np.mean(data['f1_scores']) if data['f1_scores'] else 0.0,
                'avg_llm_judge': np.mean(data['llm_judge_scores']) if data['llm_judge_scores'] else 0.0,
                'count': data['count']
            }

        return category_scores
