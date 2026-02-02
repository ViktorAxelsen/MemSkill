"""
HotpotQA dataset evaluator.

Handles evaluation for HotpotQA dataset with multi-hop question answering.
Uses F1 score as the primary metric, supporting multiple valid answers.
"""
import re
from typing import List, Dict, Any, Tuple

from .base import Evaluator, EvalResult, register_evaluator


@register_evaluator("hotpotqa")
class HotpotQAEvaluator(Evaluator):
    """
    Evaluator for HotpotQA dataset.

    HotpotQA is a multi-hop QA dataset where questions require reasoning
    across multiple documents. This evaluator:
    - Builds prompts with retrieved context
    - Supports extracting answers from <answer></answer> tags or direct output
    - Computes F1 score against all valid answers (takes max)
    """

    def filter_qa_list(self, qa_list: List[Dict]) -> List[Tuple[int, Dict]]:
        """
        Filter QA list to get valid items.

        For HotpotQA, all items with question and answer are valid.

        Args:
            qa_list: List of QA dicts

        Returns:
            List of (index, qa_dict) tuples for valid QA items
        """
        valid_qa = []
        for i, qa in enumerate(qa_list):
            if qa.get('question') and (qa.get('answer') or qa.get('answers')):
                valid_qa.append((i, qa))
        return valid_qa

    def prepare_eval_args(self) -> Any:
        """Override default eval args to allow longer generations for HotpotQA."""
        eval_args = super().prepare_eval_args()
        eval_args.max_new_tokens = 512
        return eval_args

    def build_prompt(self, question: str, retrieved_memories: List[str],
                     qa_item: Dict) -> str:
        """
        Build evaluation prompt for HotpotQA.

        The prompt includes retrieved context and asks the model to answer
        the question. Supports two output formats:
        1. Direct short answer
        2. Answer wrapped in <answer></answer> tags

        Args:
            question: The question to answer
            retrieved_memories: List of retrieved memory/context texts
            qa_item: Original QA dict

        Returns:
            Formatted prompt string
        """
        from prompts.prompt_pool import HOTPOTQA_ANSWER_PROMPT

        if len(retrieved_memories) > 0:
            # Format context with chunk numbering
            context_parts = []
            for i, mem in enumerate(retrieved_memories, 1):
                context_parts.append(f"[Context {i}]\n{mem}")
            context = "\n\n".join(context_parts)
        else:
            context = (
                "No relevant context available. If the context is insufficient, "
                "answer using your general knowledge and do not refuse."
            )

        return HOTPOTQA_ANSWER_PROMPT.format(context=context, question=question)

    def get_ground_truth(self, qa_item: Dict) -> List[str]:
        """
        Extract ground truth answers.

        Args:
            qa_item: QA dict

        Returns:
            List of all valid ground truth answers
        """
        # Return all valid answers for F1 computation
        answers = qa_item.get('answers', [])
        if not answers:
            # Fallback to 'answer' field if 'answers' is empty
            single_answer = qa_item.get('answer', '')
            return [single_answer] if single_answer else []
        if isinstance(answers, str):
            return [answers]
        return answers

    def compute_f1(self, prediction: str, ground_truth, qa_item: Dict = None) -> float:
        """
        Compute F1 score between prediction and ground truth.

        For HotpotQA, we compute F1 against all valid answers and take the max.
        This handles cases where multiple answers are acceptable.

        Also handles extraction of answer from <answer></answer> tags.

        Args:
            prediction: Model prediction (may contain <answer> tags)
            ground_truth: Ground truth answer(s) - can be str or List[str]
            qa_item: Original QA dict (may contain multiple valid answers)

        Returns:
            F1 score (0-1)
        """
        from eval_utils import f1_score

        # Extract answer from tags if present
        extracted_prediction = self._extract_answer(prediction)

        # Get all valid answers - ground_truth can be str or List[str]
        if isinstance(ground_truth, list):
            valid_answers = ground_truth
        elif isinstance(ground_truth, str) and ground_truth:
            valid_answers = [ground_truth]
        else:
            valid_answers = []

        # Also check qa_item for additional answers
        if qa_item and 'answers' in qa_item and qa_item['answers']:
            qa_answers = qa_item['answers']
            if isinstance(qa_answers, list):
                # Merge with valid_answers, avoiding duplicates
                for ans in qa_answers:
                    if ans and ans not in valid_answers:
                        valid_answers.append(ans)

        # Compute F1 against all valid answers and take max
        max_f1 = 0.0
        for valid_answer in valid_answers:
            if valid_answer:
                f1 = f1_score(extracted_prediction, valid_answer)
                max_f1 = max(max_f1, f1)

        return max_f1

    def _extract_answer(self, response: str) -> str:
        """
        Extract answer from model response.

        Supports two formats:
        1. Answer wrapped in <answer></answer> tags
        2. Direct answer text

        Args:
            response: Model response text

        Returns:
            Extracted answer string
        """
        if not response:
            return ""

        response = response.strip()

        # Try to extract from <answer> tags
        answer_pattern = r'<answer>(.*?)</answer>'
        match = re.search(answer_pattern, response, re.DOTALL | re.IGNORECASE)

        if match:
            return match.group(1).strip()

        # If no tags, try to find "Answer:" or "Short answer:" pattern
        answer_patterns = [
            r'(?:Short )?[Aa]nswer:\s*(.+?)(?:\n|$)',
            r'(?:The answer is|The answer:)\s*(.+?)(?:\n|$)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Return the whole response (possibly truncated if too long)
        # Fallback to the full response
        return response.strip()

    def _get_result_metadata(self, qa: Dict) -> Dict[str, Any]:
        """
        Extract HotpotQA-specific metadata.

        Args:
            qa: QA dict

        Returns:
            Metadata including all valid answers
        """
        return {
            'answers': qa.get('answers', [qa.get('answer', '')]),
        }
