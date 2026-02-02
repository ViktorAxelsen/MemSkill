"""
LongMemEval dataset evaluator.

Handles evaluation for LongMemEval dataset with its specific prompt format.
"""
from typing import List, Dict, Any, Tuple

from .base import Evaluator, register_evaluator


@register_evaluator("longmemeval")
class LongMemEvalEvaluator(Evaluator):
    """
    Evaluator for LongMemEval dataset.

    LongMemEval has a simpler structure with single question-answer pairs
    and includes a question_date field for temporal context.
    """

    def prepare_eval_args(self) -> Any:
        """Override default eval args to allow longer generations for HotpotQA."""
        eval_args = super().prepare_eval_args()
        eval_args.max_new_tokens = 512
        return eval_args

    def filter_qa_list(self, qa_list: List[Dict]) -> List[Tuple[int, Dict]]:
        """
        Filter QA list. LongMemEval typically has all valid QA items.

        Args:
            qa_list: List of QA dicts

        Returns:
            List of (index, qa_dict) tuples for valid QA items
        """
        valid_qa = []
        for i, qa in enumerate(qa_list):
            # Check that question and answer exist
            if 'question' in qa and 'answer' in qa:
                valid_qa.append((i, qa))
        return valid_qa

    def build_prompt(self, question: str, retrieved_memories: List[str],
                     qa_item: Dict) -> str:
        """
        Build evaluation prompt for LongMemEval.

        Args:
            question: The question to answer
            retrieved_memories: List of retrieved memory texts
            qa_item: Original QA dict (contains question_date)

        Returns:
            Formatted prompt string
        """
        from prompts.prompt_pool import LONGMEMEVAL_ANSWER_PROMPT

        if len(retrieved_memories) > 0:
            history_string = ""
            for j, mem in enumerate(retrieved_memories):
                history_string += f'\n### Session {j + 1}:\nSession Content:\n{mem}\n'
        else:
            history_string = "No relevant chat history available."

        question_date = qa_item.get('question_date', '')

        return LONGMEMEVAL_ANSWER_PROMPT.format(
            history_string,
            question_date,
            question
        )

    def _get_result_metadata(self, qa: Dict) -> Dict[str, Any]:
        """
        Extract LongMemEval-specific metadata.

        Args:
            qa: QA dict

        Returns:
            Metadata including question_date
        """
        return {
            'question_date': qa.get('question_date', '')
        }
