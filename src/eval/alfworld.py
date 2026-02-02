"""
ALFWorld dataset evaluator.

ALFWorld is an interactive environment dataset where success is usually
measured by task completion rather than QA.
"""
from typing import List, Dict, Tuple

from .base import Evaluator, register_evaluator


@register_evaluator("alfworld")
class ALFWorldEvaluator(Evaluator):
    """Minimal evaluator for ALFWorld-style datasets."""
    def filter_qa_list(self, qa_list: List[Dict]) -> List[Tuple[int, Dict]]:
        return []

    def build_prompt(self, question: str, retrieved_memories: List[str],
                     qa_item: Dict) -> str:
        return ""

    def get_episode_reward(self, data: Dict) -> float:
        if isinstance(data, dict):
            for key in ("_alfworld_episode_reward", "episode_reward"):
                value = data.get(key)
                if isinstance(value, (int, float)):
                    return float(value)

        for key in ("reward", "score", "episode_reward"):
            value = data.get(key)
            if isinstance(value, (int, float)):
                return float(value)

        for key in ("success", "task_success", "completed"):
            value = data.get(key)
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            if isinstance(value, (int, float)):
                return float(value)

        rewards = data.get("rewards")
        if isinstance(rewards, list) and rewards:
            try:
                return float(sum(float(r) for r in rewards))
            except (TypeError, ValueError):
                return 0.0

        return 0.0
