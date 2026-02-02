"""
ALFWorld offline dataset helpers.

Provides loaders and chunking utilities for expert trajectories.
"""
import json
import random
from typing import Dict, List, Any, Optional, Tuple


class ALFWorldOfflineDataset:
    """Offline ALFWorld dataset loader/sampler for expert trajectories."""
    def __init__(self, data: Dict[str, Dict[str, Dict[str, Any]]]):
        if not isinstance(data, dict):
            raise ValueError("ALFWorld offline data must be a dict keyed by task_type.")
        self.data = data

    @classmethod
    def load(cls, path: str) -> "ALFWorldOfflineDataset":
        with open(path, "r") as f:
            payload = json.load(f)
        return cls(payload)

    def task_types(self) -> List[str]:
        return list(self.data.keys())

    def gamefiles_for_type(self, task_type: str) -> List[str]:
        by_type = self.data.get(task_type, {})
        if isinstance(by_type, dict):
            return list(by_type.keys())
        return []

    def get_entry(self, task_type: str, gamefile: str) -> Optional[Dict[str, Any]]:
        by_type = self.data.get(task_type, {})
        if isinstance(by_type, dict):
            return by_type.get(gamefile)
        return None

    def sample_batch(self, task_type: str, size: int,
                     exclude: Optional[set] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
        if size <= 0:
            return []
        gamefiles = self.gamefiles_for_type(task_type)
        if exclude:
            gamefiles = [g for g in gamefiles if g not in exclude]
        if not gamefiles:
            return []

        if len(gamefiles) >= size:
            selected = random.sample(gamefiles, size)
        else:
            selected = list(gamefiles)
            while len(selected) < size:
                selected.append(random.choice(gamefiles))

        batch = []
        for gamefile in selected:
            entry = self.get_entry(task_type, gamefile)
            if isinstance(entry, dict):
                batch.append((task_type, gamefile, entry))
        return batch

    def sample_pair(self, batch_a_min: int, batch_a_max: int, batch_b_size: int,
                    same_type_prob: float = 0.8) -> Tuple[
                        List[Tuple[str, str, Dict[str, Any]]],
                        List[Tuple[str, str, Dict[str, Any]]]
                    ]:
        task_types = self.task_types()
        if not task_types:
            return [], []

        batch_a_size = max(1, random.randint(batch_a_min, batch_a_max))

        if random.random() < same_type_prob:
            task_type = random.choice(task_types)
            gamefiles = self.gamefiles_for_type(task_type)
            if not gamefiles:
                return [], []

            if len(gamefiles) >= batch_a_size + batch_b_size:
                selected = random.sample(gamefiles, batch_a_size + batch_b_size)
                batch_a_files = selected[:batch_a_size]
                batch_b_files = selected[batch_a_size:]
            else:
                batch_a_files = random.sample(gamefiles, min(batch_a_size, len(gamefiles)))
                remaining = [g for g in gamefiles if g not in batch_a_files]
                if not remaining:
                    remaining = list(gamefiles)
                if len(remaining) >= batch_b_size:
                    batch_b_files = random.sample(remaining, batch_b_size)
                else:
                    batch_b_files = list(remaining)
                    while len(batch_b_files) < batch_b_size and remaining:
                        batch_b_files.append(random.choice(remaining))

            batch_a = [(task_type, g, self.get_entry(task_type, g))
                       for g in batch_a_files if self.get_entry(task_type, g)]
            batch_b = [(task_type, g, self.get_entry(task_type, g))
                       for g in batch_b_files if self.get_entry(task_type, g)]
            return batch_a, batch_b

        task_type_a = random.choice(task_types)
        other_types = [t for t in task_types if t != task_type_a]
        task_type_b = random.choice(other_types) if other_types else task_type_a

        batch_a = self.sample_batch(task_type_a, batch_a_size)
        batch_b = self.sample_batch(task_type_b, batch_b_size)
        return batch_a, batch_b


def chunk_trajectories_by_tokens(trajectories: List[str], chunk_size: Optional[int],
                                 tokenizer_name: str = "cl100k_base",
                                 separator: str = "\n\n=== TRAJECTORY ===\n\n") -> List[str]:
    """Group full trajectories into token-bounded chunks without splitting any trajectory."""
    cleaned = [str(t).strip() for t in trajectories if isinstance(t, str) and t.strip()]
    if not cleaned:
        return []
    if chunk_size is None or chunk_size <= 0:
        return cleaned

    import tiktoken
    tokenizer = tiktoken.get_encoding(tokenizer_name)

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for traj in cleaned:
        traj_tokens = len(tokenizer.encode(traj))

        if not current:
            current.append(traj)
            current_tokens = traj_tokens
            continue

        if current_tokens + traj_tokens <= chunk_size:
            current.append(traj)
            current_tokens += traj_tokens
        else:
            chunks.append(separator.join(current))
            current = [traj]
            current_tokens = traj_tokens

    if current:
        chunks.append(separator.join(current))

    return chunks
