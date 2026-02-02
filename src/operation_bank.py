"""
Operation Bank: Stores and evolves memory operations
"""
import numpy as np
from typing import List, Dict, Optional
import copy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts.operation_templates import get_initial_operations


class Operation:
    """Single memory operation"""
    def __init__(self, name: str, description: str,
                 instruction_template: str, update_type: str,
                 meta_info: Optional[Dict] = None):
        self.name = name
        self.description = description
        self.instruction_template = instruction_template
        self.update_type = update_type  # insert, update, delete, noop
        self.meta_info = meta_info or {
            "usage_count": 0,
            "avg_reward": 0.0,
            "recent_rewards": [],
            "recent_usage_ema": 0.0,
            "created_at": "unknown",
            "last_modified": "unknown"
        }
        # Ensure recent_usage_ema exists for backward compatibility
        if "recent_usage_ema" not in self.meta_info:
            self.meta_info["recent_usage_ema"] = 0.0
        self.embedding = None  # Will be set by operation bank

    def get_description_text(self) -> str:
        """Get text description for embedding (no guide characters)"""
        return self.description

    def format_instruction(self, session_text: str, retrieved_memories: str) -> str:
        """Format instruction template with current context"""
        template = self.instruction_template
        if '{session_text}' in template or '{retrieved_memories}' in template:
            try:
                return template.format(
                    session_text=session_text,
                    retrieved_memories=retrieved_memories
                )
            except Exception:
                return template
        return template

    def update_stats(self, reward: float):
        """Update operation statistics when this operation is selected.

        Note: EMA updates are handled separately by OperationBank.update_all_ema()
        which is called at each step during training.

        Args:
            reward: The reward received for this operation
        """
        self.meta_info["usage_count"] += 1

        # Update average reward
        n = self.meta_info["usage_count"]
        old_avg = self.meta_info["avg_reward"]
        new_avg = old_avg + (reward - old_avg) / n
        self.meta_info["avg_reward"] = new_avg

        # Keep recent rewards (last 20)
        self.meta_info["recent_rewards"].append(reward)
        if len(self.meta_info["recent_rewards"]) > 20:
            self.meta_info["recent_rewards"] = self.meta_info["recent_rewards"][-20:]

    def decay_ema(self, ema_alpha: float = 0.1):
        """Decay EMA when this operation is NOT selected

        Should be called for all non-selected operations each step.
        EMA = (1 - alpha) * old_ema (decays toward 0)
        """
        old_ema = self.meta_info.get("recent_usage_ema", 0.0)
        self.meta_info["recent_usage_ema"] = (1.0 - ema_alpha) * old_ema

    def to_dict(self):
        return {
            'name': self.name,
            'description': self.description,
            'instruction_template': self.instruction_template,
            'update_type': self.update_type,
            'meta_info': self.meta_info,
            'embedding': self.embedding.tolist() if self.embedding is not None else None
        }

    @classmethod
    def from_dict(cls, data):
        op = cls(
            name=data['name'],
            description=data['description'],
            instruction_template=data['instruction_template'],
            update_type=data['update_type'],
            meta_info=data.get('meta_info', {})
        )
        if data.get('embedding') is not None:
            op.embedding = np.array(data['embedding'])
        return op


class OperationBank:
    """
    Operation Bank stores memory operations and supports dynamic evolution
    """
    def __init__(self, encoder=None, max_ops: int = 20,
                 skip_noop: bool = False):
        self.operations: Dict[str, Operation] = {}
        self.encoder = encoder  # OpEncoder object (supports state encoder backbones)
        self.max_ops = max_ops  # Maximum number of operations allowed in the bank
        self.new_operation_names = set()
        self.skip_noop = skip_noop
        self._initialize_with_seeds()

    def _initialize_with_seeds(self):
        """Initialize with seed operations"""
        initial_ops = get_initial_operations(include_noop=not self.skip_noop)
        for op_name, op_data in initial_ops.items():
            operation = Operation(
                name=op_data['name'],
                description=op_data['description'],
                instruction_template=op_data['instruction_template'],
                update_type=op_data['update_type'],
                meta_info=op_data['meta_info']
            )
            self.operations[op_name] = operation

        # Compute embeddings for seed operations if encoder is available
        if self.encoder is not None:
            self._recompute_embeddings()

    def set_encoder(self, encoder):
        """Set encoder for operation embeddings"""
        self.encoder = encoder
        self._recompute_embeddings()

    def _recompute_embeddings(self):
        """Recompute embeddings for all operations"""
        if self.encoder is None:
            return

        texts = [op.get_description_text() for op in self.operations.values()]
        if len(texts) == 0:
            return

        # Use the encoder to get embeddings
        embeddings = self.encoder.encode(texts)

        # Assign embeddings to operations
        for i, op_name in enumerate(self.operations.keys()):
            self.operations[op_name].embedding = embeddings[i]

    def load_from_templates(self, templates: Dict[str, Dict]):
        """Replace operation bank using a template dict (sanity check helper)."""
        self.operations = {}
        self.new_operation_names = set()

        for op_name, op_data in templates.items():
            operation = Operation(
                name=op_data['name'],
                description=op_data['description'],
                instruction_template=op_data['instruction_template'],
                update_type=op_data['update_type'],
                meta_info=copy.deepcopy(op_data.get('meta_info', {}))
            )
            self.operations[op_name] = operation

        if self.encoder is not None:
            self._recompute_embeddings()

    def set_new_operation_names(self, names: List[str]):
        """Set which operation names are treated as new for exploration bias."""
        self.new_operation_names = set(names)

    def get_new_action_indices(self, candidate_ops: Optional[List[Operation]] = None) -> List[int]:
        """Get indices of new actions in the candidate list."""
        if candidate_ops is None:
            candidate_ops = self.get_candidate_operations()
        return [i for i, op in enumerate(candidate_ops) if op.name in self.new_operation_names]

    def get_candidate_operations(self) -> List[Operation]:
        """
        Get all operations in the bank for controller to select from.

        Returns:
            List of all Operation objects
        """
        if len(self.operations) == 0:
            return []

        # Return all operations sorted by name for stable ordering.
        # This ensures action_idx mapping is deterministic across calls.
        # Note: Even though we store op_embeddings per step (making PPO self-consistent),
        # stable ordering is still good practice to avoid subtle bugs when Designer
        # adds/removes operations between episodes.
        names = sorted(self.operations.keys())
        return [self.operations[name] for name in names]

    def add_operation(self, operation: Operation) -> bool:
        """
        Add a new operation to the bank.

        If the bank is at capacity (max_ops), the operation with the lowest
        avg_reward will be replaced.

        Args:
            operation: The operation to add

        Returns:
            True if operation was added successfully
        """
        # If operation already exists, just update it
        if operation.name in self.operations:
            self.operations[operation.name] = operation
            if self.encoder is not None:
                self._recompute_embeddings()
            return True

        # Check if at capacity
        if len(self.operations) >= self.max_ops:
            # Find the operation with lowest avg_reward (that has been used at least once)
            worst_op_name = None
            worst_reward = float('inf')

            for name, op in self.operations.items():
                # Only consider operations that have been used
                if op.meta_info.get('usage_count', 0) > 0:
                    avg_reward = op.meta_info.get('avg_reward', 0.0)
                    if avg_reward < worst_reward:
                        worst_reward = avg_reward
                        worst_op_name = name

            # If no used operations found, pick the one with lowest usage count
            if worst_op_name is None:
                min_usage = float('inf')
                for name, op in self.operations.items():
                    usage = op.meta_info.get('usage_count', 0)
                    if usage < min_usage:
                        min_usage = usage
                        worst_op_name = name

            # Replace worst operation
            if worst_op_name is not None:
                del self.operations[worst_op_name]
            else:
                return False

        self.operations[operation.name] = operation
        self.new_operation_names.add(operation.name)
        # Recompute embeddings if encoder is available
        if self.encoder is not None:
            self._recompute_embeddings()
        return True

    def update_operation(self, name: str, **kwargs):
        """Update an existing operation"""
        if name not in self.operations:
            raise KeyError(f"Operation {name} not found")

        op = self.operations[name]
        for key, value in kwargs.items():
            if hasattr(op, key):
                setattr(op, key, value)

        # Recompute embeddings if encoder is available
        if self.encoder is not None:
            self._recompute_embeddings()
        self.new_operation_names.add(name)

    def remove_operation(self, name: str):
        """Remove an operation from the bank"""
        if name in self.operations:
            del self.operations[name]
            self.new_operation_names.discard(name)

    def get_operation(self, name: str) -> Operation:
        """Get operation by name"""
        if name not in self.operations:
            raise KeyError(f"Operation {name} not found")
        return self.operations[name]

    def get_all_operations(self) -> List[Operation]:
        """Get all operations"""
        return list(self.operations.values())

    def get_operation_stats(self) -> Dict:
        """Get statistics for all operations"""
        stats = {}
        for name, op in self.operations.items():
            stats[name] = {
                'usage_count': op.meta_info['usage_count'],
                'avg_reward': op.meta_info['avg_reward'],
                'recent_rewards': op.meta_info['recent_rewards'],
                'recent_usage_ema': op.meta_info.get('recent_usage_ema', 0.0)
            }
        return stats

    def update_all_ema(self, selected_op_name: str, ema_alpha: float = 0.1):
        """Update EMA for all operations after a selection.

        - Selected operation: EMA spikes toward 1.0
        - Non-selected operations: EMA decays toward 0.0

        Args:
            selected_op_name: Name of the operation that was selected
            ema_alpha: EMA smoothing factor
        """
        for name, op in self.operations.items():
            if name == selected_op_name:
                # Spike: EMA = alpha * 1.0 + (1 - alpha) * old_ema
                old_ema = op.meta_info.get("recent_usage_ema", 0.0)
                op.meta_info["recent_usage_ema"] = ema_alpha * 1.0 + (1.0 - ema_alpha) * old_ema
            else:
                op.decay_ema(ema_alpha)

    def batch_update_ema(self, op_usage_counts: Dict[str, int], total_steps: int, ema_alpha: float = 0.1):
        """Update EMA based on batch-level usage counts.

        This is used for parallel episode collection where we aggregate usage
        across all episodes and update EMA once at the end of the batch.

        The update simulates `total_steps` EMA updates where each op's selection
        frequency determines how often it spikes vs decays.

        Args:
            op_usage_counts: Dict mapping op_name -> number of times selected in batch
            total_steps: Total number of steps across all episodes in the batch
            ema_alpha: EMA smoothing factor
        """
        if total_steps == 0:
            return

        for name, op in self.operations.items():
            old_ema = op.meta_info.get("recent_usage_ema", 0.0)
            count = op_usage_counts.get(name, 0)

            # Approximate EMA after `total_steps` updates:
            # - Op was selected `count` times (spike to 1.0)
            # - Op was not selected `total_steps - count` times (decay)
            #
            # The closed-form approximation for repeated EMA updates:
            # After n steps with selection frequency f = count/total_steps:
            # EMA converges toward f, with decay rate (1-alpha)^n toward old value
            #
            # new_ema = target * (1 - decay_factor) + old_ema * decay_factor
            # where target = selection_frequency, decay_factor = (1-alpha)^total_steps

            selection_freq = count / total_steps
            decay_factor = (1.0 - ema_alpha) ** total_steps

            # The EMA update formula for batch:
            # new_ema blends toward selection_freq, with old_ema decayed
            new_ema = selection_freq * (1.0 - decay_factor) + old_ema * decay_factor

            op.meta_info["recent_usage_ema"] = new_ema

    def __len__(self):
        return len(self.operations)

    def to_dict(self):
        """Serialize to dict"""
        return {
            'operations': {name: op.to_dict() for name, op in self.operations.items()},
            'max_ops': self.max_ops,
            'skip_noop': self.skip_noop
        }

    @classmethod
    def from_dict(cls, data, encoder=None):
        """Deserialize from dict"""
        # Create bank without initializing seeds (we'll load operations from data)
        bank = cls.__new__(cls)
        bank.operations = {}
        bank.encoder = encoder
        bank.max_ops = data.get('max_ops', 20)
        bank.skip_noop = data.get('skip_noop', False)
        bank.new_operation_names = set()

        for name, op_data in data.get('operations', {}).items():
            bank.operations[name] = Operation.from_dict(op_data)

        # Recompute embeddings with current encoder for consistency
        if bank.encoder is not None:
            bank._recompute_embeddings()

        return bank

    def copy(self):
        """Create a deep copy of the operation bank"""
        return copy.deepcopy(self)
