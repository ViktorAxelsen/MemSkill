"""
Controller: Trainable agent that selects operations
Uses PPO (Proximal Policy Optimization) for training with dynamic action space
- Dual-encoder architecture: state_net + op_net with interaction layer
- Actor-Critic: policy head + value head
- On-policy learning with GAE (Generalized Advantage Estimation)
- Clipped surrogate objective for stable updates
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


class PPOBuffer:
    """
    Buffer for PPO training
    Stores episode trajectories with states, actions, log_probs, values, rewards

    Supports both single action (K=1) and top-K action selection:
    - actions: List of action indices (single int for K=1, List[int] for K>1)
    - log_probs: Joint log probability of selected action(s)
    """
    def __init__(self):
        self.states = []
        self.op_embs = []
        self.new_op_masks = []
        self.actions = []  # List[int] or List[List[int]] for top-K
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def push(self, state_emb: np.ndarray, op_embs: np.ndarray,
             action: Union[int, List[int]], log_prob: float, value: float, reward: float = 0.0,
             new_op_mask: Optional[np.ndarray] = None):
        """
        Store a transition with optional immediate reward (e.g., exploration bonus)

        Args:
            action: Single action index (int) for K=1, or list of action indices for top-K
            log_prob: Joint log probability of selected action(s)
        """
        self.states.append(state_emb)
        self.op_embs.append(op_embs)
        if new_op_mask is None:
            new_op_mask = np.zeros(op_embs.shape[0], dtype=np.float32)
        self.new_op_masks.append(new_op_mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)  # Can include shaping rewards like exploration bonus
        self.dones.append(False)

    def merge(self, other: 'PPOBuffer'):
        """Merge another buffer into this one (for parallel episode collection)"""
        self.states.extend(other.states)
        self.op_embs.extend(other.op_embs)
        self.new_op_masks.extend(other.new_op_masks)
        self.actions.extend(other.actions)
        self.log_probs.extend(other.log_probs)
        self.values.extend(other.values)
        self.rewards.extend(other.rewards)
        self.dones.extend(other.dones)

    def finish_episode(self, final_reward: float, redistribute: bool = True,
                       redistribution_decay: float = 0.9,
                       final_reward_last_ratio: float = 0.6):
        """
        Mark episode as finished and handle final reward.

        Args:
            final_reward: The delayed reward (e.g., QA performance)
            redistribute: If True, spread final reward across all steps in the episode
                         using exponential decay. This helps with credit assignment
                         in long horizon settings.
            redistribution_decay: Decay factor for reward redistribution (0-1).
                                 Higher values give more credit to later steps.
            final_reward_last_ratio: Portion of final_reward added directly to the last
                                     step when redistribute is True. Remainder is
                                     redistributed across all steps.
        """
        if len(self.rewards) == 0:
            return

        # Find the start of the current episode (after last done=True or beginning)
        episode_start = 0
        for i in range(len(self.dones) - 1, -1, -1):
            if self.dones[i]:
                episode_start = i + 1
                break

        episode_length = len(self.rewards) - episode_start

        if redistribute and episode_length > 1:
            final_reward_last_ratio = max(0.0, min(final_reward_last_ratio, 1.0))
            last_reward = final_reward * final_reward_last_ratio
            remaining_reward = final_reward - last_reward

            if remaining_reward != 0.0:
                # Redistribute remaining reward across all steps with exponential decay
                # Later steps get more credit than earlier steps
                weights = np.array([redistribution_decay ** (episode_length - 1 - i)
                                   for i in range(episode_length)])
                weights = weights / weights.sum()  # Normalize to sum to 1

                for i, w in enumerate(weights):
                    self.rewards[episode_start + i] += remaining_reward * w

            self.rewards[-1] += last_reward
        else:
            # Original behavior: add all final reward to last step
            self.rewards[-1] += final_reward

        self.dones[-1] = True

    def compute_returns_and_advantages(self, gamma: float = 0.99,
                                        gae_lambda: float = 0.95,
                                        last_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and GAE advantages
        Correctly handles multiple episodes in the buffer by resetting GAE at episode boundaries.

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            last_value: Value estimate for the state after last step (0 if episode ended)
        Returns:
            returns: [N] array of discounted returns
            advantages: [N] array of GAE advantages
        """
        n = len(self.rewards)
        if n == 0:
            return np.array([]), np.array([])

        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones, dtype=np.float32)

        # Compute GAE with proper episode boundary handling
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            # Non-terminal mask: 0 if done (episode ended), 1 otherwise
            next_non_terminal = 1.0 - dones[t]

            # TD error
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]

            # GAE: reset last_gae when episode ends (next_non_terminal = 0)
            # This prevents advantage from bleeding across episode boundaries
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        # Returns = advantages + values
        returns = advantages + values

        return returns, advantages

    def get_batch(self) -> Dict:
        """
        Get all data as a batch dict

        For top-K actions:
        - If actions are lists (K>1), returns 'actions' as 2D array [N, K]
        - If actions are ints (K=1), returns 'actions' as 1D array [N] for backward compatibility
        """
        # Determine if we have top-K actions (list) or single actions (int)
        if len(self.actions) > 0 and isinstance(self.actions[0], (list, np.ndarray)):
            # Top-K case: convert to 2D array [N, K]
            actions_array = np.array(self.actions)
        else:
            # Single action case: 1D array [N]
            actions_array = np.array(self.actions)

        return {
            'states': self.states,
            'op_embs': self.op_embs,
            'new_op_masks': self.new_op_masks,
            'actions': actions_array,
            'log_probs': np.array(self.log_probs),
            'values': np.array(self.values),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones)
        }

    def clear(self):
        """Clear the buffer"""
        self.states = []
        self.op_embs = []
        self.new_op_masks = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        return len(self.states)


class PPOController(nn.Module):
    """
    PPO Controller with Actor-Critic architecture for operation selection
    - Dual-encoder: state_net + op_net for handling dynamic action space
    - Actor (Policy): outputs action logits for each operation
    - Critic (Value): outputs state value V(s)
    - Uses clipped surrogate objective for stable policy updates
    - Supports top-K action selection (action_top_k parameter)
    """
    def __init__(self, state_dim: int = 768, op_dim: int = 768,
                 hidden_dim: int = 256, device: str = 'cuda',
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, vf_clip: float = 0.0,
                 new_action_p_min: float = 0.0, new_action_delta_max: float = 0.0,
                 action_top_k: int = 1):
        super(PPOController, self).__init__()

        self.state_dim = state_dim
        self.op_dim = op_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.vf_clip = vf_clip
        self.new_action_p_min = new_action_p_min
        self.new_action_delta_max = new_action_delta_max
        self.new_action_bias_scale = 0.0
        self.action_top_k = action_top_k  # Number of top actions to select per step

        # State encoder network (shared backbone)
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Operation encoder network
        self.op_net = nn.Sequential(
            nn.Linear(op_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head: computes action logits from [state_h, op_h] pairs
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output score for each (state, op) pair
        )

        # Critic head: computes state value from state_h only
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.to(device)

    def encode_state(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """Encode state"""
        return self.state_net(state_embedding)

    def encode_ops(self, op_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode operations"""
        return self.op_net(op_embeddings)

    def get_action_logits(self, state_h: torch.Tensor, op_h: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute action logits for each operation
        state_h: [B, hidden_dim]
        op_h: [B, num_ops, hidden_dim]
        mask: [B, num_ops] - 1 for valid ops, 0 for padding
        Returns: [B, num_ops] logits
        """
        batch_size = state_h.shape[0]
        num_ops = op_h.shape[1]

        # Expand state to match ops
        state_expanded = state_h.unsqueeze(1).expand(-1, num_ops, -1)  # [B, num_ops, hidden_dim]

        # Concatenate
        combined = torch.cat([state_expanded, op_h], dim=-1)  # [B, num_ops, hidden_dim*2]

        # Compute logits
        logits = self.actor_head(combined).squeeze(-1)  # [B, num_ops]

        # Apply mask (set padded positions to large negative value)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))

        return logits

    def get_value(self, state_h: torch.Tensor) -> torch.Tensor:
        """
        Compute state value
        state_h: [B, hidden_dim]
        Returns: [B] values
        """
        return self.critic_head(state_h).squeeze(-1)

    def set_new_action_bias_scale(self, bias_scale: float):
        """Set current bias scale for new-action exploration."""
        self.new_action_bias_scale = float(bias_scale)

    def _apply_new_action_bias(self, logits: torch.Tensor,
                               new_op_mask: torch.Tensor) -> torch.Tensor:
        if new_op_mask is None:
            return logits
        if self.new_action_p_min <= 0.0 or self.new_action_delta_max <= 0.0:
            return logits
        if self.new_action_bias_scale <= 0.0:
            return logits

        if logits.dim() == 1:
            logits_in = logits.unsqueeze(0)
            mask_in = new_op_mask.unsqueeze(0)
        else:
            logits_in = logits
            mask_in = new_op_mask

        mask_in = mask_in.float()

        mask_sum = mask_in.sum(dim=-1)
        has_new = mask_sum > 0
        if not torch.any(has_new):
            return logits

        with torch.no_grad():
            probs = torch.softmax(logits_in, dim=-1)
            p_new = (probs * mask_in).sum(dim=-1)
            target = float(self.new_action_p_min)
            need_bias = has_new & (p_new < target)
            if not torch.any(need_bias):
                return logits
            eps = 1e-8
            safe_p = torch.where(need_bias, p_new, torch.ones_like(p_new))
            delta = torch.log(target / (safe_p + eps))
            delta = torch.where(need_bias, delta, torch.zeros_like(delta))
            delta = torch.clamp(delta, min=0.0, max=float(self.new_action_delta_max))
            delta = delta * float(self.new_action_bias_scale)

        logits_out = logits_in + delta.unsqueeze(-1) * mask_in
        if logits.dim() == 1:
            return logits_out[0]
        return logits_out

    def _compute_topk_log_prob(self, logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute joint log probability for top-K action selection (without replacement).

        For Gumbel-top-k sampling (sampling K items without replacement), the joint
        probability is:
            P(a1, a2, ..., aK) = p(a1) × p(a2)/(1-p(a1)) × p(a3)/(1-p(a1)-p(a2)) × ...

        In log form:
            log P = Σ_i [log p(ai) - log(1 - Σ_{j<i} p(aj))]

        This correctly matches the Gumbel-top-k sampling distribution used in forward().

        Args:
            logits: [B, num_ops] or [num_ops] action logits
            actions: [B, K] or [K] indices of selected actions

        Returns:
            [B] or scalar joint log probabilities
        """
        single_batch = logits.dim() == 1
        if single_batch:
            logits = logits.unsqueeze(0)  # [1, num_ops]
            actions = actions.unsqueeze(0)  # [1, K]

        batch_size = logits.shape[0]
        k = actions.shape[1]

        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=-1)  # [B, num_ops]

        # Gather probs for selected actions in order
        selected_probs = torch.gather(probs, dim=-1, index=actions)  # [B, K]

        # Compute log probability for without-replacement sampling
        # log P = Σ_i [log p(ai) - log(1 - Σ_{j<i} p(aj))]
        joint_log_prob = torch.zeros(batch_size, device=logits.device)
        remaining_prob = torch.ones(batch_size, device=logits.device)  # 1 - cumsum of selected probs
        eps = 1e-8

        for i in range(k):
            p_i = torch.clamp(selected_probs[:, i], min=eps)
            # log P(ai | a1,...,ai-1 selected) = log(p_i / remaining_prob)
            # = log(p_i) - log(remaining_prob)
            denom = torch.clamp(remaining_prob, min=eps)
            joint_log_prob = joint_log_prob + torch.log(p_i) - torch.log(denom)
            remaining_prob = torch.clamp(remaining_prob - p_i, min=eps)

        if single_batch:
            return joint_log_prob[0]  # scalar
        return joint_log_prob

    def _compute_topk_stats(self, logits: torch.Tensor,
                            actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute top-K diagnostics from the action probabilities.

        Returns:
            topk_entropy: [B] entropy of normalized top-K probabilities
            topk_mass: [B] total probability mass assigned to top-K actions
            topk_bin_entropy: [B] binary entropy between top-K and tail mass
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)

        probs = torch.softmax(logits, dim=-1)  # [B, num_ops]
        selected_probs = torch.gather(probs, dim=-1, index=actions)  # [B, K]

        mass_topk = selected_probs.sum(dim=-1)  # [B]
        eps = 1e-8
        safe_mass = mass_topk.clamp(min=eps)
        normalized = (selected_probs / safe_mass.unsqueeze(-1)).clamp(min=eps)
        topk_entropy = -(normalized * torch.log(normalized)).sum(dim=-1)

        m = mass_topk.clamp(min=eps, max=1 - eps)
        topk_bin_entropy = -(m * torch.log(m) + (1 - m) * torch.log(1 - m))

        return topk_entropy, mass_topk, topk_bin_entropy

    def forward(self, state_embedding: torch.Tensor,
                op_embeddings: torch.Tensor,
                deterministic: bool = False,
                new_op_mask: Optional[torch.Tensor] = None
                ) -> Tuple[Union[int, List[int]], float, float]:
        """
        Forward pass with action selection (supports top-K)

        Args:
            state_embedding: [state_dim]
            op_embeddings: [num_ops, op_dim]
            deterministic: If True, always select top actions by logits
            new_op_mask: Optional mask for new action bias

        Returns:
            - action: int (if K=1) or List[int] (if K>1) selected action indices
            - log_prob: float, joint log probability of selected action(s)
            - value: float, state value estimate
        """
        # Add batch dimension
        state_emb = state_embedding.unsqueeze(0)  # [1, state_dim]
        op_embs = op_embeddings.unsqueeze(0)  # [1, num_ops, op_dim]

        # Encode
        state_h = self.encode_state(state_emb)  # [1, hidden_dim]
        op_h = self.encode_ops(op_embs)  # [1, num_ops, hidden_dim]

        # Get action logits and value
        logits = self.get_action_logits(state_h, op_h)  # [1, num_ops]
        value = self.get_value(state_h)  # [1]

        logits = logits[0]  # [num_ops]
        value = value[0]  # scalar
        if new_op_mask is not None:
            if not torch.is_tensor(new_op_mask):
                new_op_mask = torch.tensor(new_op_mask, dtype=torch.float32, device=logits.device)
            else:
                new_op_mask = new_op_mask.to(logits.device)
            logits = self._apply_new_action_bias(logits, new_op_mask)

        num_ops = logits.shape[0]
        k = min(self.action_top_k, num_ops)  # Can't select more than available ops

        if k == 1:
            # Original single-action behavior
            dist = torch.distributions.Categorical(logits=logits)

            if deterministic:
                action = torch.argmax(logits).item()
            else:
                action = dist.sample().item()

            log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
            return action, log_prob, value.item()
        else:
            # Top-K action selection
            if deterministic:
                # Select top-K by logits
                _, top_k_indices = torch.topk(logits, k)
                actions = top_k_indices.tolist()
            else:
                # Sample K actions without replacement using Gumbel-top-k trick
                # This provides proper sampling from the categorical distribution
                gumbel = torch.distributions.Gumbel(0, 1).sample(logits.shape).to(logits.device)
                perturbed = logits + gumbel
                _, top_k_indices = torch.topk(perturbed, k)
                actions = top_k_indices.tolist()

            # Compute joint log probability
            actions_tensor = torch.tensor(actions, device=self.device)
            log_prob = self._compute_topk_log_prob(logits, actions_tensor).item()

            return actions, log_prob, value.item()

    def evaluate_actions(self, state_embs: torch.Tensor, op_embs: torch.Tensor,
                         actions: torch.Tensor, op_masks: Optional[torch.Tensor] = None,
                         new_op_masks: Optional[torch.Tensor] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Evaluate actions for PPO update (supports top-K actions)

        Args:
            state_embs: [B, state_dim]
            op_embs: [B, max_ops, op_dim]
            actions: [B] for single action or [B, K] for top-K actions
            op_masks: [B, max_ops] - 1 for valid, 0 for padding

        Returns:
            log_probs: [B] log probabilities of taken action(s)
            values: [B] state values
            entropy: [B] policy entropy
            topk_stats: dict with top-K diagnostics (entropy/mass/bin-entropy)
        """
        # Encode
        state_h = self.encode_state(state_embs)  # [B, hidden_dim]
        op_h = self.encode_ops(op_embs)  # [B, max_ops, hidden_dim]

        # Get logits and values
        logits = self.get_action_logits(state_h, op_h, mask=op_masks)  # [B, max_ops]
        values = self.get_value(state_h)  # [B]
        if new_op_masks is not None:
            logits = self._apply_new_action_bias(logits, new_op_masks)

        # Compute log probs and entropy
        dist = torch.distributions.Categorical(logits=logits)
        entropy = dist.entropy()

        # Handle single action [B] vs top-K actions [B, K]
        if actions.dim() == 1:
            # Single action case (K=1)
            log_probs = dist.log_prob(actions)
        else:
            # Top-K case: compute joint log probability
            log_probs = self._compute_topk_log_prob(logits, actions)

        with torch.no_grad():
            topk_entropy, topk_mass, topk_bin_entropy = self._compute_topk_stats(
                logits.detach(), actions
            )
        topk_stats = {
            'topk_entropy': topk_entropy,
            'topk_mass': topk_mass,
            'topk_bin_entropy': topk_bin_entropy,
        }

        return log_probs, values, entropy, topk_stats

    def compute_ppo_loss(self, batch: Dict, returns: np.ndarray,
                         advantages: np.ndarray) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PPO loss with value function clipping
        Args:
            batch: dict with states, op_embs, actions, log_probs, values (old)
            returns: [N] array of returns
            advantages: [N] array of advantages
        Returns:
            total_loss: combined loss
            loss_info: dict with individual loss components
        """
        n = len(batch['states'])

        # Handle variable number of operations (dynamic action space)
        op_dim = batch['op_embs'][0].shape[1]
        max_ops = max(op.shape[0] for op in batch['op_embs'])

        # Pad op_embs and create masks
        op_embs_padded = np.zeros((n, max_ops, op_dim), dtype=np.float32)
        op_masks = np.zeros((n, max_ops), dtype=np.float32)
        new_op_masks_padded = None
        if 'new_op_masks' in batch:
            new_op_masks_padded = np.zeros((n, max_ops), dtype=np.float32)

        for i, op in enumerate(batch['op_embs']):
            n_ops = op.shape[0]
            op_embs_padded[i, :n_ops] = op
            op_masks[i, :n_ops] = 1.0
            if new_op_masks_padded is not None:
                new_mask = batch['new_op_masks'][i]
                if new_mask is not None:
                    new_op_masks_padded[i, :n_ops] = new_mask

        # Convert to tensors
        state_embs = torch.FloatTensor(np.array(batch['states'])).to(self.device)
        op_embs = torch.FloatTensor(op_embs_padded).to(self.device)
        op_masks = torch.FloatTensor(op_masks).to(self.device)
        new_op_masks = None
        if new_op_masks_padded is not None:
            new_op_masks = torch.FloatTensor(new_op_masks_padded).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)
        old_values = torch.FloatTensor(batch['values']).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)

        # NOTE: Advantages should already be normalized at the full-batch level
        # in trainer.py before minibatch splitting. Do NOT normalize again here,
        # as per-minibatch normalization introduces inconsistent scales.

        # Evaluate actions
        new_log_probs, values, entropy, topk_stats = self.evaluate_actions(
            state_embs, op_embs, actions, op_masks, new_op_masks
        )

        # Policy loss (clipped surrogate objective)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages_t
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_t
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss (optional clipping)
        if self.vf_clip is not None and self.vf_clip > 0:
            value_pred_clipped = old_values + torch.clamp(
                values - old_values, -self.vf_clip, self.vf_clip
            )
            value_loss_unclipped = (values - returns_t) ** 2
            value_loss_clipped = (value_pred_clipped - returns_t) ** 2
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
        else:
            value_loss = 0.5 * ((values - returns_t) ** 2).mean()

        # Entropy bonus (encourage exploration)
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        # Compute explained variance for value function quality
        with torch.no_grad():
            explained_var = 1 - (returns_t - values).var() / (returns_t.var() + 1e-8)

        loss_info = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
            'topk_entropy': topk_stats['topk_entropy'].mean().item(),
            'topk_mass': topk_stats['topk_mass'].mean().item(),
            'topk_bin_entropy': topk_stats['topk_bin_entropy'].mean().item(),
            'approx_kl': ((old_log_probs - new_log_probs).mean()).item(),
            'clip_frac': ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item(),
            'explained_variance': explained_var.item(),
            'value_mean': values.mean().item(),
            'return_mean': returns_t.mean().item(),
            'advantage_mean': advantages_t.mean().item(),
        }

        return total_loss, loss_info


# Backward compatibility alias
Controller = PPOController


class BaseTextEncoder:
    """
    Base text encoder that holds the shared state encoder backbone.
    This class can be shared between StateEncoder and OpEncoder to avoid
    loading multiple copies of the same model.
    """
    def __init__(self, model_name: str = "allenai/longformer-base-4096", device: str = 'cuda',
                 encode_batch_size: int = 64, use_flash_attn: bool = True):
        self.model_name = model_name
        self.device = device
        self.encode_batch_size = encode_batch_size
        self.use_flash_attn = bool(use_flash_attn)
        model_name_lower = model_name.lower()
        self._use_qwen_embedding = "qwen3-embedding" in model_name_lower
        self._use_sentence_transformer = (
            model_name.startswith("sentence-transformers/")
            or self._use_qwen_embedding
        )

        if self._use_sentence_transformer:
            if self._use_qwen_embedding:
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float16
                model_kwargs = {"torch_dtype": torch_dtype}
                if self.use_flash_attn:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                tokenizer_kwargs = {"padding_side": "left"}
                if str(device).startswith("cuda"):
                    self.model = SentenceTransformer(
                        model_name,
                        device=device,
                        model_kwargs=model_kwargs,
                        tokenizer_kwargs=tokenizer_kwargs,
                    )
                else:
                    model_kwargs["device_map"] = "auto"
                    self.model = SentenceTransformer(
                        model_name,
                        device=device,
                        model_kwargs=model_kwargs,
                        tokenizer_kwargs=tokenizer_kwargs,
                    )
            else:
                self.model = SentenceTransformer(model_name, device=device)
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            self._embedding_dim = self.model.config.hidden_size

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling to get sentence embedding from token embeddings"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: Union[str, List[str]], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Encode text(s) using the underlying model with batch processing.

        Args:
            texts: Single text or list of texts to encode
            batch_size: Override default batch size (default: self.encode_batch_size)

        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        if len(texts) == 0:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)

        if batch_size is None:
            batch_size = self.encode_batch_size

        if self._use_sentence_transformer:
            return self.model.encode(texts, convert_to_numpy=True, batch_size=batch_size)
        else:
            # Process in batches to avoid OOM
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                with torch.no_grad():
                    encoded = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=4096,
                        return_tensors='pt'
                    )
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                    outputs = self.model(**encoded)
                    embeddings = self._mean_pooling(outputs, encoded['attention_mask'])
                    all_embeddings.append(embeddings.cpu().numpy())
            return np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]


class StateEncoder:
    """
    Encodes session + retrieved memories into state embedding.
    Supports HuggingFace encoder backbones and SentenceTransformer models.

    Can optionally use a shared BaseTextEncoder to avoid loading multiple model copies.
    """
    def __init__(self, model_name: str = "allenai/longformer-base-4096", device: str = 'cuda',
                 fusion_mode: str = "mean", fusion_tau: float = 1.0,
                 base_encoder: Optional['BaseTextEncoder'] = None,
                 encode_batch_size: int = 64, use_flash_attn: bool = True):
        self.fusion_mode = fusion_mode
        self.fusion_tau = fusion_tau

        if base_encoder is not None:
            # Use shared encoder
            self._base_encoder = base_encoder
            self.model_name = base_encoder.model_name
            self.device = base_encoder.device
        else:
            # Create own encoder
            self._base_encoder = BaseTextEncoder(
                model_name=model_name,
                device=device,
                encode_batch_size=encode_batch_size,
                use_flash_attn=use_flash_attn
            )
            self.model_name = model_name
            self.device = device

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension of the encoder"""
        return self._base_encoder.embedding_dim

    def _encode_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) using the underlying model"""
        return self._base_encoder.encode(texts)

    def _fuse_memory_embeddings(self, session_emb: np.ndarray, memory_embs: np.ndarray,
                                fusion_mode: str, fusion_tau: float) -> np.ndarray:
        mem_arr = np.asarray(memory_embs)
        if mem_arr.ndim == 1:
            mem_arr = mem_arr.reshape(1, -1)

        if fusion_mode == "mean":
            return np.mean(mem_arr, axis=0)
        if fusion_mode == "sim_weighted":
            sess = session_emb.astype(np.float32)
            mem = mem_arr.astype(np.float32)
            sess_norm = sess / (np.linalg.norm(sess) + 1e-8)
            mem_norms = np.linalg.norm(mem, axis=1, keepdims=True)
            mem_norm = mem / (mem_norms + 1e-8)
            sims = np.dot(mem_norm, sess_norm)
            tau = max(float(fusion_tau), 1e-8)
            sims = sims / tau
            sims = sims - np.max(sims)
            weights = np.exp(sims).astype(np.float32)
            weights = weights / (weights.sum() + 1e-8)
            return np.sum(mem * weights[:, None], axis=0)

        raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

    def encode(self, session_text: str, retrieved_memories: List[str],
               session_embedding: Optional[np.ndarray] = None,
               memory_embeddings: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
               fusion_mode: Optional[str] = None,
               fusion_tau: Optional[float] = None) -> np.ndarray:
        """
        Encode state from session and retrieved memories.
        Strategy: Encode session and memories separately, then concatenate.
        If precomputed embeddings are provided, reuse them to skip redundant encoding.
        fusion_mode controls how memory embeddings are combined.

        Returns: state embedding vector [session_emb || memory_emb]
        """
        if fusion_mode is None:
            fusion_mode = self.fusion_mode
        if fusion_tau is None:
            fusion_tau = self.fusion_tau

        # Encode current session if not provided
        if session_embedding is None:
            session_emb = self._encode_texts(session_text)
            if session_emb.ndim == 2:
                session_emb = session_emb[0]
        else:
            session_emb = session_embedding
            if session_emb.ndim == 2:
                session_emb = session_emb[0]

        memory_emb = None
        mem_arr = None
        if memory_embeddings is not None:
            if isinstance(memory_embeddings, list):
                if len(memory_embeddings) == 0:
                    memory_embeddings = None
                else:
                    memory_embeddings = np.vstack(memory_embeddings)

            if memory_embeddings is not None:
                mem_arr = np.asarray(memory_embeddings)
                if mem_arr.size == 0:
                    mem_arr = None

        if mem_arr is None:
            if len(retrieved_memories) == 0:
                # No memories retrieved, use zero vector for memory part
                embedding_dim = session_emb.shape[0]
                memory_emb = np.zeros(embedding_dim, dtype=np.float32)
            else:
                # Encode each retrieved memory separately and average
                mem_arr = self._encode_texts(retrieved_memories)

        if memory_emb is None:
            memory_emb = self._fuse_memory_embeddings(
                session_emb=session_emb,
                memory_embs=mem_arr,
                fusion_mode=fusion_mode,
                fusion_tau=fusion_tau
            )

        # Concatenate session and memory embeddings
        state_emb = np.concatenate([session_emb, memory_emb], axis=0)

        return state_emb


class OpEncoder:
    """
    Encodes operation descriptions into embeddings.
    Supports HuggingFace encoder backbones and SentenceTransformer models.

    Can optionally use a shared BaseTextEncoder to avoid loading multiple model copies.
    """
    def __init__(self, model_name: str = "allenai/longformer-base-4096", device: str = 'cuda',
                 base_encoder: Optional['BaseTextEncoder'] = None,
                 encode_batch_size: int = 64, use_flash_attn: bool = True):
        if base_encoder is not None:
            # Use shared encoder
            self._base_encoder = base_encoder
            self.model_name = base_encoder.model_name
            self.device = base_encoder.device
        else:
            # Create own encoder
            self._base_encoder = BaseTextEncoder(
                model_name=model_name,
                device=device,
                encode_batch_size=encode_batch_size,
                use_flash_attn=use_flash_attn
            )
            self.model_name = model_name
            self.device = device

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension of the encoder"""
        return self._base_encoder.embedding_dim

    def encode(self, op_descriptions: List[str]) -> np.ndarray:
        """
        Encode operation descriptions
        Returns: [num_ops, op_dim]
        """
        return self._base_encoder.encode(op_descriptions)

    def encode_single(self, op_description: str) -> np.ndarray:
        """Encode single operation"""
        emb = self._base_encoder.encode(op_description)
        if emb.ndim == 2:
            return emb[0]
        return emb
