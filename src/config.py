"""
Configuration for Agentic Memory System
"""
import argparse
import math

_NEW_ACTION_P_REF = 0.01


def _compute_new_action_delta_max(p_min: float, p_ref: float) -> float:
    eps = 1e-8
    p_min = min(max(p_min, eps), 1.0 - eps)
    p_ref = min(max(p_ref, eps), 1.0 - eps)
    return float(math.log((p_min * (1.0 - p_ref)) / (p_ref * (1.0 - p_min))))


class AgenticMemoryConfig:
    def __init__(self):
        # Memory Bank settings
        self.mem_top_k = 5
        self.mem_top_k_eval = None

        # Operation Bank settings
        self.max_ops = 30  # Maximum number of operations in the bank
        self.initial_ops = ["insert", "update", "delete", "noop"]
        self.skip_noop = False  # Skip noop in initial operations

        # Controller settings
        self.controller_hidden_dim = 256
        self.controller_lr = 1e-4
        # Encoder models for state and operation embedding
        # Default state encoder model supports long sequences up to 4096 tokens
        # Alternative: sentence-transformers/all-MiniLM-L6-v2 (faster, shorter context)
        self.state_encoder = "allenai/longformer-base-4096"
        self.op_encoder = "allenai/longformer-base-4096"
        self.use_flash_attn = True  # Enable flash attention for compatible encoders
        # State fusion strategy for session + memory embeddings
        self.state_fusion = "mean"  # mean or sim_weighted
        self.state_fusion_tau = 1.0  # temperature for sim_weighted
        self.encode_batch_size = 64  # batch size for text encoding (to avoid OOM)
        # NOTE: state_dim and op_embedding_dim are derived from actual encoder models at runtime
        # state_dim = 2 * state_encoder_dim (session_emb || memory_emb concatenated)
        # These placeholders will be overwritten in trainer initialization
        self.state_dim = None  # Will be set from encoder's embedding_dim * 2
        self.op_embedding_dim = None  # Will be set from encoder's embedding_dim

        # Training settings
        self.inner_epochs = 20
        self.outer_epochs = 10
        self.gamma = 0.99  # discount factor
        # Episode reward metric: f1 or llm_judge
        self.reward_metric = "f1"

        # PPO settings
        self.gae_lambda = 0.95  # GAE lambda for advantage estimation
        self.clip_epsilon = 0.2  # PPO clipping parameter
        self.action_top_k = 1  # Number of top actions to select per step
        self.vf_clip = 0.0  # value function clipping; 0 disables (recommended default)
        self.ppo_epochs = 4  # number of PPO update epochs per batch
        self.minibatch_size = 32  # minibatch size for PPO updates; 0 means use full batch
        self.entropy_coef = 0.01  # entropy bonus coefficient
        self.value_coef = 0.5  # value loss coefficient
        # Early stopping: stop PPO epochs when approx_kl exceeds this threshold
        # Note: approx_kl is a sample estimate and can be slightly negative due to noise.
        self.target_kl = 0.02
        self.max_grad_norm = 0.5  # gradient clipping

        # Designer settings
        self.designer_freq = 1  # trigger every N outer epochs
        self.min_op_usage = 5  # minimum usage before considering deletion
        self.low_reward_threshold = 0.3
        # Case collection settings for designer
        self.collect_epochs_before_designer = 5  # Legacy (unused; rolling failure pool is always active)
        self.designer_failure_window_epochs = 100  # Rolling window (global inner epochs); 0 disables
        self.designer_failure_pool_size = 2000  # Max failures kept in pool; 0 disables
        self.designer_num_clusters = 5  # Number of clusters for failure case analysis
        self.designer_samples_per_cluster = 3  # Number of cases to sample from each cluster
        self.designer_f1_threshold = 0.5  # F1 threshold for success/failure classification
        self.op_evolution_trials = 3  # Max retry attempts for operation evolution
        self.designer_new_skill_hint = False  # Encourage proposing new skills in designer prompts
        self.designer_reflection_cycles = 3  # Max reflection cycles for designer analysis
        self.designer_max_changes = 1  # Max number of skill changes per evolution cycle
        self.designer_refine_only = False  # If true, ignore add_new operation changes

        # Evolution feedback and early stopping
        self.max_designer_evolves = 6  # Maximum number of evolution cycles before stopping
        self.designer_early_stop_patience = 3  # Stop if no improvement for N consecutive evolves
        self.stage_reward_fraction = 0.25  # Use last 25% of steps in each stage for reward calculation
        self.stage_reward_use_moving_avg = False  # Use moving average over tail instead of raw mean
        self.stage_reward_window = 5  # Window size for moving average if enabled

        # Executor settings
        self.max_new_tokens = 1024
        self.temperature = 0.0
        self.designer_model = None
        self.llm_judge_model = "openai/gpt-oss-120b"

        # Dataset settings
        self.dataset = "locomo"  # locomo, longmemeval, hotpotqa, or alfworld
        self.batch_size = 4  # episodes per PPO update
        self.hotpotqa_eval_file = "./data/eval_200.json"
        self.alfworld_eval_file = "./data/alfworld_expert_eval_in_distribution.json"

        # HotpotQA specific settings (fixed-length chunking)
        self.chunk_size = 1024  # Target chunk size in tokens for fixed-length chunking
        self.chunk_overlap = 128  # Overlap between chunks in tokens

        # ALFWorld settings (interactive)
        self.alfworld_action_max_tokens = 32  # Max tokens for action LLM
        self.alfworld_action_temperature = 0.0  # Action LLM temperature
        self.alfworld_action_top_p = 1.0  # Action LLM top-p
        self.alfworld_include_inventory = True  # Include inventory in chunk text
        self.alfworld_eval_query_source = "first_observation"  # objective or first_observation
        # ALFWorld offline pair training settings
        self.alfworld_offline_data = None  # Path to offline expert trajectories
        self.alfworld_pair_a_min = 100  # Min trajectories in batch A
        self.alfworld_pair_a_max = 200  # Max trajectories in batch A
        self.alfworld_pair_b_size = 0  # Batch B size (interactive envs); 0 uses batch_size
        self.alfworld_pair_same_type_prob = 0.8  # Probability A/B share same task type
        self.alfworld_pair_chunk_size = 2048  # Token budget for batch-A chunking
        self.alfworld_pair_max_steps = 50  # Max steps per interactive episode (batch B)
        self.alfworld_pair_b_workers = 64  # Max workers for batch-B envs


        self.ema_alpha = 0.1  # EMA smoothing factor for recent usage tracking
        # New-action exploration bias (logit adjustment)
        self.new_action_p_min = 0.30  # minimum total probability for new actions
        # max logit bias applied to new actions, computed for p_ref=0.01
        self.new_action_delta_max = _compute_new_action_delta_max(self.new_action_p_min, _NEW_ACTION_P_REF)
        self.new_action_bias_steps = 50  # linear decay steps after new actions appear

        # Process reward: budget-based allocation for match/mismatch feedback
        # Total process reward contribution is bounded by these budgets (episode-level)
        self.process_budget = 0.10  # +/- quota per step = process_budget / episode_length
        self.failure_budget = 0.05  # penalty for LLM parsing/API failures
        self.noop_match_scale = 0.2  # reduce NOOP match reward to discourage "always NOOP" hacking
        self.shaping_scale = 1.0  # global scaling factor for shaping rewards

        # Reward shaping for long horizon
        # redistribute_reward: spread final reward across all steps using exponential decay
        # This helps with credit assignment in long episodes
        self.redistribute_reward = True
        self.reward_redistribution_decay = 0.9  # How much weight decays per step backwards
        self.final_reward_last_ratio = 0.6  # Portion added to last step; remainder is redistributed

        # Checkpoint loading settings
        self.skip_load_operation_bank = False  # Skip loading operation bank from checkpoint
        self.skip_load_snapshot_manager = False  # Skip loading snapshot manager from checkpoint
        self.memory_cache_suffix = ""  # Optional suffix for memory cache filenames

    def update_from_args(self, args):
        """Update config from command line arguments"""
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)
        if self.mem_top_k_eval is None:
            self.mem_top_k_eval = self.mem_top_k
        if self.designer_model is None:
            self.designer_model = getattr(args, "model", None)
        return self


def get_agentic_memory_args():
    parser = argparse.ArgumentParser()

    # Dataset args
    parser.add_argument('--dataset', type=str, default='locomo',
                        choices=['locomo', 'longmemeval', 'hotpotqa', 'alfworld'])
    parser.add_argument('--data-file', type=str, default='./data/locomo10.json')
    parser.add_argument('--hotpotqa-eval-file', type=str, default='./data/eval_200.json',
                        help='HotpotQA eval data file (same format as training)')
    parser.add_argument('--alfworld-eval-file', type=str,
                        default='./data/alfworld_expert_eval_in_distribution.json',
                        help='ALFWorld eval data file (same format as offline train data)')
    parser.add_argument('--out-file', type=str, default='./results/agentic_memory_results.json')
    parser.add_argument('--session-mode', type=str, default='turn-pair',
                        choices=['turn', 'turn-pair', 'full-session', 'fixed-length', 'paragraph'],
                        help='Session granularity: turn (each utterance), turn-pair (dialogue exchanges), full-session (entire session), fixed-length (for hotpotqa)')
    parser.add_argument('--chunk-size', type=int, default=2048,
                        help='Target chunk size in tokens for fixed-length chunking (hotpotqa)')
    parser.add_argument('--chunk-overlap', type=int, default=256,
                        help='Overlap between chunks in tokens (hotpotqa)')
    # ALFWorld args
    parser.add_argument('--alfworld-action-max-tokens', type=int, default=32,
                        help='Max tokens for ALFWorld action LLM')
    parser.add_argument('--alfworld-action-temperature', type=float, default=0.0,
                        help='Temperature for ALFWorld action LLM')
    parser.add_argument('--alfworld-action-top-p', type=float, default=1.0,
                        help='Top-p for ALFWorld action LLM')
    parser.add_argument('--alfworld-include-inventory', type=int, default=1, choices=[0, 1],
                        help='Include inventory in ALFWorld chunk text (1=yes, 0=no)')
    parser.add_argument('--alfworld-eval-query-source', type=str, default='first_observation',
                        choices=['objective', 'first_observation'],
                        help='Query source for retrieval in ALFWorld eval')
    parser.add_argument('--alfworld-offline-data', type=str, default="./data/alfworld_train_offline.json",
                        help='Path to offline ALFWorld expert trajectories (pair training)')
    parser.add_argument('--alfworld-pair-a-min', type=int, default=60,
                        help='Min number of trajectories in batch A (offline pair training)')
    parser.add_argument('--alfworld-pair-a-max', type=int, default=80,
                        help='Max number of trajectories in batch A (offline pair training)')
    parser.add_argument('--alfworld-pair-b-size', type=int, default=8,
                        help='Batch B size (interactive envs); 0 uses --batch-size')
    parser.add_argument('--alfworld-pair-same-type-prob', type=float, default=0.8,
                        help='Probability A/B share same task type in offline pair sampling')
    parser.add_argument('--alfworld-pair-chunk-size', type=int, default=2048,
                        help='Token budget for batch-A chunking in offline pair training')
    parser.add_argument('--alfworld-pair-max-steps', type=int, default=50,
                        help='Max steps per interactive episode for batch B')
    parser.add_argument('--alfworld-pair-b-workers', type=int, default=64,
                        help='Max workers for batch-B envs')
    parser.add_argument('--overwrite', action='store_true',
                        help='Force re-extraction of memories even if cached files exist')

    # Model args
    parser.add_argument('--model', type=str, default='qwen/qwen3-next-80b-a3b-instruct')
    parser.add_argument('--designer-model', type=str, default=None,
                        help='Model for designer LLM; defaults to --model')
    parser.add_argument('--api', action='store_true', help='Use API instead of local model')
    parser.add_argument('--api-base', type=str, default='[YOUR_API_BASE]')
    parser.add_argument('--api-key', type=str, nargs='+',
                        default=["YOUR_API_KEY_1",
                                 "YOUR_API_KEY_2"])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--llm-judge-model', type=str, default='openai/gpt-oss-120b',
                        help='Model for LLM judge scoring')
    parser.add_argument('--batch-size', type=int, default=4, help='Episodes per PPO update')

    # Retriever args
    parser.add_argument('--retriever', type=str, default='contriever', choices=['dpr', 'contriever', 'dragon'])
    parser.add_argument('--mem-top-k', type=int, default=5, help='Top-k memories to retrieve during training')
    parser.add_argument('--mem-top-k-eval', type=int, default=None,
                        help='Top-k memories to retrieve during evaluation (defaults to --mem-top-k)')
    parser.add_argument('--max-ops', type=int, default=30,
                        help='Maximum number of operations in the bank')
    parser.add_argument('--skip-noop', action='store_true',
                        help='Skip noop operation in the initial operation bank')

    # Encoder args
    parser.add_argument('--state-encoder', type=str, default='Qwen/Qwen3-Embedding-0.6B',
                        help='Model for state encoding. Use "sentence-transformers/..." or "Qwen/Qwen3-Embedding-0.6B" for SentenceTransformer models')
    parser.add_argument('--op-encoder', type=str, default='Qwen/Qwen3-Embedding-0.6B',
                        help='Model for operation encoding. Use "sentence-transformers/..." or "Qwen/Qwen3-Embedding-0.6B" for SentenceTransformer models')
    parser.add_argument('--disable-flash-attn', action='store_false', dest='use_flash_attn', default=True,
                        help='Disable flash attention for compatible embedding encoders')
    parser.add_argument('--state-fusion', type=str, default='mean',
                        choices=['mean', 'sim_weighted'],
                        help='Fusion strategy for session and memory embeddings')
    parser.add_argument('--state-fusion-tau', type=float, default=1.0,
                        help='Temperature for sim_weighted fusion (higher -> flatter weights)')
    parser.add_argument('--encode-batch-size', type=int, default=64,
                        help='Batch size for text encoding (to avoid OOM on large batches)')

    # Training args
    parser.add_argument('--inner-epochs', type=int, default=20)
    parser.add_argument('--outer-epochs', type=int, default=10)
    parser.add_argument('--controller-lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--reward-metric', type=str, default='f1',
                        choices=['f1', 'llm_judge'],
                        help='Episode reward metric: f1 or llm_judge')

    # PPO args
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda for advantage estimation')
    parser.add_argument('--clip-epsilon', type=float, default=0.2, help='PPO clipping parameter')
    parser.add_argument('--action-top-k', type=int, default=1,
                        help='Number of top actions to select per step (default 1)')
    parser.add_argument('--vf-clip', type=float, default=0.0,
                        help='Value function clipping range; 0 disables value clipping')
    parser.add_argument('--ppo-epochs', type=int, default=4, help='Number of PPO update epochs per batch')
    parser.add_argument('--minibatch-size', type=int, default=32, help='Minibatch size for PPO; 0 uses full batch')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy bonus coefficient')
    parser.add_argument('--value-coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--target-kl', type=float, default=0.02,
                        help='Early stop PPO update epochs when approx_kl exceeds this threshold')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Gradient clipping')
    parser.add_argument('--new-action-p-min', type=float, default=0.30,
                        help='Minimum total probability mass for new actions')
    parser.add_argument('--new-action-bias-steps', type=int, default=50,
                        help='Linear decay steps for new-action bias after changes')

    # Designer args
    parser.add_argument('--designer-freq', type=int, default=1, help='Trigger designer every N outer epochs')
    parser.add_argument('--enable-designer', action='store_true', help='Enable operation evolution')
    parser.add_argument('--collect-epochs-before-designer', type=int, default=5,
                        help='Legacy (unused; rolling failure pool is always active)')
    parser.add_argument('--designer-failure-window-epochs', type=int, default=100,
                        help='Rolling window in global inner epochs for failure pool; 0 disables')
    parser.add_argument('--designer-failure-pool-size', type=int, default=2000,
                        help='Max failures kept in pool; 0 disables')
    parser.add_argument('--designer-num-clusters', type=int, default=5,
                        help='Number of clusters for failure case analysis')
    parser.add_argument('--designer-samples-per-cluster', type=int, default=3,
                        help='Number of cases to sample from each cluster')
    parser.add_argument('--designer-f1-threshold', type=float, default=0.5,
                        help='F1 threshold for success/failure classification')
    parser.add_argument('--designer-new-skill-hint', action='store_true',
                        help='Encourage proposing new skills in designer prompts')
    parser.add_argument('--designer-reflection-cycles', type=int, default=3,
                        help='Max reflection cycles for designer analysis (>=1)')
    parser.add_argument('--designer-max-changes', type=int, default=1,
                        help='Max number of skill changes per evolution cycle (>=1)')
    parser.add_argument('--designer-refine-only', action='store_true',
                        help='Only allow refining existing skills; ignore add_new changes')
    parser.add_argument('--op-evolution-trials', type=int, default=3,
                        help='Max retry attempts for operation evolution')
    parser.add_argument('--max-designer-evolves', type=int, default=20,
                        help='Maximum number of evolution cycles before stopping training')
    parser.add_argument('--designer-early-stop-patience', type=int, default=5,
                        help='Stop training if no improvement for N consecutive evolves')
    parser.add_argument('--stage-reward-fraction', type=float, default=0.25,
                        help='Use last X fraction of steps in each stage for reward calculation')
    parser.add_argument('--stage-reward-use-moving-avg', action='store_true',
                        help='Use moving average over tail instead of raw mean for stage reward')
    parser.add_argument('--stage-reward-window', type=int, default=5,
                        help='Window size for moving average stage reward (>=1)')

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--round', type=int, default=10, help='Number of retry rounds for LLM API calls')

    # Evaluation
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--load-checkpoint', type=str, default=None)
    parser.add_argument('--skip-load-operation-bank', action='store_true',
                        help='Skip loading operation bank from checkpoint')
    parser.add_argument('--skip-load-snapshot-manager', action='store_true',
                        help='Skip loading snapshot manager from checkpoint')
    parser.add_argument('--memory-cache-suffix', type=str, default='',
                        help='Optional suffix appended to memory cache filenames')

    # Wandb args
    parser.add_argument('--wandb-project', type=str, default='memskill', help='Wandb project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='Wandb run name')

    args = parser.parse_args()
    return args
