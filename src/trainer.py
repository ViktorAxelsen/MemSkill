"""
Trainer: Training loop with PPO (Proximal Policy Optimization) algorithm
"""
import torch
import torch.optim as optim
import numpy as np
import os
import json
import random
import logging
import multiprocessing as mp
import threading
import ast
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import sys
import wandb
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller import PPOController, StateEncoder, OpEncoder, PPOBuffer
from src.memory_bank import MemoryBank
from src.operation_bank import OperationBank
from src.executor import Executor
from src.designer import Designer, DesignerCase, EvolutionSnapshotManager
from src.data_processing import get_processor, DataProcessor
from src.data_processing.alfworld import (
    ALFWorldOfflineDataset,
    chunk_trajectories_by_tokens,
)
from src.eval import get_evaluator
from rag_utils import get_embeddings, init_context_model, init_query_model
from eval_utils import llm_judge
from prompts.prompt_pool import LLM_JUDGE_GENERAL_PROMPT


class BaseTrainer:
    """
    Trainer for Agentic Memory System
    Uses PPO (Proximal Policy Optimization) for controller training
    - On-policy learning with trajectory collection
    - GAE (Generalized Advantage Estimation) for advantage computation
    - Clipped surrogate objective for stable updates
    - Multiple update epochs on collected data
    """
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = args.device

        # Initialize encoders (each with its own state encoder backbone to reduce thread contention)
        encode_batch_size = getattr(config, 'encode_batch_size', 64)
        use_flash_attn = bool(getattr(config, "use_flash_attn", True))
        self.state_encoder = StateEncoder(
            model_name=config.state_encoder,
            device=self.device,
            fusion_mode=getattr(config, 'state_fusion', 'mean'),
            fusion_tau=getattr(config, 'state_fusion_tau', 1.0),
            encode_batch_size=encode_batch_size,
            use_flash_attn=use_flash_attn
        )
        self.op_encoder = OpEncoder(
            model_name=config.op_encoder,
            device=self.device,
            encode_batch_size=encode_batch_size,
            use_flash_attn=use_flash_attn
        )
        # Dedicated state encoder for memory bank embeddings (avoid contention with policy encoder)
        self.memory_bank_encoder = StateEncoder(
            model_name=config.state_encoder,
            device=self.device,
            fusion_mode=getattr(config, 'state_fusion', 'mean'),
            fusion_tau=getattr(config, 'state_fusion_tau', 1.0),
            encode_batch_size=encode_batch_size,
            use_flash_attn=use_flash_attn
        )

        # Derive embedding dimensions from actual encoders
        state_encoder_dim = self.state_encoder.embedding_dim
        op_encoder_dim = self.op_encoder.embedding_dim

        # State dim = session_emb + memory_emb (concatenated), both from state_encoder
        state_dim = 2 * state_encoder_dim
        op_dim = op_encoder_dim

        # Update config with actual dimensions for reference
        config.state_dim = state_dim
        config.op_embedding_dim = op_dim

        # PPO Controller
        self.controller = PPOController(
            state_dim=state_dim,
            op_dim=op_dim,
            hidden_dim=config.controller_hidden_dim,
            device=self.device,
            gamma=getattr(config, 'gamma', 0.99),
            gae_lambda=getattr(config, 'gae_lambda', 0.95),
            clip_epsilon=getattr(config, 'clip_epsilon', 0.2),
            entropy_coef=getattr(config, 'entropy_coef', 0.01),
            value_coef=getattr(config, 'value_coef', 0.5),
            vf_clip=getattr(config, 'vf_clip', 0.0),
            new_action_p_min=getattr(config, 'new_action_p_min', 0.0),
            new_action_delta_max=getattr(config, 'new_action_delta_max', 0.0),
            action_top_k=getattr(config, 'action_top_k', 1)
        )

        self.operation_bank = OperationBank(
            encoder=self.op_encoder,
            max_ops=getattr(config, 'max_ops', 20),
            skip_noop=getattr(config, 'skip_noop', False)
        )
        self.operation_bank.set_new_operation_names([])
        self.new_action_bias_active = False
        self.new_action_bias_step = 0

        self.executor = Executor(args)

        # Setup logging (before Designer so it can use the logger)
        self.logger = self._setup_logging()

        # Initialize Designer with shared encoder for clustering
        if args.enable_designer:
            self.designer = Designer(
                args,
                collect_epochs_before_designer=getattr(config, 'collect_epochs_before_designer', 5),
                failure_window_epochs=getattr(config, 'designer_failure_window_epochs', 200),
                failure_pool_size=getattr(config, 'designer_failure_pool_size', 1000),
                num_clusters=getattr(config, 'designer_num_clusters', 5),
                samples_per_cluster=getattr(config, 'designer_samples_per_cluster', 3),
                f1_threshold=getattr(config, 'designer_f1_threshold', 0.5),
                encoder=self.state_encoder._base_encoder,
                logger=self.logger
            )
        else:
            self.designer = None

        # Optimizer
        self.optimizer = optim.Adam(self.controller.parameters(), lr=config.controller_lr)

        # PPO parameters
        self.ppo_epochs = getattr(config, 'ppo_epochs', 4)
        self.minibatch_size = getattr(config, 'minibatch_size', 32)
        self.max_grad_norm = getattr(config, 'max_grad_norm', 0.5)
        self.batch_size = getattr(config, 'batch_size', 4)
        self.target_kl = getattr(config, 'target_kl', 0.02)

        # Training state
        self.training_logs = []
        self.total_steps = 0

        # Stage reward tracking (for evolution feedback)
        # Collects rewards from each inner epoch within a stage
        self.stage_rewards: List[float] = []

        # Evolution snapshot manager (tracks operation bank snapshots across evolution stages)
        if args.enable_designer:
            self.snapshot_manager = EvolutionSnapshotManager(logger=self.logger)
        else:
            self.snapshot_manager = None

        # Initialize data processor and evaluator for the dataset
        self.data_processor = self._build_data_processor()
        self.evaluator = get_evaluator(args.dataset, args)
        self._retriever_prewarmed = False

    def is_interactive(self) -> bool:
        """Return True for interactive datasets (override in subclasses)."""
        return False

    def supports_parallel_env(self) -> bool:
        """Return True if interactive env can run parallel episodes."""
        return True

    def _prepare_conversation_for_episode(self, conversation: Dict) -> Dict:
        """Allow subclasses to clone/mutate per-episode inputs safely."""
        return conversation

    def _get_episode_workers(self) -> int:
        if self.is_interactive() and not self.supports_parallel_env():
            return 1
        return self.batch_size

    def _build_data_processor(self):
        session_mode = getattr(self.args, 'session_mode', 'turn-pair')
        if self.args.dataset == 'hotpotqa':
            chunk_size = getattr(self.config, 'chunk_size', 1024)
            chunk_overlap = getattr(self.config, 'chunk_overlap', 128)
            return get_processor(
                self.args.dataset,
                chunk_mode='fixed-length',
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        if session_mode == "fixed-length":
            chunk_size = getattr(self.config, 'chunk_size', 1024)
            chunk_overlap = getattr(self.config, 'chunk_overlap', 128)
            return get_processor(
                self.args.dataset,
                chunk_mode=session_mode,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        return get_processor(self.args.dataset, chunk_mode=session_mode)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging to both console and file"""
        logger = logging.getLogger('AgenticMemory')
        logger.setLevel(logging.INFO)

        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # File handler
        log_dir = getattr(self.args, 'log_dir', './logs')
        os.makedirs(log_dir, exist_ok=True)

        run_name = getattr(self.args, 'wandb_run_name', None) or 'default'
        run_name = run_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{run_name}_{timestamp}.log')

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        logger.info(f"Logging to {log_file}")
        return logger

    def _prewarm_retriever_models(self):
        if self._retriever_prewarmed:
            return
        retriever = getattr(self.args, "retriever", None)
        if not retriever:
            return
        try:
            self.log(f"Pre-warming retriever models for '{retriever}'...")
            init_context_model(retriever)
            init_query_model(retriever)
            self._retriever_prewarmed = True
            self.log(f"Pre-warming complete for '{retriever}'.")
        except Exception as exc:
            self.log(f"Pre-warming retriever failed: {exc}", level="warning")

    def log(self, message: str, level: str = 'info'):
        """Log message to both console and file"""
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'debug':
            self.logger.debug(message)

    def _get_args_snapshot(self) -> Dict:
        """Get a sanitized args snapshot for reproducibility (redacts secrets)."""
        try:
            args_dict = dict(vars(self.args))
        except Exception:
            return {}

        snapshot = {}
        for key, value in args_dict.items():
            key_lower = str(key).lower()
            if any(tok in key_lower for tok in ('key', 'token', 'password', 'secret')):
                snapshot[key] = '<redacted>'
            else:
                snapshot[key] = value
        return snapshot

    def _get_rng_state(self) -> Dict:
        """Capture RNG states so training can be resumed deterministically (best-effort)."""
        rng_state = {
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state['torch_cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        return rng_state

    def _set_rng_state(self, rng_state: Dict):
        """Restore RNG states from checkpoint (best-effort)."""
        if not isinstance(rng_state, dict):
            return

        python_state = rng_state.get('python_random_state', None)
        if python_state is not None:
            try:
                random.setstate(python_state)
            except Exception:
                pass

        numpy_state = rng_state.get('numpy_random_state', None)
        if numpy_state is not None:
            try:
                np.random.set_state(numpy_state)
            except Exception:
                pass

        torch_state = rng_state.get('torch_rng_state', None)
        if torch_state is not None:
            try:
                torch.set_rng_state(torch_state)
            except Exception:
                pass

        cuda_states = rng_state.get('torch_cuda_rng_state_all', None)
        if cuda_states is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass

    def train_episode(self, conversation_data: Dict, memory_bank: MemoryBank,
                      sessions, ppo_buffer: PPOBuffer,
                      outer_epoch: int = 0, inner_epoch: int = 0,
                      episode_length: int = None, precompute_embeddings: bool = True) -> Dict:
        """
        Train on one conversation (one episode)
        Collects trajectory into PPOBuffer
        Args:
            conversation_data: dict with conversation sessions and QA
            memory_bank: initialized memory bank
            sessions: list or iterator of session texts
            ppo_buffer: PPOBuffer to collect transitions
            outer_epoch: current outer epoch index
            inner_epoch: current inner epoch index
            episode_length: optional episode length for shaping budgets
            precompute_embeddings: whether to precompute session embeddings
        Returns:
            episode_log: dict with episode statistics
        """
        episode_log = {
            'steps': [],
            'total_reward': 0.0,
            'final_qa_performance': 0.0,
            'raw_performance': 0.0
        }

        # Process each session sequentially
        if precompute_embeddings and isinstance(sessions, list):
            if episode_length is None:
                episode_length = len(sessions)
            # Batch compute all session embeddings using the state encoder
            # Note: Retriever embeddings are computed on-demand when adding memories
            session_embeddings = self.state_encoder._encode_texts(sessions)
        else:
            session_embeddings = None
            if episode_length is None:
                episode_length = self.data_processor.get_episode_length(conversation_data)
                if episode_length is None:
                    self.log("Episode length unknown for interactive data; deferring process rewards.",
                             level='warning')

        if episode_length is None or episode_length <= 0:
            episode_length = 1

        desc = f'[Outer {outer_epoch+1}/{self.config.outer_epochs}, Inner {inner_epoch+1}/{self.config.inner_epochs}] Sessions'
        total = episode_length if episode_length > 0 else None
        steps_processed = 0
        for session_idx, session_text in tqdm(enumerate(sessions), total=total, desc=desc):
            step_log = self._process_session(
                session_text=session_text,
                memory_bank=memory_bank,
                session_idx=session_idx,
                ppo_buffer=ppo_buffer,
                episode_length=episode_length,
                session_embedding=session_embeddings[session_idx] if session_embeddings is not None else None
            )
            episode_log['steps'].append(step_log)
            steps_processed += 1

        # Evaluate on QA tasks (this is the delayed reward)
        # Pass context for case collection
        conversation_id = conversation_data.get('sample_id', str(id(conversation_data)))
        qa_performance, raw_performance = self._evaluate_qa(
            conversation_data, memory_bank,
            conversation_id=conversation_id,
            outer_epoch=outer_epoch,
            inner_epoch=inner_epoch,
            step=steps_processed  # step = number of sessions processed
        )
        episode_log['final_qa_performance'] = qa_performance
        episode_log['raw_performance'] = raw_performance
        episode_log['total_reward'] = qa_performance

        # Finish episode in buffer (set final reward)
        # Use reward redistribution for long horizon if configured
        redistribute = getattr(self.config, 'redistribute_reward', False)
        redistribution_decay = getattr(self.config, 'reward_redistribution_decay', 0.95)
        final_reward_last_ratio = getattr(self.config, 'final_reward_last_ratio', 0.0)
        ppo_buffer.finish_episode(
            final_reward=qa_performance,
            redistribute=redistribute,
            redistribution_decay=redistribution_decay,
            final_reward_last_ratio=final_reward_last_ratio
        )

        return episode_log

    def _process_session(self, session_text: str, memory_bank: MemoryBank,
                         session_idx: int, ppo_buffer: PPOBuffer,
                         episode_length: int = None,
                         session_embedding: np.ndarray = None) -> Dict:
        """
        Process one session: retrieve memories, select operation, execute
        For PPO: stores state, action, log_prob, value in buffer

        Args:
            session_embedding: Pre-computed session embedding (for batch efficiency).
                               If None, will compute on-the-fly.
            episode_length: Optional episode length for reward shaping. If None,
                            process rewards are deferred until episode ends.
        """
        step_log = {}

        # Use pre-computed state encoder embedding if provided, otherwise compute on-the-fly
        if session_embedding is None:
            session_embedding = self.state_encoder._encode_texts(session_text)
            if session_embedding.ndim == 2:
                session_embedding = session_embedding[0]

        # Retrieve relevant memories using state encoder embeddings (for training)
        retrieved_memories, retrieved_indices, retrieved_memory_embeddings = memory_bank.retrieve(
            session_embedding, use_state_encoder=True, return_embeddings=True
        )
        step_log['retrieved_memories'] = retrieved_memories
        step_log['retrieved_indices'] = retrieved_indices

        # Encode state
        state_embedding = self.state_encoder.encode(
            session_text,
            retrieved_memories,
            session_embedding=session_embedding,
            memory_embeddings=retrieved_memory_embeddings
        )
        step_log['state_embedding'] = state_embedding

        # Get candidate operations
        candidate_ops = self.operation_bank.get_candidate_operations()
        step_log['candidate_ops'] = [op.name for op in candidate_ops]

        # Get operation embeddings
        op_embeddings = np.vstack([op.embedding for op in candidate_ops])
        new_op_indices = self.operation_bank.get_new_action_indices(candidate_ops)
        if len(new_op_indices) > 0:
            new_op_mask = np.zeros(len(candidate_ops), dtype=np.float32)
            new_op_mask[new_op_indices] = 1.0
        else:
            new_op_mask = None

        # Controller selects operation(s) (samples from policy)
        # Use no_grad during rollout to avoid unnecessary graph construction
        state_tensor = torch.tensor(state_embedding, dtype=torch.float32).to(self.device)
        op_tensor = torch.tensor(op_embeddings, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            action_idx, log_prob, value = self.controller(
                state_tensor, op_tensor, deterministic=False, new_op_mask=new_op_mask
            )

        # Handle both single action (K=1) and top-K actions
        if isinstance(action_idx, list):
            # Top-K case: multiple selected operations
            selected_ops = [candidate_ops[idx] for idx in action_idx]
            selected_op_names = [op.name for op in selected_ops]
            self.log(f"Selected ops (top-K): {selected_op_names}")
            step_log['selected_op'] = selected_op_names  # List of names for top-K
        else:
            # Single action case
            selected_ops = [candidate_ops[action_idx]]
            self.log(f"Selected op: {selected_ops[0].name}")
            step_log['selected_op'] = selected_ops[0].name  # Single name for backward compatibility

        step_log['selected_ops'] = selected_ops  # List of Operation objects
        step_log['action_idx'] = action_idx
        step_log['log_prob'] = log_prob
        step_log['value'] = value

        # Execute operation(s) - optionally prepend fixed initial operations
        executor_ops = selected_ops
        # Executor handles the logic for single vs multiple operations internally
        exec_results = self.executor.execute_operation(
            operation=executor_ops,
            session_text=session_text,
            retrieved_memories=retrieved_memories
        )
        step_log['exec_results'] = [str(r) for r in exec_results]
        step_log['parse_total'] = len(exec_results)
        step_log['parse_fail_count'] = sum(1 for r in exec_results if not r.success)

        process_reward_meta = self._build_process_reward_meta(selected_ops, exec_results)
        step_len = episode_length if episode_length is not None and episode_length > 0 else 1
        process_reward = self._compute_process_reward_from_meta(process_reward_meta, step_len)
        step_log['process_reward'] = process_reward

        # Immediate reward = process reward
        immediate_reward = process_reward

        # Store transition in PPO buffer
        ppo_buffer.push(
            state_emb=state_embedding,
            op_embs=op_embeddings,
            action=action_idx,  # int or List[int]
            log_prob=log_prob,
            value=value,
            reward=immediate_reward,
            new_op_mask=new_op_mask
        )

        # Apply all results to memory bank
        operation_names = []
        seen = set()
        for op in selected_ops:
            name = getattr(op, "name", None)
            if name and name not in seen:
                seen.add(name)
                operation_names.append(name)
        success = self.executor.apply_to_memory_bank(
            results=exec_results,
            memory_bank=memory_bank,
            retrieved_indices=retrieved_indices,
            operation_name=operation_names
        )
        step_log['apply_success'] = success

        # Store for logging
        step_log['session_text'] = session_text
        step_log['session_idx'] = session_idx

        # Update memory bank timestep
        memory_bank.step()

        return step_log

    def _score_qa_responses(self, qa_list: List[Dict], valid_qa_indices: List[int],
                            ret, reward_metric: str, eval_args):
        f1_scores = []
        predictions = {}
        f1_scores_map = {}

        for i, response, _, success in ret:
            qa = qa_list[i]
            ground_truth = self.evaluator.get_ground_truth(qa)
            prediction = response.strip() if success and response is not None else ""
            if str(getattr(self.args, "dataset", "")).lower() == "hotpotqa":
                extractor = getattr(self.evaluator, "_extract_answer", None)
                if callable(extractor):
                    prediction = extractor(prediction)

            f1 = self.evaluator.compute_f1(prediction, ground_truth, qa)
            f1_scores.append(f1)
            predictions[i] = prediction
            f1_scores_map[i] = f1

        llm_judge_scores = {}
        if reward_metric == 'llm_judge':
            judge_task_args = []
            for qa_idx in valid_qa_indices:
                qa = qa_list[qa_idx]
                ground_truth = self.evaluator.get_ground_truth(qa)
                prediction = predictions.get(qa_idx, "")
                prompt = self.evaluator.build_judge_prompt(
                    question=qa['question'],
                    ground_truth=ground_truth,
                    prediction=prediction,
                    qa_item=qa
                )
                judge_task_args.append((qa_idx, prompt, eval_args))

            if len(judge_task_args) > 0:
                judge_scores = llm_judge(task_args=judge_task_args, args=eval_args)
                for idx, (qa_idx, _, _) in enumerate(judge_task_args):
                    llm_judge_scores[qa_idx] = judge_scores[idx]

        if reward_metric == 'llm_judge':
            judge_scores_list = [llm_judge_scores.get(idx, 0.0) for idx in valid_qa_indices]
            avg_reward = np.mean(judge_scores_list) if len(judge_scores_list) > 0 else 0.0
        else:
            avg_reward = np.mean(f1_scores) if len(f1_scores) > 0 else 0.0

        return avg_reward, predictions, f1_scores_map, llm_judge_scores, f1_scores

    def _evaluate_qa(self, conversation_data: Dict, memory_bank: MemoryBank,
                     conversation_id: str = None, outer_epoch: int = 0,
                     inner_epoch: int = 0, step: int = 0):
        """
        Evaluate QA performance on the conversation.

        Uses the evaluator for the configured dataset to handle:
        - QA filtering (e.g., skip adversarial questions)
        - Prompt building
        - Metric computation

        Also collects DesignerCase objects if case collection is active.

        Args:
            conversation_data: Conversation data dict
            memory_bank: Memory bank to retrieve from
            conversation_id: ID for case tracking
            outer_epoch: Current outer epoch for case tracking
            inner_epoch: Current inner epoch for case tracking
            step: Current step for case tracking

        Returns:
            Tuple(reward_value, raw_performance)
        """
        from llm_utils import get_llm_response

        self.log("Evaluating QA...")
        eval_args = self.evaluator.prepare_eval_args()

        # Get QA list using data processor
        qa_list = self.data_processor.get_qa_list(conversation_data)
        if len(qa_list) == 0:
            if hasattr(self.evaluator, "get_episode_reward"):
                reward = float(self.evaluator.get_episode_reward(conversation_data))
                return reward, reward
            return 0.0, 0.0

        # Filter valid QA items using evaluator
        valid_qa = self.evaluator.filter_qa_list(qa_list)
        if len(valid_qa) == 0:
            return 0.0, 0.0

        # Collect questions for batch embedding
        valid_qa_indices = [idx for idx, _ in valid_qa]
        questions = [qa['question'] for _, qa in valid_qa]

        # Batch compute all question embeddings at once
        q_embeddings = get_embeddings(
            eval_args.retriever,
            questions,
            'query'
        )

        # Build task_args for parallel LLM calls
        # Also store retrieval info for case collection
        task_args = []
        retrieval_info = {}  # qa_idx -> (retrieved_mems, retrieved_indices)

        for idx, (qa_idx, qa) in enumerate(valid_qa):
            question = qa['question']
            q_embedding = q_embeddings[idx]

            # Retrieve from memory bank
            retrieved_mems, retrieved_indices = memory_bank.retrieve(
                q_embedding, use_state_encoder=False
            )
            retrieval_info[qa_idx] = (retrieved_mems, list(retrieved_indices))

            # Build prompt using evaluator
            prompt = self.evaluator.build_prompt(question, retrieved_mems, qa)
            task_args.append((qa_idx, prompt, eval_args))

        # Call LLM in parallel
        if len(task_args) == 0:
            return 0.0, 0.0

        ret = get_llm_response(args=eval_args, task_args=task_args)

        # Compute scores and collect cases (rolling failure pool)
        reward_metric = getattr(self.config, 'reward_metric', 'f1').lower()
        collect_cases = (self.designer is not None)
        f1_threshold = getattr(self.config, 'designer_f1_threshold', 0.5)
        need_llm_judge = reward_metric == 'llm_judge'

        # Get memory bank snapshot once if collecting (avoid repeated serialization)
        memory_snapshot = memory_bank.to_dict() if collect_cases else None

        avg_reward, predictions, f1_scores_map, llm_judge_scores, _ = self._score_qa_responses(
            qa_list, valid_qa_indices, ret, reward_metric, eval_args
        )

        if collect_cases:
            # Collect cases
            for qa_idx in valid_qa_indices:
                qa = qa_list[qa_idx]
                retrieved_mems, retrieved_indices = retrieval_info.get(qa_idx, ([], []))
                f1 = f1_scores_map.get(qa_idx, 0.0)
                prediction = predictions.get(qa_idx, "")
                judge_score = llm_judge_scores.get(qa_idx, 0.0)

                # Get ground truth using evaluator (handles both 'answer' and 'answers' fields)
                case_ground_truth = self.evaluator.get_ground_truth(qa)
                if isinstance(case_ground_truth, list):
                    case_ground_truth_str = ', '.join(str(ans) for ans in case_ground_truth)
                else:
                    case_ground_truth_str = str(case_ground_truth)

                if need_llm_judge:
                    is_correct = (judge_score == 1.0)
                else:
                    is_correct = (f1 >= f1_threshold)

                case = DesignerCase(
                    query_id=f"{conversation_id}_{qa_idx}" if conversation_id else str(qa_idx),
                    question=qa['question'],
                    ground_truth=case_ground_truth_str,
                    evidence=qa.get('evidence', None),
                    category=qa.get('category', None),
                    memory_bank_snapshot=memory_snapshot,
                    retrieved_memories=retrieved_mems,
                    retrieved_indices=retrieved_indices,
                    prediction=prediction,
                    is_correct=is_correct,
                    f1_score=f1,
                    llm_judge_score=judge_score,
                    conversation_id=conversation_id,
                    epoch=outer_epoch * self.config.inner_epochs + inner_epoch,
                    step=step
                )
                self.designer.case_collector.add_case(case)

        return avg_reward, avg_reward

    def update_controller(self, ppo_buffer: PPOBuffer) -> Dict:
        """
        Update controller using PPO with minibatch updates
        - Compute returns and advantages using GAE
        - Multiple epochs of PPO updates with shuffled minibatches
        """
        if len(ppo_buffer) == 0:
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'topk_entropy': 0.0,
                'topk_mass': 0.0,
                'topk_bin_entropy': 0.0,
                'skipped': True
            }

        # Compute returns and advantages
        returns, advantages = ppo_buffer.compute_returns_and_advantages(
            gamma=self.controller.gamma,
            gae_lambda=self.controller.gae_lambda,
            last_value=0.0  # Episode ended
        )

        # Normalize advantages ONCE for the full batch (before minibatch splitting)
        # This ensures consistent scale across all minibatches
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get batch data
        batch = ppo_buffer.get_batch()
        n_samples = len(batch['states'])

        # Determine minibatch size (0 means full batch)
        minibatch_size = self.minibatch_size if self.minibatch_size > 0 else n_samples

        # Multiple PPO update epochs with minibatch
        total_loss_info = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0,
                          'topk_entropy': 0.0, 'topk_mass': 0.0, 'topk_bin_entropy': 0.0,
                          'approx_kl': 0.0, 'clip_frac': 0.0,
                          'explained_variance': 0.0, 'value_mean': 0.0,
                          'return_mean': 0.0, 'advantage_mean': 0.0}
        n_updates = 0
        early_stop = False

        for epoch in range(self.ppo_epochs):
            if early_stop:
                break

            # Shuffle indices at the start of each epoch
            indices = np.random.permutation(n_samples)

            # Track epoch-level KL for more stable early stopping
            epoch_kl_sum = 0.0
            epoch_mb_count = 0

            # Process minibatches
            for start in range(0, n_samples, minibatch_size):
                end = min(start + minibatch_size, n_samples)
                mb_indices = indices[start:end]

                # Extract minibatch
                mb_batch = {
                    'states': [batch['states'][i] for i in mb_indices],
                    'op_embs': [batch['op_embs'][i] for i in mb_indices],
                    'new_op_masks': [batch['new_op_masks'][i] for i in mb_indices],
                    'actions': [batch['actions'][i] for i in mb_indices],
                    'log_probs': [batch['log_probs'][i] for i in mb_indices],
                    'values': [batch['values'][i] for i in mb_indices],
                }
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]

                # Compute PPO loss on minibatch
                loss, loss_info = self.controller.compute_ppo_loss(mb_batch, mb_returns, mb_advantages)

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Accumulate loss info
                for key in total_loss_info:
                    if key in loss_info:
                        total_loss_info[key] += loss_info[key]
                n_updates += 1

                # Accumulate epoch-level KL
                epoch_kl_sum += loss_info.get('approx_kl', 0.0)
                epoch_mb_count += 1

            # Early stopping based on epoch-average KL (more stable than per-minibatch)
            if self.target_kl is not None and self.target_kl > 0 and epoch_mb_count > 0:
                epoch_avg_kl = epoch_kl_sum / epoch_mb_count
                if epoch_avg_kl > self.target_kl:
                    early_stop = True

        # Average loss info
        n_updates = max(n_updates, 1)
        for key in total_loss_info:
            total_loss_info[key] /= n_updates

        total_loss_info['n_updates'] = n_updates
        total_loss_info['skipped'] = False

        self.total_steps += 1

        return total_loss_info

    def train(self, train_data: List[Dict]):
        """
        Main training loop with PPO
        - Collects episodes (on-policy)
        - Computes advantages using GAE
        - Multiple PPO update epochs per batch
        """
        # Initialize wandb
        if mp.current_process().name == "MainProcess":
            wandb_key = getattr(self.args, 'wandb_key', None)
            if wandb_key:
                wandb.login(key=wandb_key, relogin=True)
        wandb.init(
            project=getattr(self.args, 'wandb_project', 'memskill'),
            name=getattr(self.args, 'wandb_run_name', None),
            config={
                # PPO hyperparameters
                'gamma': self.controller.gamma,
                'gae_lambda': self.controller.gae_lambda,
                'clip_epsilon': self.controller.clip_epsilon,
                'vf_clip': getattr(self.controller, 'vf_clip', 0.0),
                'entropy_coef': self.controller.entropy_coef,
                'value_coef': self.controller.value_coef,
                'ppo_epochs': self.ppo_epochs,
                'minibatch_size': self.minibatch_size,
                'target_kl': self.target_kl,
                'max_grad_norm': self.max_grad_norm,
                'batch_size': self.batch_size,
                'controller_lr': self.config.controller_lr,
                'action_top_k': self.controller.action_top_k,  # Top-K action selection
                # Architecture
                'controller_hidden_dim': self.config.controller_hidden_dim,
                'state_dim': self.config.state_dim,
                'op_embedding_dim': self.config.op_embedding_dim,
                # Training schedule
                'outer_epochs': self.config.outer_epochs,
                'inner_epochs': self.config.inner_epochs,
                'ema_alpha': getattr(self.config, 'ema_alpha', 0.1),
                # Other
                'dataset': self.args.dataset,
            }
        )

        self.log("=" * 80)
        self.log("Starting Agentic Memory Training (PPO)")
        self.log("=" * 80)

        self._prewarm_retriever_models()

        for outer_epoch in range(self.config.outer_epochs):
            self.log(f"\n{'='*80}")
            self.log(f"Outer Epoch {outer_epoch + 1}/{self.config.outer_epochs}")
            self.log(f"{'='*80}")

            # Inner loop: train controller with fixed op bank
            inner_logs = []

            for inner_epoch in range(self.config.inner_epochs):
                # Create fresh PPO buffer for collecting episodes
                ppo_buffer = PPOBuffer()

                # Collect multiple episodes (batch)
                batch_rewards = []
                batch_raw_rewards = []
                batch_steps = []

                bias_scale = 0.0
                bias_steps = getattr(self.config, 'new_action_bias_steps', 0)
                if self.new_action_bias_active and bias_steps > 0:
                    bias_scale = max(0.0, 1.0 - (self.new_action_bias_step / bias_steps))
                self.controller.set_new_action_bias_scale(bias_scale)

                # Parallel episode collection using ThreadPoolExecutor
                max_workers = self._get_episode_workers()
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            self._run_single_episode,
                            train_data, outer_epoch, inner_epoch
                        )
                        for _ in range(self.batch_size)
                    ]

                    for future in as_completed(futures):
                        result = future.result()
                        episode_log = result['episode_log']
                        local_buffer = result['local_buffer']
                        op_stats = result['op_stats']

                        # Merge local buffer into main buffer
                        ppo_buffer.merge(local_buffer)
                        batch_rewards.append(episode_log['total_reward'])
                        batch_raw_rewards.append(
                            episode_log.get('raw_performance', episode_log['total_reward'])
                        )
                        batch_steps.extend(episode_log['steps'])

                        # Update operation statistics (in main thread)
                        for stat in op_stats:
                            op = self.operation_bank.get_operation(stat['op_name'])
                            op.update_stats(stat['reward'])

                # Update EMA based on batch-level usage (after all episodes collected)
                # Handle both single op (string) and top-K ops (list of strings)
                op_usage_counts = {}
                for step in batch_steps:
                    selected_op = step.get('selected_op', 'unknown')
                    if isinstance(selected_op, list):
                        for op_name in selected_op:
                            op_usage_counts[op_name] = op_usage_counts.get(op_name, 0) + 1
                    else:
                        op_usage_counts[selected_op] = op_usage_counts.get(selected_op, 0) + 1
                total_steps = len(batch_steps)
                ema_alpha = getattr(self.config, 'ema_alpha', 0.1)
                self.operation_bank.batch_update_ema(op_usage_counts, total_steps, ema_alpha)

                # PPO update on collected episodes
                update_info = self.update_controller(ppo_buffer)

                # Log
                avg_reward = np.mean(batch_rewards)
                avg_raw_performance = (
                    np.mean(batch_raw_rewards) if len(batch_raw_rewards) > 0 else 0.0
                )
                inner_logs.append({
                    'inner_epoch': inner_epoch,
                    'reward': avg_reward,
                    'raw_performance': avg_raw_performance,
                    'steps': batch_steps,
                    **update_info
                })

                if self.new_action_bias_active and bias_steps > 0:
                    self.new_action_bias_step += 1
                    if self.new_action_bias_step >= bias_steps:
                        self.new_action_bias_active = False
                        self.controller.set_new_action_bias_scale(0.0)
                        self.operation_bank.set_new_operation_names([])

                # Wandb logging for PPO metrics
                global_step = outer_epoch * self.config.inner_epochs + inner_epoch
                # Compute process reward stats
                process_rewards = [s.get('process_reward', 0.0) for s in batch_steps]
                match_rate = sum(1 for p in process_rewards if p == 0.0) / max(len(process_rewards), 1)

                # Compute operation usage statistics
                # Handle both single op (string) and top-K ops (list of strings)
                op_counts = {}
                for step in batch_steps:
                    selected_op = step.get('selected_op', 'unknown')
                    if isinstance(selected_op, list):
                        for op_name in selected_op:
                            op_counts[op_name] = op_counts.get(op_name, 0) + 1
                    else:
                        op_counts[selected_op] = op_counts.get(selected_op, 0) + 1
                num_episodes = max(len(batch_rewards), 1)

                # New-operation exploration stats
                # Handle both single op (string) and top-K ops (list of strings)
                new_op_names = set(self.operation_bank.new_operation_names)
                new_op_selected = 0
                for step in batch_steps:
                    selected_op = step.get('selected_op')
                    if isinstance(selected_op, list):
                        new_op_selected += sum(1 for op in selected_op if op in new_op_names)
                    elif selected_op in new_op_names:
                        new_op_selected += 1
                new_op_selected_rate = new_op_selected / max(len(batch_steps), 1)

                parse_fail_count = sum(step.get('parse_fail_count', 0) for step in batch_steps)
                parse_total = sum(step.get('parse_total', 0) for step in batch_steps)
                parse_fail_rate = parse_fail_count / max(parse_total, 1)

                wandb_log = {
                    # Training metrics
                    'train/reward': avg_reward,
                    'train/raw_performance': avg_raw_performance,
                    'train/reward_std': np.std(batch_rewards) if len(batch_rewards) > 1 else 0.0,
                    'train/reward_min': np.min(batch_rewards) if len(batch_rewards) > 0 else 0.0,
                    'train/reward_max': np.max(batch_rewards) if len(batch_rewards) > 0 else 0.0,
                    'train/episode_length': len(batch_steps) / num_episodes,
                    # Process reward (op selection quality)
                    'train/process_reward_mean': np.mean(process_rewards) if process_rewards else 0.0,
                    # PPO losses
                    'ppo/policy_loss': update_info.get('policy_loss', 0.0),
                    'ppo/value_loss': update_info.get('value_loss', 0.0),
                    'ppo/entropy': update_info.get('entropy', 0.0),
                    'ppo/topk_entropy': update_info.get('topk_entropy', 0.0),
                    'ppo/topk_mass': update_info.get('topk_mass', 0.0),
                    'ppo/topk_bin_entropy': update_info.get('topk_bin_entropy', 0.0),
                    'ppo/approx_kl': update_info.get('approx_kl', 0.0),
                    'ppo/clip_frac': update_info.get('clip_frac', 0.0),
                    'ppo/n_updates': update_info.get('n_updates', 0),
                    # Value function diagnostics
                    'ppo/explained_variance': update_info.get('explained_variance', 0.0),
                    'ppo/value_mean': update_info.get('value_mean', 0.0),
                    'ppo/return_mean': update_info.get('return_mean', 0.0),
                    'ppo/advantage_mean': update_info.get('advantage_mean', 0.0),
                    # Training state
                    'train/total_steps': self.total_steps,
                    'train/outer_epoch': outer_epoch,
                    'train/inner_epoch': inner_epoch,
                    'train/global_step': global_step,
                    'train/parse_fail_count': parse_fail_count,
                    'train/parse_fail_rate': parse_fail_rate,
                    # Operation statistics
                    'operation/num_operations': len(self.operation_bank.operations),
                    'operation/match_rate': match_rate,
                    # New op exploration
                    'explore/new_op_count': len(new_op_names),
                    'explore/new_op_selected_count': new_op_selected,
                    'explore/new_op_selected_rate': new_op_selected_rate,
                    'explore/new_action_bias_scale': bias_scale,
                    'explore/new_action_bias_active': float(self.new_action_bias_active),
                    'explore/new_action_bias_step': self.new_action_bias_step,
                    'explore/new_action_p_min': getattr(self.config, 'new_action_p_min', 0.0),
                    'explore/new_action_delta_max': getattr(self.config, 'new_action_delta_max', 0.0),
                }
                # Add per-operation average call counts
                for op_name, count in op_counts.items():
                    avg_calls = count / num_episodes
                    wandb_log[f'operation/{op_name}_avg'] = avg_calls
                wandb.log(wandb_log, step=global_step)


                # Track reward for stage average calculation
                self.stage_rewards.append(avg_reward)

                if (inner_epoch + 1) % 1 == 0:
                    recent_logs = inner_logs[-10:]
                    avg_reward = np.mean([log['reward'] for log in recent_logs])
                    avg_policy_loss = np.mean([log.get('policy_loss', 0) for log in recent_logs])
                    avg_value_loss = np.mean([log.get('value_loss', 0) for log in recent_logs])
                    avg_entropy = np.mean([log.get('entropy', 0) for log in recent_logs])
                    self.log(f"  Inner Epoch {inner_epoch + 1}/{self.config.inner_epochs}: "
                             f"Reward={avg_reward:.4f}, PolicyLoss={avg_policy_loss:.4f}, "
                             f"ValueLoss={avg_value_loss:.4f}, Entropy={avg_entropy:.4f}")

            # Outer loop: evolve operation bank
            if self.designer is not None and (outer_epoch + 1) % self.config.designer_freq == 0:
                self.log(f"\n{'='*80}")
                self.log("Evolving Operation Bank...")
                self.log(f"{'='*80}")

                # =====================================================================
                # Step 1: Calculate stage average reward (using last 1/4 of rewards)
                # =====================================================================
                stage_reward_fraction = getattr(self.config, 'stage_reward_fraction', 0.25)
                if len(self.stage_rewards) > 0:
                    # Use last fraction of stage rewards for more stable estimate
                    n_rewards = len(self.stage_rewards)
                    start_idx = int(n_rewards * (1.0 - stage_reward_fraction))
                    stable_rewards = self.stage_rewards[start_idx:]
                    use_moving_avg = bool(getattr(self.config, 'stage_reward_use_moving_avg', False))
                    if use_moving_avg and len(stable_rewards) > 0:
                        window = int(getattr(self.config, 'stage_reward_window', 5) or 1)
                        window = max(1, window)
                        rewards_arr = np.array(stable_rewards, dtype=np.float32)
                        window = min(window, len(rewards_arr))
                        if window == 1:
                            stage_avg_reward = float(np.mean(rewards_arr))
                        else:
                            cumsum = np.cumsum(rewards_arr, dtype=np.float64)
                            cumsum = np.concatenate(([0.0], cumsum))
                            moving_avg = (cumsum[window:] - cumsum[:-window]) / window
                            stage_avg_reward = float(np.mean(moving_avg)) if len(moving_avg) > 0 else 0.0
                        self.log(
                            f"Stage average reward (last {stage_reward_fraction*100:.0f}%, "
                            f"moving avg window={window}): {stage_avg_reward:.4f} "
                            f"(from {len(stable_rewards)} samples)"
                        )
                    else:
                        stage_avg_reward = np.mean(stable_rewards) if len(stable_rewards) > 0 else 0.0
                        self.log(
                            f"Stage average reward (last {stage_reward_fraction*100:.0f}%): "
                            f"{stage_avg_reward:.4f} (from {len(stable_rewards)} samples)"
                        )
                else:
                    stage_avg_reward = 0.0
                    self.log("No stage rewards collected")

                # =====================================================================
                # Step 2: Add current snapshot to manager
                # =====================================================================
                # The pending_evolution_result (set after last evolve) will be consumed here
                is_new_best = self.snapshot_manager.add_snapshot(
                    operation_bank=self.operation_bank,
                    avg_reward=stage_avg_reward
                )

                # Clear stage rewards for next stage
                self.stage_rewards = []

                # =====================================================================
                # Step 3: Check early stopping conditions
                # =====================================================================
                max_evolves = getattr(self.config, 'max_designer_evolves', 6)
                patience = getattr(self.config, 'designer_early_stop_patience', 3)

                if self.snapshot_manager.should_stop_evolving(max_evolves, patience):
                    self.log(f"\n{'='*80}")
                    self.log("STOPPING TRAINING: Evolution early stop triggered")
                    self.log(f"Total evolves: {self.snapshot_manager.total_evolves}, "
                             f"Consecutive no improvement: {self.snapshot_manager.consecutive_no_improvement}")
                    self.log(f"{'='*80}")

                    # Log early stop to wandb
                    wandb.log({
                        'evolution/early_stop': 1,
                        'evolution/total_evolves': self.snapshot_manager.total_evolves,
                        'evolution/best_reward': self.snapshot_manager.get_best_snapshot().avg_reward if self.snapshot_manager.get_best_snapshot() else 0.0,
                    }, step=global_step)

                    # Break out of outer loop
                    break

                # =====================================================================
                # Step 4: Rollback to best snapshot if current is not best
                # =====================================================================
                best_snapshot = self.snapshot_manager.get_best_snapshot()
                if not is_new_best and best_snapshot is not None:
                    self.log(f"Rolling back to best snapshot (stage {best_snapshot.stage_id}, "
                             f"reward={best_snapshot.avg_reward:.4f})")
                    # Restore operation bank from best snapshot
                    self.operation_bank = OperationBank.from_dict(
                        best_snapshot.operation_bank_dict,
                        encoder=self.op_encoder
                    )

                # =====================================================================
                # Step 5: Generate feedback for prompts
                # =====================================================================
                # Simple feedback for analysis prompt (focuses on failure cases)
                evolution_feedback = self.snapshot_manager.format_feedback_for_prompt(detailed=True)
                # Detailed feedback for refinement prompt (includes before/after comparison)
                evolution_feedback_detailed = self.snapshot_manager.format_evolution_feedback_for_refinement()
                if evolution_feedback:
                    self.log(f"Evolution feedback (for analysis):\n{evolution_feedback}")
                if evolution_feedback_detailed:
                    self.log(f"Evolution feedback (for refinement):\n{evolution_feedback_detailed}")

                # =====================================================================
                # Step 6: Log failure pool stats
                # =====================================================================
                num_cases = 0
                num_failures = 0
                if self.designer is not None:
                    num_cases = len(self.designer.case_collector.get_all_cases())
                    num_failures = num_cases
                    self.log(f"Failure pool size: {num_cases} cases")

                # =====================================================================
                # Step 7: Two-stage designer evolution with retry logic
                # =====================================================================
                max_retries = getattr(self.config, 'op_evolution_trials', 3)
                evolution_applied = False
                evolution_result = None

                # Determine whether to use saved analysis cases from best snapshot
                # or prepare fresh cases from current stage's collection.
                # If current is NOT best (rolled back to best), use best's saved cases.
                use_saved_cases = (not is_new_best and best_snapshot is not None
                                   and best_snapshot.analysis_cases is not None
                                   and len(best_snapshot.analysis_cases) > 0)

                if use_saved_cases:
                    # Use best snapshot's saved analysis cases
                    self.log(f"Using saved analysis cases from best snapshot (stage {best_snapshot.stage_id})")
                    analysis_prompt = self.designer.build_analysis_prompt_from_saved_cases(
                        saved_cases=best_snapshot.analysis_cases,
                        operation_bank=self.operation_bank,
                        evolution_feedback=evolution_feedback
                    )
                    prepared_data = {
                        'analysis_prompt': analysis_prompt,
                        'analysis_cases': best_snapshot.analysis_cases,  # Already serialized
                        'evolution_feedback': evolution_feedback
                    }
                else:
                    # Prepare fresh evolution (expensive: embedding, clustering)
                    prepared_data = self.designer.prepare_evolution(
                        self.operation_bank, evolution_feedback=evolution_feedback
                    )
                    # Save analysis_cases to current snapshot (for future use if this becomes best)
                    if prepared_data is not None and 'analysis_cases' in prepared_data:
                        # Serialize DesignerCase objects to dicts for storage
                        serialized_cases = [case.to_dict() for case in prepared_data['analysis_cases']]
                        self.snapshot_manager.set_latest_snapshot_analysis_cases(serialized_cases)
                        self.log(f"Saved {len(serialized_cases)} analysis cases to current snapshot")

                if prepared_data is None:
                    self.log("No cases to analyze, skipping evolution")
                    evolution_result = {'action': 'no_change', 'reasoning': 'No cases to analyze'}
                else:
                    # Retry the LLM calls up to max_retries times
                    for attempt in range(1, max_retries + 1):
                        self.log(f"Designer evolution attempt {attempt}/{max_retries}")
                        evolution_result = self.designer.run_evolution(
                            operation_bank=self.operation_bank,
                            prepared_data=prepared_data,
                            evolution_feedback_for_refinement=evolution_feedback_detailed
                        )
                        evolution_applied = self.designer.apply_evolution(self.operation_bank, evolution_result)

                        if evolution_applied:
                            self.log(f"Evolution succeeded on attempt {attempt}")
                            break
                        else:
                            self.log(f"Evolution attempt {attempt} returned no changes: {evolution_result.get('reasoning', 'N/A')}")
                            if attempt < max_retries:
                                self.log(f"Retrying evolution...")

                    # Reset collection state but keep rolling failure pool
                    self.designer.case_collector.clear(reset_pool=False)

                # =====================================================================
                # Step 8: Store evolution result for next snapshot and increment counter
                # =====================================================================
                # The evolution_result will be associated with the NEXT snapshot
                # (because this evolution will affect the next stage's training)
                self.snapshot_manager.set_pending_evolution_result(evolution_result)
                self.snapshot_manager.increment_evolve_count()

                # Activate new action bias if changes were applied
                if evolution_applied:
                    self.new_action_bias_active = True
                    self.new_action_bias_step = 0
                    self.log(f"Evolution applied: {evolution_result.get('action', 'unknown')}")
                else:
                    reason = evolution_result.get('reasoning', 'N/A') if evolution_result else 'No cases to analyze'
                    self.log(f"No evolution applied after {max_retries} attempts: {reason}")

                # Log operation bank evolution to wandb
                action_map = {'add_new': 1, 'refine_existing': 2, 'no_change': 0, 'multi': 3}
                best_snap = self.snapshot_manager.get_best_snapshot()
                num_changes = 0
                if evolution_result:
                    changes = evolution_result.get('changes')
                    if isinstance(changes, list):
                        num_changes = len(changes)
                wandb.log({
                    'evolution/num_operations': len(self.operation_bank.operations),
                    'evolution/outer_epoch': outer_epoch,
                    'evolution/collected_cases': num_cases,
                    'evolution/failure_cases': num_failures,
                    'evolution/action': action_map.get(evolution_result.get('action', 'no_change'), 0),
                    'evolution/num_changes': num_changes,
                    'evolution/applied': 1 if evolution_applied else 0,
                    # Snapshot management metrics
                    'evolution/stage_avg_reward': stage_avg_reward,
                    'evolution/best_reward': best_snap.avg_reward if best_snap else 0.0,
                    'evolution/is_new_best': 1 if is_new_best else 0,
                    'evolution/total_evolves': self.snapshot_manager.total_evolves,
                    'evolution/consecutive_no_improvement': self.snapshot_manager.consecutive_no_improvement,
                    'evolution/num_snapshots': len(self.snapshot_manager.snapshots),
                    'evolution/failed_attempts_accumulated': len(self.snapshot_manager.failed_evolution_attempts),
                    'evolution/used_saved_cases': 1 if use_saved_cases else 0,
                }, step=global_step)

            # Save checkpoint
            if (outer_epoch + 1) % 1 == 0:
                self.save_checkpoint(outer_epoch + 1)

            # Store logs
            self.training_logs.extend(inner_logs)

            # Log outer epoch summary to wandb
            outer_epoch_reward = np.mean([log['reward'] for log in inner_logs])
            wandb.log({
                'epoch/outer_epoch': outer_epoch,
                'epoch/avg_reward': outer_epoch_reward,
            }, step=(outer_epoch + 1) * self.config.inner_epochs - 1)

        # Final summary
        final_avg_reward = np.mean([log['reward'] for log in self.training_logs[-self.config.inner_epochs:]])
        wandb.log({
            'final/avg_reward': final_avg_reward,
            'final/total_steps': self.total_steps,
            'final/num_operations': len(self.operation_bank.operations),
        })

        # Finish wandb run
        wandb.finish()

        self.log("\n" + "=" * 80)
        self.log("Training Completed!")
        self.log("=" * 80)

    def _compute_process_reward(self, selected_ops, exec_results: List, episode_length: int = 1) -> float:
        """
        Compute process reward using budget-based allocation.

        Budget design ensures total process reward contribution is bounded:
        - process_budget: episode-level budget for match/mismatch (+/- quota per step)
        - failure_budget: episode-level budget for LLM failures
        - noop_match_scale: reduce NOOP match reward to discourage "always NOOP" hacking
        - shaping_scale: global scaling factor for all process rewards

        Args:
            selected_ops: Selected Operation object or list of Operation objects
            exec_results: List of ExecutionResult from LLM
            episode_length: Number of steps in the episode (for scaling)
        Returns:
            Process reward (positive for match, negative for mismatch/failure)
        """
        meta = self._build_process_reward_meta(selected_ops, exec_results)
        return self._compute_process_reward_from_meta(meta, episode_length)

    def _build_process_reward_meta(self, selected_ops, exec_results: List) -> Dict[str, object]:
        if selected_ops is None:
            selected_ops = []
        elif not isinstance(selected_ops, (list, tuple)):
            selected_ops = [selected_ops]

        selected_types = {op.update_type.lower() for op in selected_ops if op is not None}
        successful_results = [r for r in exec_results if r.success]
        actual_types = set(r.action_type.lower() for r in successful_results)

        return {
            'selected_types': sorted(selected_types),
            'actual_types': sorted(actual_types),
            'exec_total': len(exec_results),
            'exec_success': len(successful_results)
        }

    def _compute_process_reward_from_meta(self, meta: Dict[str, object], episode_length: int) -> float:
        # Budget parameters (episode-level)
        process_budget = getattr(self.config, 'process_budget', 0.10)
        failure_budget = getattr(self.config, 'failure_budget', 0.05)
        noop_match_scale = getattr(self.config, 'noop_match_scale', 0.2)
        shaping_scale = getattr(self.config, 'shaping_scale', 1.0)

        L = max(int(episode_length), 1)
        quota = process_budget / L          # +/- quota per step
        failure_quota = failure_budget / L  # failure penalty per step

        selected_types = set(meta.get('selected_types', []))
        actual_types = set(meta.get('actual_types', []))
        exec_total = int(meta.get('exec_total', 0) or 0)
        exec_success = int(meta.get('exec_success', 0) or 0)

        # Case 1: All results failed (LLM parsing/API failure)
        if exec_total > 0 and exec_success == 0:
            return -shaping_scale * failure_quota

        # Case 2: No results at all (edge case, treat as neutral)
        if exec_total == 0:
            return 0.0

        # Check match against selected ops (set-level)
        if selected_types == actual_types:
            if selected_types == {'noop'}:
                return shaping_scale * (noop_match_scale * quota)
            return shaping_scale * quota

        intersection_types = selected_types.intersection(actual_types)
        intersection_count = len(intersection_types)
        if intersection_count > 0:
            union_count = len(selected_types.union(actual_types))
            if union_count > 0:
                overlap_reward = shaping_scale * quota * (intersection_count / union_count)
                if intersection_types == {'noop'}:
                    return noop_match_scale * overlap_reward
                return overlap_reward
            return 0.0

        return -shaping_scale * quota

    def _run_single_episode(self, train_data: List[Dict], outer_epoch: int,
                            inner_epoch: int) -> Dict:
        """
        Run a single episode for parallel batch collection.
        Each episode uses its own PPOBuffer and MemoryBank.

        Args:
            train_data: list of conversation data
            outer_epoch: current outer epoch index
            inner_epoch: current inner epoch index
        Returns:
            dict with 'episode_log', 'local_buffer', 'op_stats'
        """
        # Sample a conversation
        conv_idx = np.random.randint(len(train_data))
        conversation = self._prepare_conversation_for_episode(train_data[conv_idx])

        sessions, episode_length, precompute = self._prepare_sessions(conversation)

        # Initialize empty memory bank (independent per episode)
        memory_bank = self._initialize_memory_bank()

        # Create local PPO buffer for this episode
        local_buffer = PPOBuffer()

        # Train episode
        episode_log = self.train_episode(
            conversation, memory_bank, sessions, local_buffer,
            outer_epoch=outer_epoch, inner_epoch=inner_epoch,
            episode_length=episode_length, precompute_embeddings=precompute
        )

        # Collect operation stats to update later (thread-safe)
        # Handle both single op (string) and top-K ops (list of strings)
        op_stats = []
        for step in episode_log['steps']:
            selected_op = step['selected_op']
            if isinstance(selected_op, list):
                # Top-K case: record stats for each selected operation
                for op_name in selected_op:
                    op_stats.append({
                        'op_name': op_name,
                        'reward': episode_log['total_reward']
                    })
            else:
                # Single action case
                op_stats.append({
                    'op_name': selected_op,
                    'reward': episode_log['total_reward']
                })

        return {
            'episode_log': episode_log,
            'local_buffer': local_buffer,
            'op_stats': op_stats
        }

    def _initialize_memory_bank(self, top_k: Optional[int] = None) -> MemoryBank:
        """
        Initialize an EMPTY memory bank with state encoder for dual embeddings
        Returns:
            MemoryBank: empty memory bank
        """
        if top_k is None:
            top_k = self.config.mem_top_k
        memory_bank = MemoryBank(
            retriever_name=self.args.retriever,
            top_k=top_k,
            state_encoder=self.memory_bank_encoder
        )
        return memory_bank

    def _prepare_sessions(self, conversation: Dict):
        """
        Prepare session iterator with optional precompute support.

        Returns:
            sessions: list or iterator of chunks
            episode_length: int or None
            precompute_embeddings: bool
        """
        if self.is_interactive():
            sessions = self.data_processor.iter_chunks(conversation)
            episode_length = self.data_processor.get_episode_length(conversation)
            return sessions, episode_length, False

        sessions = self.data_processor.extract_chunks(conversation)
        return sessions, len(sessions), True

    def _extract_sessions(self, conversation: Dict) -> List[str]:
        """
        Extract session/chunk texts from conversation data (non-interactive only).

        This method is kept for backward compatibility; new code should use
        _prepare_sessions() to handle interactive datasets.
        """
        return self.data_processor.extract_chunks(conversation)

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint (PPO version)"""
        os.makedirs(self.args.save_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'controller_state_dict': self.controller.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'operation_bank': self.operation_bank.to_dict(),
            'total_steps': self.total_steps,
            # Reproducibility / resume
            'args': self._get_args_snapshot(),
            'config': vars(self.config),
            'rng_state': self._get_rng_state(),
            # Snapshot manager state
            'snapshot_manager': self.snapshot_manager.to_dict() if self.snapshot_manager else None,
            'stage_rewards': self.stage_rewards,
        }

        run = getattr(wandb, 'run', None)
        if run is not None:
            checkpoint['wandb_run_id'] = getattr(run, 'id', None)
            checkpoint['wandb_run_name'] = getattr(run, 'name', None)

        # Build filename with wandb run name if available
        run_name = getattr(self.args, 'wandb_run_name', None) or 'default'
        # Sanitize run name for filename (replace invalid chars)
        run_name = run_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        path = os.path.join(self.args.save_dir, f'{run_name}_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        self.log(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint (PPO version)"""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        self.controller.load_state_dict(checkpoint['controller_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load operation bank (controlled by config)
        if not getattr(self.config, 'skip_load_operation_bank', False):
            self.operation_bank = OperationBank.from_dict(
                checkpoint['operation_bank'],
                encoder=self.op_encoder
            )
            self.operation_bank.skip_noop = getattr(self.config, 'skip_noop', False)
        else:
            self.log("Skipping operation bank loading (skip_load_operation_bank=True)")

        self.total_steps = checkpoint.get('total_steps', 0)

        rng_state = checkpoint.get('rng_state', None)
        if rng_state is not None:
            self._set_rng_state(rng_state)

        # Restore snapshot manager state (controlled by config)
        if not getattr(self.config, 'skip_load_snapshot_manager', False):
            snapshot_manager_data = checkpoint.get('snapshot_manager', None)
            if snapshot_manager_data is not None and self.snapshot_manager is not None:
                self.snapshot_manager = EvolutionSnapshotManager.from_dict(
                    snapshot_manager_data, logger=self.logger
                )
        else:
            self.log("Skipping snapshot manager loading (skip_load_snapshot_manager=True)")

        # Restore stage rewards
        self.stage_rewards = checkpoint.get('stage_rewards', [])
        self.log(f"Loaded checkpoint from {path}")


class OfflineTrainer(BaseTrainer):
    """Trainer for offline datasets with full-session/chunk access."""

    def is_interactive(self) -> bool:
        return False


class _NoOpProcessor(DataProcessor):
    """Minimal processor placeholder for trainers that don't consume processor output."""
    def extract_chunks(self, data: Dict) -> List[str]:
        return []

    def get_sample_id(self, data: Dict) -> str:
        return str(data.get("sample_id") if isinstance(data, dict) else id(data))

    def get_qa_list(self, data: Dict) -> List[Dict[str, Any]]:
        return []


class AlfworldPairTrainer(BaseTrainer):
    """Trainer for ALFWorld offline pair sampling (A=memory build, B=interactive eval)."""

    def __init__(self, args, config):
        super().__init__(args, config)
        data_path = getattr(config, "alfworld_offline_data", None) or getattr(args, "data_file", None)
        if not data_path:
            raise ValueError("ALFWorld offline data path is required.")
        self.offline_dataset = ALFWorldOfflineDataset.load(data_path)
        self._eval_pool = None
        self._eval_pool_lock = threading.Lock()

        # Replace QA designer with interactive designer if enabled.
        if args.enable_designer:
            from src.interactive_designer import InteractiveDesigner
            self.designer = InteractiveDesigner(
                args,
                collect_epochs_before_designer=getattr(config, 'collect_epochs_before_designer', 5),
                failure_window_epochs=getattr(config, 'designer_failure_window_epochs', 200),
                failure_pool_size=getattr(config, 'designer_failure_pool_size', 1000),
                num_clusters=getattr(config, 'designer_num_clusters', 5),
                samples_per_cluster=getattr(config, 'designer_samples_per_cluster', 3),
                f1_threshold=getattr(config, 'designer_f1_threshold', 0.5),
                encoder=self.state_encoder._base_encoder,
                logger=self.logger
            )

    def is_interactive(self) -> bool:
        return True

    def supports_parallel_env(self) -> bool:
        # Avoid parallel episodes; batch-B runs its own parallel envs if enabled.
        return False

    def _get_episode_workers(self) -> int:
        return self.batch_size

    def _build_data_processor(self):
        return _NoOpProcessor()

    def _get_eval_pool(self) -> ProcessPoolExecutor:
        if self._eval_pool is None:
            with self._eval_pool_lock:
                if self._eval_pool is None:
                    workers = int(getattr(self.config, "alfworld_pair_b_workers", 0) or 0)
                    if workers <= 0:
                        workers = max(1, self.batch_size)
                    ctx = mp.get_context("spawn")
                    self._eval_pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        return self._eval_pool

    def _shutdown_eval_pool(self):
        if self._eval_pool is not None:
            self._eval_pool.shutdown(wait=True)
            self._eval_pool = None

    def train(self, train_data: List[Dict]):
        try:
            super().train(train_data)
        finally:
            self._shutdown_eval_pool()

    def _sample_pair(self) -> Tuple[List[Tuple[str, str, Dict]], List[Tuple[str, str, Dict]]]:
        batch_a_min = int(getattr(self.config, "alfworld_pair_a_min", 100))
        batch_a_max = int(getattr(self.config, "alfworld_pair_a_max", 200))
        batch_b_size = int(getattr(self.config, "alfworld_pair_b_size", 0) or 0)
        if batch_b_size <= 0:
            batch_b_size = self.batch_size
        same_type_prob = float(getattr(self.config, "alfworld_pair_same_type_prob", 0.8))
        return self.offline_dataset.sample_pair(
            batch_a_min=batch_a_min,
            batch_a_max=batch_a_max,
            batch_b_size=batch_b_size,
            same_type_prob=same_type_prob
        )

    def _build_memory_from_batch_a(self, batch_a: List[Tuple[str, str, Dict]],
                                   memory_bank: MemoryBank,
                                   ppo_buffer: PPOBuffer,
                                   outer_epoch: int,
                                   inner_epoch: int) -> List[Dict]:
        trajectories = []
        for _, _, entry in batch_a:
            if isinstance(entry, dict):
                traj = entry.get("trajectory") or ""
                if isinstance(traj, str) and traj.strip():
                    trajectories.append(traj.strip())

        chunk_size = getattr(self.config, "alfworld_pair_chunk_size", None)
        if chunk_size is None:
            chunk_size = getattr(self.config, "chunk_size", None)

        chunks = chunk_trajectories_by_tokens(trajectories, chunk_size)
        if not chunks:
            return []

        episode_length = len(chunks)
        session_embeddings = self.state_encoder._encode_texts(chunks)
        if hasattr(session_embeddings, "ndim") and session_embeddings.ndim == 1:
            session_embeddings = session_embeddings.reshape(1, -1)

        step_logs = []
        for session_idx, session_text in tqdm(
            enumerate(chunks),
            total=len(chunks),
            desc="ALFWorld batch A",
        ):
            step_log = self._process_session(
                session_text=session_text,
                memory_bank=memory_bank,
                session_idx=session_idx,
                ppo_buffer=ppo_buffer,
                episode_length=episode_length,
                session_embedding=session_embeddings[session_idx] if session_embeddings is not None else None
            )
            step_logs.append(step_log)
        return step_logs

    def _run_batch_b(self, batch_b: List[Tuple[str, str, Dict]],
                     memory_bank: MemoryBank) -> List[Dict]:
        if not batch_b:
            return []

        from src.alfworld_env_runner import run_alfworld_episode

        objectives = []
        query_texts = []
        expert_plans = []
        query_source = str(getattr(self.config, "alfworld_eval_query_source", "first_observation")
                           or "first_observation").lower()
        if query_source not in ("objective", "first_observation"):
            query_source = "first_observation"
        for _, _, entry in batch_b:
            objective = ""
            first_obs = ""
            expert_plan = []
            if isinstance(entry, dict):
                objective = entry.get("objective")
                if not isinstance(objective, str):
                    objective = ""
                if not objective:
                    first_obs = entry.get("first_observation") or ""
                    if isinstance(first_obs, str) and "Your task is to:" in first_obs:
                        match = re.search(r"Your task is to:\s*(.+)", first_obs, re.IGNORECASE)
                        if match:
                            objective = match.group(1).strip().rstrip(".")
                if not first_obs:
                    first_obs = entry.get("first_observation") or ""
                steps = entry.get("steps")
                if isinstance(steps, list) and steps:
                    first_step = steps[0] if isinstance(steps[0], dict) else {}
                    raw_plan = first_step.get("expert_plan") if isinstance(first_step, dict) else None
                    if isinstance(raw_plan, list):
                        if len(raw_plan) == 1 and isinstance(raw_plan[0], str):
                            candidate = raw_plan[0].strip()
                            if candidate.startswith("[") and candidate.endswith("]"):
                                try:
                                    parsed = ast.literal_eval(candidate)
                                    if isinstance(parsed, list):
                                        raw_plan = parsed
                                except (ValueError, SyntaxError):
                                    pass
                        expert_plan = [str(item).strip() for item in raw_plan if item]
                    elif isinstance(raw_plan, str):
                        raw_plan = raw_plan.strip()
                        expert_plan = [raw_plan] if raw_plan else []
            if not isinstance(first_obs, str):
                first_obs = ""
            objectives.append(objective)
            query_texts.append(objective if query_source == "objective" else first_obs)
            expert_plans.append(expert_plan)

        query_embeddings = {}
        valid_queries = [(idx, text) for idx, text in enumerate(query_texts) if text and text.strip()]
        if valid_queries:
            indices, texts = zip(*valid_queries)
            emb_matrix = get_embeddings(self.args.retriever, list(texts), 'query')
            for emb_idx, query_idx in enumerate(indices):
                query_embeddings[query_idx] = emb_matrix[emb_idx]

        tasks = []
        for idx, (task_type, gamefile, entry) in enumerate(batch_b):
            objective = objectives[idx]
            retrieved_memories = []
            retrieved_indices = []
            emb = query_embeddings.get(idx)
            if emb is not None and len(memory_bank.memories) > 0:
                retrieved_memories, retrieved_indices = memory_bank.retrieve(
                    emb, use_state_encoder=False
                )
            tasks.append({
                "task_type": task_type,
                "gamefile": gamefile,
                "objective": objective,
                "query": query_texts[idx],
                "retrieved_memories": list(retrieved_memories),
                "retrieved_indices": list(retrieved_indices),
                "expert_plan": list(expert_plans[idx]) if expert_plans else []
            })

        llm_args = {
            "model": getattr(self.args, "model", ""),
            "api_base": getattr(self.args, "api_base", ""),
            "api_key": getattr(self.args, "api_key", None),
            "temperature": getattr(self.config, "alfworld_action_temperature", 0.0),
            "top_p": getattr(self.config, "alfworld_action_top_p", 1.0),
            "max_tokens": getattr(self.config, "alfworld_action_max_tokens", 32),
            "seed": getattr(self.args, "seed", 42)
        }
        max_steps = int(getattr(self.config, "alfworld_pair_max_steps", 50))
        include_inventory = bool(getattr(self.config, "alfworld_include_inventory", True))

        results = []
        executor = self._get_eval_pool()
        futures = {
            executor.submit(
                run_alfworld_episode,
                task["gamefile"],
                task["objective"],
                task["retrieved_memories"],
                max_steps,
                llm_args,
                include_inventory,
                query_source,
                task.get("expert_plan") or []
            ): task for task in tasks
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="ALFWorld batch B",
        ):
            task = futures[future]
            try:
                outcome = future.result()
            except Exception as exc:
                outcome = {"error": str(exc)}
            merged = dict(task)
            merged.update(outcome)
            results.append(merged)

        return results

    def _run_single_episode(self, train_data: List[Dict], outer_epoch: int,
                            inner_epoch: int) -> Dict:
        batch_a, batch_b = self._sample_pair()

        memory_bank = self._initialize_memory_bank()
        local_buffer = PPOBuffer()

        step_logs = self._build_memory_from_batch_a(
            batch_a=batch_a,
            memory_bank=memory_bank,
            ppo_buffer=local_buffer,
            outer_epoch=outer_epoch,
            inner_epoch=inner_epoch
        )

        batch_b_results = self._run_batch_b(batch_b, memory_bank)
        successes = [1.0 if r.get("success") else 0.0 for r in batch_b_results]
        avg_reward = float(np.mean(successes)) if successes else 0.0

        if self.designer is not None:
            memory_snapshot = memory_bank.to_dict()
            batch_a_episode_length = len(step_logs)
            for result in batch_b_results:
                success = bool(result.get("success"))
                prediction = "SUCCESS" if success else "FAILED"
                query_text = result.get("query")
                if not isinstance(query_text, str) or not query_text.strip():
                    query_text = result.get("objective") or ""
                case = DesignerCase(
                    query_id=f"{result.get('task_type')}::{result.get('gamefile')}",
                    question=query_text,
                    ground_truth="task_success",
                    evidence=result.get("trajectory") or "",
                    category=None,
                    memory_bank_snapshot=memory_snapshot,
                    retrieved_memories=result.get("retrieved_memories") or [],
                    retrieved_indices=result.get("retrieved_indices") or [],
                    prediction=prediction,
                    is_correct=success,
                    f1_score=1.0 if success else 0.0,
                    llm_judge_score=1.0 if success else 0.0,
                    conversation_id=result.get("task_type"),
                    epoch=outer_epoch * self.config.inner_epochs + inner_epoch,
                    step=batch_a_episode_length
                )
                self.designer.case_collector.add_case(case)

        redistribute = getattr(self.config, 'redistribute_reward', False)
        redistribution_decay = getattr(self.config, 'reward_redistribution_decay', 0.95)
        final_reward_last_ratio = getattr(self.config, 'final_reward_last_ratio', 0.0)
        local_buffer.finish_episode(
            final_reward=avg_reward,
            redistribute=redistribute,
            redistribution_decay=redistribution_decay,
            final_reward_last_ratio=final_reward_last_ratio
        )

        episode_log = {
            'steps': step_logs,
            'total_reward': avg_reward,
            'final_qa_performance': avg_reward,
            'raw_performance': avg_reward,
            'batch_b_success': sum(successes),
            'batch_b_total': len(successes)
        }

        op_stats = []
        for step in step_logs:
            selected_op = step.get('selected_op')
            if isinstance(selected_op, list):
                for op_name in selected_op:
                    op_stats.append({'op_name': op_name, 'reward': avg_reward})
            else:
                op_stats.append({'op_name': selected_op, 'reward': avg_reward})

        return {
            'episode_log': episode_log,
            'local_buffer': local_buffer,
            'op_stats': op_stats
        }


def get_trainer(args, config) -> BaseTrainer:
    if getattr(args, "dataset", None) == "alfworld":
        offline_path = getattr(config, "alfworld_offline_data", None) or getattr(args, "alfworld_offline_data", None)
        if not offline_path:
            raise ValueError("ALFWorld requires --alfworld-offline-data for offline pair training.")
        config.alfworld_offline_data = offline_path
        return AlfworldPairTrainer(args, config)
    return OfflineTrainer(args, config)
