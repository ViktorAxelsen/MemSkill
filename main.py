"""
Main entry point for Agentic Memory System
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import random
import re
import multiprocessing as mp
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Any, List, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from src.config import AgenticMemoryConfig, get_agentic_memory_args
from src.trainer import BaseTrainer, get_trainer
from src.executor import ExecutionResult
from src.memory_bank import MemoryBank
from src.data_processing.alfworld import chunk_trajectories_by_tokens
from src.alfworld_env_runner import run_alfworld_episode
from rag_utils import get_embeddings
from eval_utils import llm_judge
from llm_utils import get_llm_response
from prompts.prompt_pool import LLM_JUDGE_GENERAL_PROMPT

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(data_file: str, dataset_type: str):
    """Load dataset"""
    with open(data_file, 'r') as f:
        if dataset_type == 'locomo':
            data = json.load(f)
        elif dataset_type == 'longmemeval':
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                data = [json.loads(line) for line in f.readlines()]
        elif dataset_type == 'hotpotqa':
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                data = [json.loads(line) for line in f.readlines()]
        elif dataset_type == 'alfworld':
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                data = [json.loads(line) for line in f.readlines()]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    return data


def split_data(data, dataset_type: str):
    """Split data into train/val/test"""
    if dataset_type == 'locomo':
        # LoCoMo uses fixed indices
        train_index = [0, 1, 2, 3, 4, 5]
        val_index = [6, 7]
        test_index = [8, 9]

        train_data = [data[i] for i in train_index if i < len(data)]
        val_data = [data[i] for i in val_index if i < len(data)]
        test_data = [data[i] for i in test_index if i < len(data)]

    elif dataset_type == 'longmemeval':
        # LongMemEval uses predefined splits
        splits_file = "./data/longmemeval_s_splits.json"
        if os.path.exists(splits_file):
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            train_idx = splits["train"]
            val_idx = splits["val"]
            test_idx = splits["test"]

            train_data = [data[i] for i in train_idx if i < len(data)]
            val_data = [data[i] for i in val_idx if i < len(data)]
            test_data = [data[i] for i in test_idx if i < len(data)]
        else:
            raise FileNotFoundError

    elif dataset_type == 'hotpotqa':
        train_data = data
        val_data = []
        test_data = []
    elif dataset_type == 'alfworld':
        if isinstance(data, dict):
            train_data = data
            val_data = {}
            test_data = {}
        else:
            n = len(data)
            train_end = int(0.8 * n)
            val_end = int(0.9 * n)
            train_data = data[:train_end]
            val_data = data[train_end:val_end]
            test_data = data[val_end:]
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return train_data, val_data, test_data


def _extract_memory_actions_step(trainer: BaseTrainer,
                                 memory_bank,
                                 session_text: str,
                                 session_embedding: np.ndarray = None,
                                 executor=None):
    if session_embedding is None:
        session_embedding = trainer.state_encoder._encode_texts(session_text)
        if session_embedding.ndim == 2:
            session_embedding = session_embedding[0]

    retrieved_memories, retrieved_indices, retrieved_memory_embeddings = memory_bank.retrieve(
        session_embedding, use_state_encoder=True, return_embeddings=True
    )

    state_embedding = trainer.state_encoder.encode(
        session_text,
        retrieved_memories,
        session_embedding=session_embedding,
        memory_embeddings=retrieved_memory_embeddings
    )

    candidate_ops = trainer.operation_bank.get_candidate_operations()
    op_embeddings = np.vstack([op.embedding for op in candidate_ops])

    state_tensor = torch.tensor(state_embedding, dtype=torch.float32).to(trainer.device)
    op_tensor = torch.tensor(op_embeddings, dtype=torch.float32).to(trainer.device)

    action_idx, _, _ = trainer.controller(state_tensor, op_tensor, deterministic=True)
    if isinstance(action_idx, list):
        selected_ops = [candidate_ops[idx] for idx in action_idx]
    else:
        selected_ops = [candidate_ops[action_idx]]

    executor_ops = [
        op for op in selected_ops
        if str(getattr(op, "name", "")).lower() != "noop"
        and str(getattr(op, "update_type", "")).lower() != "noop"
    ]
    # print([i.instruction_template for i in executor_ops])
    exec_impl = executor if executor is not None else trainer.executor
    exec_result = exec_impl.execute_operation(
        operation=executor_ops,
        session_text=session_text,
        retrieved_memories=retrieved_memories
    )

    operation_names = []
    seen = set()
    for op in selected_ops:
        name = getattr(op, "name", None)
        if name and name not in seen:
            seen.add(name)
            operation_names.append(name)
    return {
        "results": exec_result,
        "retrieved_indices": list(retrieved_indices),
        "operation_names": operation_names
    }


def _apply_extracted_memory_actions(trainer: BaseTrainer,
                                    memory_bank,
                                    extracted: Dict[str, Any],
                                    executor=None):
    exec_impl = executor if executor is not None else trainer.executor
    exec_impl.apply_to_memory_bank(
        results=extracted.get("results", []),
        memory_bank=memory_bank,
        retrieved_indices=extracted.get("retrieved_indices", []),
        operation_name=extracted.get("operation_names", [])
    )


def _normalize_insert_content(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text or '')).strip()


def _clone_snapshot_memory_bank(snapshot_dict: Dict[str, Any], state_encoder=None) -> MemoryBank:
    snapshot_bank = MemoryBank.from_dict(snapshot_dict)
    if state_encoder is not None:
        snapshot_bank.set_state_encoder(state_encoder)
    return snapshot_bank


def _merge_batch_extracted_actions(extracted_batch: List[Dict[str, Any]],
                                   batch_memory_size: int) -> Dict[str, Any]:
    ordered = sorted(extracted_batch, key=lambda x: x["session_idx"])

    update_by_actual: Dict[int, ExecutionResult] = {}
    delete_by_actual: Dict[int, ExecutionResult] = {}
    insert_seen = set()
    merged_inserts: List[ExecutionResult] = []
    merged_operation_names = []
    op_seen = set()

    for entry in ordered:
        extracted = entry.get("extracted", {})

        for op_name in extracted.get("operation_names", []) or []:
            name = str(op_name).strip()
            if name and name not in op_seen:
                op_seen.add(name)
                merged_operation_names.append(name)

        retrieved_indices = extracted.get("retrieved_indices", []) or []
        results = extracted.get("results", []) or []
        for result in results:
            if not getattr(result, "success", False):
                continue

            action_type = str(getattr(result, "action_type", "")).upper()
            reasoning = str(getattr(result, "reasoning", "") or "")

            if action_type == "INSERT":
                content = str(getattr(result, "memory_content", "") or "").strip()
                if not content:
                    continue
                norm_key = _normalize_insert_content(content)
                if not norm_key or norm_key in insert_seen:
                    continue
                insert_seen.add(norm_key)
                merged_inserts.append(ExecutionResult(
                    action_type="INSERT",
                    success=True,
                    memory_content=content,
                    reasoning=reasoning
                ))
                continue

            if action_type not in ("UPDATE", "DELETE"):
                continue

            try:
                rel_idx = int(getattr(result, "memory_index", -1))
            except Exception:
                print(
                    f"[Memory][SessionParallel] Dropped {action_type}: "
                    "invalid MEMORY_INDEX value."
                )
                continue
            if rel_idx < 0 or rel_idx >= len(retrieved_indices):
                print(
                    f"[Memory][SessionParallel] Dropped {action_type}: "
                    f"MEMORY_INDEX {rel_idx} out of range for retrieved_indices size {len(retrieved_indices)}."
                )
                continue

            try:
                actual_idx = int(retrieved_indices[rel_idx])
            except Exception:
                print(
                    f"[Memory][SessionParallel] Dropped {action_type}: "
                    f"retrieved_indices[{rel_idx}] is not a valid integer index."
                )
                continue
            if actual_idx < 0 or actual_idx >= batch_memory_size:
                print(
                    f"[Memory][SessionParallel] Dropped {action_type}: "
                    f"actual memory index {actual_idx} out of snapshot range [0, {batch_memory_size})."
                )
                continue

            if action_type == "DELETE":
                delete_by_actual[actual_idx] = ExecutionResult(
                    action_type="DELETE",
                    success=True,
                    memory_index=actual_idx,
                    reasoning=reasoning
                )
                if actual_idx in update_by_actual:
                    del update_by_actual[actual_idx]
            else:
                if actual_idx in delete_by_actual:
                    continue
                content = str(getattr(result, "memory_content", "") or "").strip()
                if not content:
                    continue
                update_by_actual[actual_idx] = ExecutionResult(
                    action_type="UPDATE",
                    success=True,
                    memory_index=actual_idx,
                    memory_content=content,
                    reasoning=reasoning
                )

    merged_results: List[ExecutionResult] = []
    for actual_idx in sorted(update_by_actual.keys()):
        merged_results.append(update_by_actual[actual_idx])
    for actual_idx in sorted(delete_by_actual.keys()):
        merged_results.append(delete_by_actual[actual_idx])
    merged_results.extend(merged_inserts)

    return {
        "results": merged_results,
        "retrieved_indices": list(range(batch_memory_size)),
        "operation_names": merged_operation_names
    }


def _apply_memory_extraction_step(trainer: BaseTrainer,
                                  memory_bank,
                                  session_text: str,
                                  session_embedding: np.ndarray = None,
                                  executor=None):
    extracted = _extract_memory_actions_step(
        trainer,
        memory_bank,
        session_text=session_text,
        session_embedding=session_embedding,
        executor=executor
    )
    _apply_extracted_memory_actions(
        trainer,
        memory_bank,
        extracted=extracted,
        executor=executor
    )
    memory_bank.step()


def _resolve_sample_id(trainer: BaseTrainer, conversation: Dict[str, Any], conv_idx: int) -> str:
    sample_id = None
    if hasattr(trainer, "data_processor") and trainer.data_processor is not None:
        try:
            sample_id = trainer.data_processor.get_sample_id(conversation)
        except Exception:
            sample_id = None
    if sample_id is None:
        sample_id = conversation.get('sample_id', conversation.get('index', conv_idx))
    return str(sample_id)


def _build_memory_bank_from_sessions(trainer: BaseTrainer,
                                     sessions,
                                     session_embeddings: np.ndarray = None,
                                     total: int = None,
                                     executor=None,
                                     show_progress: bool = True,
                                     session_parallel_workers: int = 1):
    memory_bank = trainer._initialize_memory_bank()

    workers = max(1, int(session_parallel_workers or 1))
    if workers == 1:
        iterator = enumerate(sessions)
        if show_progress:
            iterator = tqdm(iterator, total=total, desc="Sessions")
        for session_idx, session_text in iterator:
            session_embedding = None
            if session_embeddings is not None:
                session_embedding = session_embeddings[session_idx]
            _apply_memory_extraction_step(
                trainer,
                memory_bank,
                session_text=session_text,
                session_embedding=session_embedding,
                executor=executor
            )
        return memory_bank

    session_texts = sessions if isinstance(sessions, list) else list(sessions)
    session_records = []
    for session_idx, session_text in enumerate(session_texts):
        session_embedding = None
        if session_embeddings is not None:
            session_embedding = session_embeddings[session_idx]
        session_records.append((session_idx, session_text, session_embedding))

    total_sessions = total if total is not None else len(session_records)
    progress = tqdm(total=total_sessions, desc="Sessions") if show_progress else None

    try:
        for start in range(0, len(session_records), workers):
            batch = session_records[start:start + workers]
            if not batch:
                continue

            snapshot_dict = memory_bank.to_dict()
            snapshot_size = len(memory_bank.memories)
            local_executor_cls = executor.__class__ if executor is not None else trainer.executor.__class__

            with ThreadPoolExecutor(max_workers=min(workers, len(batch))) as pool:
                future_to_idx = {
                    pool.submit(
                        _extract_memory_actions_step,
                        trainer,
                        _clone_snapshot_memory_bank(snapshot_dict, memory_bank.state_encoder),
                        session_text,
                        session_embedding,
                        local_executor_cls(trainer.args)
                    ): session_idx
                    for session_idx, session_text, session_embedding in batch
                }

                extracted_batch = []
                for future in as_completed(future_to_idx):
                    session_idx = future_to_idx[future]
                    extracted = future.result()
                    extracted_batch.append({
                        "session_idx": session_idx,
                        "extracted": extracted
                    })

            merged = _merge_batch_extracted_actions(
                extracted_batch=extracted_batch,
                batch_memory_size=snapshot_size
            )
            _apply_extracted_memory_actions(
                trainer,
                memory_bank,
                extracted=merged,
                executor=executor
            )

            for _ in range(len(batch)):
                memory_bank.step()

            if progress is not None:
                progress.update(len(batch))
    finally:
        if progress is not None:
            progress.close()

    return memory_bank


def _save_memory_bank(memory_path: str, memory_bank: MemoryBank):
    import pickle
    payload = {"memory_bank": memory_bank.to_dict()}
    with open(memory_path, 'wb') as f:
        pickle.dump(payload, f)


def _load_memory_bank(memory_path: str) -> Optional[MemoryBank]:
    import pickle
    if not os.path.exists(memory_path):
        return None
    with open(memory_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'memory_bank' in data:
        bank_dict = data.get('memory_bank')
        if isinstance(bank_dict, dict) and 'memories' in bank_dict:
            return MemoryBank.from_dict(bank_dict)
    if isinstance(data, dict) and 'memories' in data:
        return MemoryBank.from_dict(data)
    if isinstance(data, dict) and len(data) == 1:
        only_value = next(iter(data.values()))
        if isinstance(only_value, dict) and 'memories' in only_value:
            return MemoryBank.from_dict(only_value)
    return None


def _sanitize_filename(value: str) -> str:
    sanitized = re.sub(r'[^A-Za-z0-9._-]+', '_', value)
    return sanitized.strip('_') or 'unknown'


def _memory_cache_path(
    memory_dir: str,
    args,
    sample_id: str,
    session_mode: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    action_top_k: Optional[int] = None,
    retriever: Optional[str] = None,
    model: Optional[str] = None
) -> str:
    dataset = getattr(args, 'dataset', 'data')
    if session_mode is None:
        session_mode = getattr(args, 'session_mode', 'turn-pair')
    if chunk_size is None:
        chunk_size = getattr(args, 'chunk_size', None)
    if chunk_overlap is None:
        chunk_overlap = getattr(args, 'chunk_overlap', None)
    if action_top_k is None:
        action_top_k = getattr(args, 'action_top_k', None)
    if retriever is None:
        retriever = getattr(args, 'retriever', None)
    if model is None:
        model = getattr(args, 'model', None)
    # print(model)
    name_parts = [
        "memory",
        _sanitize_filename(str(dataset)),
        f"sample_{_sanitize_filename(str(sample_id))}",
        f"mode_{_sanitize_filename(str(session_mode))}",
        f"chunk_{_sanitize_filename(str(chunk_size))}",
        f"overlap_{_sanitize_filename(str(chunk_overlap))}",
        f"topk_{_sanitize_filename(str(action_top_k))}",
        f"retriever_{_sanitize_filename(str(retriever))}",
        f"model_{_sanitize_filename(str(model).lower())}",
    ]
    if getattr(args, 'skip_load_operation_bank', False):
        name_parts.append('skipopbank')
    cache_suffix = str(getattr(args, 'memory_cache_suffix', '') or '').strip()
    if cache_suffix:
        name_parts.append(_sanitize_filename(cache_suffix))
    filename = "_".join(name_parts) + ".pkl"
    return os.path.join(memory_dir, filename)


def evaluate_text_dataset_queries(trainer: BaseTrainer,
                                  test_data,
                                  memory_banks: Dict[str, Any],
                                  args) -> Dict[str, Any]:
    eval_args = trainer.evaluator.prepare_eval_args()
    top_k_eval = getattr(trainer.config, "mem_top_k_eval", trainer.config.mem_top_k)
    is_hotpotqa = args.dataset == "hotpotqa"
    extractor = getattr(trainer.evaluator, "_extract_answer", None) if is_hotpotqa else None

    task_args = []
    meta_by_qid = {}
    next_qid = 0

    for conv_idx, conversation in enumerate(test_data):
        if args.dataset == "longmemeval":
            question_id = str(conversation.get("question_id", ""))
            if question_id.endswith("_abs"):
                print(f"[Eval] Skipping _abs sample question_id={question_id}.")
                continue
        sample_id = _resolve_sample_id(trainer, conversation, conv_idx)
        memory_bank = memory_banks.get(sample_id)
        if memory_bank is None:
            print(f"[Eval] Missing memory bank for sample_id={sample_id}, skipping.")
            continue

        qa_list = trainer.data_processor.get_qa_list(conversation)
        valid_qa = trainer.evaluator.filter_qa_list(qa_list)
        if not valid_qa:
            print(f"[Eval] No valid QA items for sample_id={sample_id}, skipping.")
            continue

        questions = [qa['question'] for _, qa in valid_qa]
        q_embeddings = get_embeddings(
            eval_args.retriever,
            questions,
            'query'
        )

        for idx, (qa_idx, qa) in enumerate(valid_qa):
            question = qa['question']
            q_embedding = q_embeddings[idx]
            retrieved_mems, retrieved_indices = memory_bank.retrieve(
                q_embedding, top_k=top_k_eval, use_state_encoder=False
            )
            # print(retrieved_mems)
            # print(top_k_eval, len(retrieved_mems))
            prompt = trainer.evaluator.build_prompt(question, retrieved_mems, qa)
            task_args.append((next_qid, prompt, eval_args))
            meta_by_qid[next_qid] = {
                "qa": qa,
                "qa_idx": qa_idx,
                "sample_id": sample_id,
                "retrieved_memories": retrieved_mems,
                "retrieved_indices": list(retrieved_indices)
            }
            next_qid += 1

    if not task_args:
        print("No evaluation queries found.")
        return {}

    ret = get_llm_response(args=eval_args, task_args=task_args)

    predictions = {}
    metrics = {
        "f1": [],
        "llm_judge": []
    }
    category_metrics = {}

    for qid, response, _, success in ret:
        meta = meta_by_qid.get(qid, {})
        qa = meta.get("qa", {})
        ground_truth = trainer.evaluator.get_ground_truth(qa)
        prediction = response.strip() if success and response is not None else ""
        if callable(extractor):
            prediction = extractor(prediction)
        # print(prediction)
        predictions[qid] = prediction

        f1 = trainer.evaluator.compute_f1(prediction, ground_truth, qa)
        metrics["f1"].append(f1)

        category = qa.get("category")
        if category is not None:
            bucket = category_metrics.setdefault(category, {"f1": [], "llm_judge": []})
            bucket["f1"].append(f1)

    # LLM judge
    judge_task_args = []
    for qid, meta in meta_by_qid.items():
        qa = meta.get("qa", {})
        ground_truth = trainer.evaluator.get_ground_truth(qa)
        if isinstance(ground_truth, list):
            ground_truth_str = ", ".join(str(ans) for ans in ground_truth)
        else:
            ground_truth_str = str(ground_truth)
        prediction = predictions.get(qid, "")
        judge_task_args.append((
            qid,
            LLM_JUDGE_GENERAL_PROMPT.format(
                question=qa.get("question", ""),
                ground_truth=ground_truth_str,
                model_answer=prediction
            ),
            eval_args
        ))

    llm_judge_scores = {}
    if judge_task_args:
        judge_scores = llm_judge(task_args=judge_task_args, args=eval_args)
        for idx, (qid, _, _) in enumerate(judge_task_args):
            llm_judge_scores[qid] = judge_scores[idx]

    for qid, score in llm_judge_scores.items():
        metrics["llm_judge"].append(float(score))
        meta = meta_by_qid.get(qid, {})
        qa = meta.get("qa", {})
        category = qa.get("category")
        if category is not None and category in category_metrics:
            category_metrics[category]["llm_judge"].append(float(score))

    def _avg(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    print("\n" + "=" * 80)
    print(f"{args.dataset} Evaluation (query-wise averages)")
    print("=" * 80)
    print(f"Total queries: {len(metrics['f1'])}")
    print(f"F1: {_avg(metrics['f1']):.4f}")
    print(f"LLM Judge: {_avg(metrics['llm_judge']):.4f}")

    if category_metrics:
        print("\nBy category:")
        for category in sorted(category_metrics.keys()):
            data = category_metrics[category]
            print(
                f"Category {category}: "
                f"F1={_avg(data['f1']):.4f}, "
                f"LLM Judge={_avg(data['llm_judge']):.4f}"
            )

    return {}


def infer_text_dataset_memories(trainer: BaseTrainer, test_data, args):
    print("\n" + "="*80)
    print("Constructing Memory Banks for Test Set")
    print("="*80)

    trainer.controller.eval()

    memory_dir = os.path.join(os.path.dirname(args.out_file), 'memories')
    os.makedirs(memory_dir, exist_ok=True)

    memory_banks: Dict[str, MemoryBank] = {}
    overwrite = bool(getattr(args, "overwrite", False))
    inference_workers = int(
        getattr(args, "inference_workers", getattr(trainer.config, "inference_workers", 1)) or 1
    )
    inference_workers = max(1, inference_workers)
    inference_session_workers = int(
        getattr(args, "inference_session_workers", getattr(trainer.config, "inference_session_workers", 1)) or 1
    )
    inference_session_workers = max(1, inference_session_workers)
    loaded_count = 0
    computed_count = 0
    pending_jobs = []

    def _compute_single_sample_memory(conversation, sample_id: str, memory_path: str, use_local_executor: bool):
        exec_impl = trainer.executor.__class__(args) if use_local_executor else trainer.executor
        sessions, episode_length, precompute = trainer._prepare_sessions(conversation)
        if precompute and isinstance(sessions, list):
            session_embeddings = trainer.state_encoder._encode_texts(sessions)
            if hasattr(session_embeddings, "ndim") and session_embeddings.ndim == 1:
                session_embeddings = session_embeddings.reshape(1, -1)
        else:
            session_embeddings = None

        total = episode_length if episode_length is not None and episode_length > 0 else None
        memory_bank = _build_memory_bank_from_sessions(
            trainer,
            sessions,
            session_embeddings=session_embeddings,
            total=total,
            executor=exec_impl,
            show_progress=True,
            session_parallel_workers=inference_session_workers
        )
        return sample_id, memory_path, memory_bank

    for conv_idx, conversation in enumerate(tqdm(test_data, desc="Processing")):
        if args.dataset == "longmemeval":
            question_id = str(conversation.get("question_id", ""))
            if question_id.endswith("_abs"):
                print(f"[Memory] Skipping _abs sample question_id={question_id}.")
                continue
        sample_id = _resolve_sample_id(trainer, conversation, conv_idx)
        memory_path = _memory_cache_path(memory_dir, args, sample_id)
        if not overwrite:
            cached_bank = _load_memory_bank(memory_path)
            if cached_bank is not None:
                memory_banks[sample_id] = cached_bank
                loaded_count += 1
                continue
        pending_jobs.append((conversation, sample_id, memory_path))

    if inference_workers == 1:
        for conversation, sample_id, memory_path in pending_jobs:
            sid, out_path, memory_bank = _compute_single_sample_memory(
                conversation=conversation,
                sample_id=sample_id,
                memory_path=memory_path,
                use_local_executor=False
            )
            memory_banks[sid] = memory_bank
            _save_memory_bank(out_path, memory_bank)
            computed_count += 1
    else:
        with ThreadPoolExecutor(max_workers=inference_workers) as pool:
            futures = [
                pool.submit(
                    _compute_single_sample_memory,
                    conversation,
                    sample_id,
                    memory_path,
                    True
                )
                for conversation, sample_id, memory_path in pending_jobs
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Computing memories"):
                sid, out_path, memory_bank = future.result()
                memory_banks[sid] = memory_bank
                _save_memory_bank(out_path, memory_bank)
                computed_count += 1

    print(
        f"\nLoaded {loaded_count} cached memory banks; "
        f"computed {computed_count} new memory banks."
    )
    print(f"Memory cache directory: {memory_dir}")
    print("Running evaluation on test queries...")
    evaluate_text_dataset_queries(trainer, test_data, memory_banks, args)
    return memory_dir


def _collect_alfworld_trajectories(train_data) -> list:
    trajectories = []
    if isinstance(train_data, dict):
        for _, games in train_data.items():
            if not isinstance(games, dict):
                continue
            for _, entry in games.items():
                if isinstance(entry, dict):
                    traj = entry.get("trajectory") or ""
                    if isinstance(traj, str) and traj.strip():
                        trajectories.append(traj.strip())
    elif isinstance(train_data, list):
        for entry in train_data:
            if isinstance(entry, dict):
                traj = entry.get("trajectory") or ""
                if isinstance(traj, str) and traj.strip():
                    trajectories.append(traj.strip())
    return trajectories


def _collect_alfworld_eval_entries(test_data) -> List[Dict[str, Any]]:
    entries = []
    if isinstance(test_data, dict):
        for task_type, games in test_data.items():
            if not isinstance(games, dict):
                continue
            for gamefile, entry in games.items():
                if isinstance(entry, dict):
                    entries.append({
                        "task_type": task_type,
                        "gamefile": gamefile,
                        "entry": entry
                    })
    elif isinstance(test_data, list):
        for entry in test_data:
            if isinstance(entry, dict):
                entries.append({
                    "task_type": entry.get("task_type", "unknown"),
                    "gamefile": entry.get("gamefile", ""),
                    "entry": entry
                })
    return entries


def _extract_alfworld_objective(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"Your task is to:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    objective = match.group(1).strip()
    if not objective:
        return ""
    objective = objective.splitlines()[0].strip()
    return objective.rstrip(".")


def infer_alfworld_memories(trainer: BaseTrainer, train_data, test_data, args):
    print("\n" + "="*80)
    print("Constructing Memory Bank from ALFWorld Train Trajectories")
    print("="*80)

    trainer.controller.eval()

    memory_dir = os.path.join(os.path.dirname(args.out_file), 'memories')
    os.makedirs(memory_dir, exist_ok=True)
    inference_session_workers = int(
        getattr(args, "inference_session_workers", getattr(trainer.config, "inference_session_workers", 1)) or 1
    )
    inference_session_workers = max(1, inference_session_workers)

    overwrite = bool(getattr(args, "overwrite", False))
    memory_path = _memory_cache_path(
        memory_dir,
        args,
        "alfworld_train",
        session_mode="alfworld",
        chunk_size=getattr(trainer.config, "alfworld_pair_chunk_size", None)
        or getattr(trainer.config, "chunk_size", None)
    )
    cached_bank = None
    if not overwrite:
        cached_bank = _load_memory_bank(memory_path)
    if cached_bank is not None:
        memory_bank = cached_bank
        print(f"Loaded cached ALFWorld memory bank from {memory_path}")
    else:
        trajectories = _collect_alfworld_trajectories(train_data)
        chunk_size = getattr(trainer.config, "alfworld_pair_chunk_size", None)
        if chunk_size is None:
            chunk_size = getattr(trainer.config, "chunk_size", None)
        chunks = chunk_trajectories_by_tokens(trajectories, chunk_size)

        if not chunks:
            print("No ALFWorld trajectories found; skipping memory extraction.")
            return memory_dir

        session_embeddings = trainer.state_encoder._encode_texts(chunks)
        if hasattr(session_embeddings, "ndim") and session_embeddings.ndim == 1:
            session_embeddings = session_embeddings.reshape(1, -1)

        memory_bank = _build_memory_bank_from_sessions(
            trainer,
            chunks,
            session_embeddings=session_embeddings,
            total=len(chunks),
            executor=trainer.executor,
            session_parallel_workers=inference_session_workers
        )
        _save_memory_bank(memory_path, memory_bank)
        print(f"Saved ALFWorld memory bank to {memory_path}")

    if not test_data:
        print("No ALFWorld eval data provided; skipping eval.")
        return memory_dir

    print("\n" + "="*80)
    print("Evaluating ALFWorld Test Environments")
    print("="*80)

    eval_entries = _collect_alfworld_eval_entries(test_data)
    if not eval_entries:
        print("No ALFWorld eval entries found; skipping eval.")
        return memory_dir

    query_source = str(getattr(trainer.config, "alfworld_eval_query_source", "first_observation")).lower()
    if query_source not in ("objective", "first_observation"):
        query_source = "first_observation"

    queries = []
    objectives = []
    for item in eval_entries:
        entry = item["entry"]
        first_obs = entry.get("first_observation") or ""
        if not isinstance(first_obs, str):
            first_obs = str(first_obs)
        objective = entry.get("objective") or ""
        if not isinstance(objective, str):
            objective = ""
        if not objective and first_obs:
            objective = _extract_alfworld_objective(first_obs)
        objectives.append(objective)
        if query_source == "objective":
            query_text = objective or first_obs
        else:
            query_text = first_obs or objective
        queries.append(query_text or "")

    retriever = getattr(trainer.args, "retriever", None) or "contriever"
    top_k_eval = getattr(trainer.config, "mem_top_k_eval", trainer.config.mem_top_k)

    query_embeddings = []
    if any(q.strip() for q in queries):
        safe_queries = [q if q.strip() else " " for q in queries]
        query_embeddings = get_embeddings(retriever, safe_queries, "query")

    tasks = []
    for idx, item in enumerate(eval_entries):
        q_text = queries[idx]
        retrieved_memories = []
        retrieved_indices = []
        if q_text.strip() and len(memory_bank.memories) > 0 and len(query_embeddings) > 0:
            emb = query_embeddings[idx]
            if hasattr(emb, "ndim") and emb.ndim == 2:
                emb = emb[0]
            retrieved_memories, retrieved_indices = memory_bank.retrieve(
                emb, top_k=top_k_eval, use_state_encoder=False
            )
        tasks.append({
            "task_type": item["task_type"],
            "gamefile": item["gamefile"],
            "objective": objectives[idx],
            "query": q_text,
            "retrieved_memories": list(retrieved_memories),
            "retrieved_indices": list(retrieved_indices)
        })

    llm_args = {
        "model": args.model,
        "api_base": args.api_base,
        "api_key": args.api_key,
        "temperature": getattr(trainer.config, "alfworld_action_temperature", 0.0),
        "top_p": getattr(trainer.config, "alfworld_action_top_p", 1.0),
        "max_tokens": getattr(trainer.config, "alfworld_action_max_tokens", 32),
        "seed": args.seed
    }
    include_inventory = bool(getattr(trainer.config, "alfworld_include_inventory", True))
    use_icl_prompt = bool(getattr(trainer.config, "alfworld_use_icl", False))
    max_steps = int(getattr(trainer.config, "alfworld_pair_max_steps", 50))
    workers = int(getattr(trainer.config, "alfworld_pair_b_workers", 0) or 0)
    if workers <= 0:
        workers = max(1, getattr(trainer.config, "batch_size", 1))

    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
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
                task_type=task.get("task_type"),
                use_icl_prompt=use_icl_prompt
            ): task for task in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="ALFWorld eval"):
            task = futures[future]
            try:
                outcome = future.result()
            except Exception as exc:
                outcome = {
                    "success": False,
                    "episode_length": max_steps,
                    "error": str(exc)
                }
            merged = dict(task)
            merged.update(outcome)
            results.append(merged)

    if not results:
        print("No ALFWorld eval results produced.")
        return memory_dir

    success_count = sum(1 for r in results if r.get("success"))
    avg_success = success_count / max(len(results), 1)
    step_counts = []
    for r in results:
        steps = r.get("episode_length")
        if not isinstance(steps, int):
            steps = max_steps
        step_counts.append(steps)
    avg_steps = float(np.mean(step_counts)) if step_counts else 0.0

    print("\n" + "=" * 80)
    print("ALFWorld Evaluation Metrics")
    print("=" * 80)
    print(f"Total envs: {len(results)}")
    print(f"Success rate: {avg_success:.4f}")
    print(f"Avg interaction steps: {avg_steps:.2f}")
    return memory_dir


def main():
    # Parse arguments
    args = get_agentic_memory_args()

    # Set seed
    set_seed(args.seed)

    # Load config
    config = AgenticMemoryConfig()
    config.update_from_args(args)

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Load dataset
    data_file = args.data_file
    if args.dataset == "alfworld" and getattr(args, "alfworld_offline_data", None):
        data_file = args.alfworld_offline_data
    print(f"Loading dataset from {data_file}")
    data = load_dataset(data_file, args.dataset)
    eval_data = None
    if args.dataset == "hotpotqa":
        eval_file = getattr(args, "hotpotqa_eval_file", "./data/eval_200.json")
        if not os.path.exists(eval_file):
            raise FileNotFoundError(f"HotpotQA eval file not found: {eval_file}")
        print(f"Loading HotpotQA eval dataset from {eval_file}")
        eval_data = load_dataset(eval_file, args.dataset)
    if args.dataset == "alfworld":
        eval_file = getattr(args, "alfworld_eval_file", "./data/alfworld_expert_eval_in_distribution.json")
        if not os.path.exists(eval_file):
            raise FileNotFoundError(f"ALFWorld eval file not found: {eval_file}")
        print(f"Loading ALFWorld eval dataset from {eval_file}")
        eval_data = load_dataset(eval_file, args.dataset)

    # Split data
    train_data, val_data, test_data = split_data(data, args.dataset)
    if args.dataset == "hotpotqa" and eval_data is not None:
        test_data = eval_data
        val_data = []
    if args.dataset == "alfworld" and eval_data is not None:
        test_data = eval_data
        val_data = {}
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Initialize trainer
    print("\nInitializing Agentic Memory Trainer...")
    trainer = get_trainer(args, config)

    def _run_inference():
        if args.dataset == "alfworld":
            return infer_alfworld_memories(trainer, train_data, test_data, args)
        return infer_text_dataset_memories(trainer, test_data, args)

    if args.eval_only:
        # Load checkpoint and infer
        if args.load_checkpoint is None:
            raise ValueError("Must specify --load-checkpoint for inference")

        print(f"Loading checkpoint from {args.load_checkpoint}")
        trainer.load_checkpoint(args.load_checkpoint)

        # Infer and save memories
        _run_inference()

    else:
        if args.load_checkpoint is not None:
            print(f"Loading checkpoint from {args.load_checkpoint}")
            trainer.load_checkpoint(args.load_checkpoint)
        # Train
        print("\nStarting training...")
        trainer.train(train_data)

        # Save final model
        trainer.save_checkpoint('final')

        # Infer on test set and save memories
        # print("\nInferring on test set and saving memories...")
        # _run_inference()

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == '__main__':
    main()
