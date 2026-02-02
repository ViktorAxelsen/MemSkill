"""
ALFWorld interactive episode runner for LLM-based action selection.
"""
import os
import re
import sys
import threading
import uuid
from typing import Any, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import textworld
import textworld.gym

from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
from llm_utils import get_llm_response_via_api


def _unwrap_single(value, default):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        first = value[0]
        if first is None:
            return default
        if isinstance(first, (list, tuple)):
            if not first:
                return default
            nested = first[0]
            return default if nested is None else nested
        return first
    return value


def _extract_objective(text: str) -> str:
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


def _extract_admissible(info: Dict[str, Any]) -> List[str]:
    commands = info.get("admissible_commands")
    if isinstance(commands, list) and commands:
        if isinstance(commands[0], list):
            commands = commands[0]
        return [str(c) for c in commands if c]
    return []


def _parse_action_response(response: str, admissible: List[str]) -> str:
    text = (response or "").strip()
    if not text:
        return admissible[0] if admissible else "look"

    text = re.sub(r'^\s*action\s*[:=\-]\s*', '', text, flags=re.IGNORECASE)
    line = text.splitlines()[0].strip().strip('"').strip("'").strip("`")
    lowered = text.lower()

    if admissible:
        for cmd in admissible:
            if line.lower() == str(cmd).lower():
                return cmd
        for cmd in admissible:
            if str(cmd).lower() in lowered:
                return cmd
        return admissible[0]

    print("ACTION PARSE ERROR")
    return line if line else "look"


def _build_action_prompt(objective: str, retrieved_memories: List[str],
                         trajectory_text: str, inventory: str,
                         admissible: List[str],
                         expert_plan: Optional[List[str]] = None) -> str:
    lines = [
        "You are controlling a text-based ALFWorld environment.",
        "Your job: choose the NEXT action as ONE text command.",
        "Output ONLY the command string, with no extra text.",
        "You MUST choose an action from the admissible actions list and copy it EXACTLY.",
    ]

    if objective:
        lines += ["", "Goal:", objective.strip()]

    if retrieved_memories and len(retrieved_memories) > 0:
        lines += ["", "Retrieved procedural tips (optional, short & actionable):"]
        for idx, mem in enumerate(retrieved_memories):
            lines.append(f"{idx + 1}. {mem}")

    # Full trajectory is allowed since max_steps=50
    lines += ["", "Interaction history so far (most recent info matters most):"]
    lines.append(trajectory_text.strip() if trajectory_text else "(empty)")

    # Inventory might be missing/empty in some setups, so treat it as optional evidence.
    if inventory and inventory.strip() and inventory.strip().lower() not in {"none", "null", "(empty)"}:
        lines += ["", "Inventory (if available):", inventory.strip()]

    if admissible:
        lines += ["", "Admissible actions (choose exactly ONE and copy it verbatim):"]
        for cmd in admissible:
            lines.append(f"- {cmd}")

        lines += [
            "",
            "Now output exactly one line: the chosen action (must match one item above)."
        ]

    return "\n".join(lines)




def _build_trajectory_text(steps: List[Dict[str, Any]]) -> str:
    if not steps:
        return ""
    parts: List[str] = []
    first_obs = steps[0].get("observation", "")
    if first_obs:
        parts.append(str(first_obs).strip())
        parts.append("")
        parts.append("")
    for step in steps[1:]:
        action = step.get("action")
        if action:
            parts.append(f"ACTION: {action}")
        obs = step.get("observation", "")
        if obs:
            parts.append(f"OBSERVATION: {str(obs).strip()}")
    last_step = steps[-1] if steps else {}
    last_done = bool(last_step.get("done"))
    last_reward = float(last_step.get("reward") or 0.0)
    status = "SUCCESS" if (last_done and last_reward == 1.0) else "FAILED"
    parts.append(f"\n\nTRAJECTORY_STATUS: {status}")
    return "\n".join(parts).strip()


def _reset_with_timeout(env, timeout_s: float):
    reset_result: Dict[str, Any] = {}

    def _do_reset() -> None:
        try:
            reset_result["value"] = env.reset()
        except Exception as exc:
            reset_result["error"] = exc

    reset_thread = threading.Thread(target=_do_reset, daemon=True)
    reset_thread.start()
    reset_thread.join(timeout_s)
    if reset_thread.is_alive():
        raise TimeoutError(f"ALFWorld env.reset() timed out after {timeout_s:.1f}s")
    if "error" in reset_result:
        raise reset_result["error"]
    return reset_result.get("value", ("", {}))


def run_alfworld_episode(gamefile: str,
                         objective: str,
                         retrieved_memories: List[str],
                         max_steps: int,
                         llm_args: Dict[str, Any],
                         include_inventory: bool = True,
                         query_source: str = "first_observation",
                         expert_plan: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run a single ALFWorld episode with LLM action selection."""
    request_infos = textworld.EnvInfos(
        feedback=True,
        description=True,
        inventory=True,
        admissible_commands=True,
        objective=True,
        extras=["gamefile"]
    )
    wrappers = [AlfredDemangler(), AlfredInfos]
    env_id = textworld.gym.register_games(
        [gamefile],
        request_infos,
        batch_size=1,
        auto_reset=False,
        max_episode_steps=max_steps,
        asynchronous=False,
        name=f"alfworld-run-{uuid.uuid4().hex}",
        wrappers=wrappers
    )

    env = textworld.gym.make(env_id)
    steps: List[Dict[str, Any]] = []
    total_reward = 0.0
    try:
        obs_batch, info_batch = _reset_with_timeout(env, 60.0)
        obs = _unwrap_single(obs_batch, "")
        info = _unwrap_single(info_batch, {})
        if not objective:
            objective = _extract_objective(str(obs))
        query_source = (query_source or "first_observation").lower()
        if query_source not in ("objective", "first_observation"):
            query_source = "first_observation"
        query_text = objective if query_source == "objective" else str(obs or "")
        steps.append({
            "step": 0,
            "action": None,
            "observation": obs,
            "reward": 0.0,
            "done": False
        })

        trajectory_lines = [str(obs).strip()] if obs else []
        for step_idx in range(1, max_steps + 1):
            admissible = _extract_admissible(info if isinstance(info, dict) else {})
            inventory = ""
            if include_inventory and isinstance(info, dict):
                inventory = info.get("inventory") or info.get("inv") or ""
                if isinstance(inventory, list):
                    inventory = inventory[0] if inventory else ""
                if inventory is None:
                    inventory = ""
                inventory = str(inventory)
            prompt = _build_action_prompt(
                objective=objective,
                retrieved_memories=retrieved_memories or [],
                trajectory_text="\n".join(trajectory_lines),
                inventory=inventory or "",
                admissible=admissible,
                expert_plan=expert_plan or []
            )
            response, _, _ = get_llm_response_via_api(
                prompt=prompt,
                LLM_MODEL=str(llm_args.get("model", "")),
                base_url=str(llm_args.get("api_base", "")),
                api_key=llm_args.get("api_key"),
                MAX_TOKENS=int(llm_args.get("max_tokens", 32)),
                TAU=float(llm_args.get("temperature", 0.0)),
                TOP_P=float(llm_args.get("top_p", 1.0)),
                SEED=int(llm_args.get("seed", 42))
            )
            action = _parse_action_response(response, admissible)
            obs_batch, scores, dones, infos = env.step([action])

            obs = _unwrap_single(obs_batch, "")
            info = _unwrap_single(infos, {})

            reward = 0.0
            if isinstance(scores, (list, tuple)) and scores:
                reward = scores[0]
            elif isinstance(scores, (int, float)):
                reward = scores
            total_reward += float(reward)

            done = False
            if isinstance(dones, (list, tuple)) and dones:
                done = bool(dones[0])
            elif isinstance(dones, (bool, int)):
                done = bool(dones)

            steps.append({
                "step": step_idx,
                "action": action,
                "observation": obs,
                "reward": float(reward),
                "done": bool(done)
            })

            if action:
                trajectory_lines.append(f"ACTION: {action}")
            if obs:
                trajectory_lines.append(f"OBSERVATION: {str(obs).strip()}")

            if done:
                print(step_idx, done, total_reward)
                print("SUCCESS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                break

        trajectory = _build_trajectory_text(steps)
        last_step = steps[-1] if steps else {}
        success = bool(last_step.get("done")) and float(last_step.get("reward") or 0.0) == 1.0
        episode_length = max(len(steps) - 1, 0)
        return {
            "success": success,
            "total_reward": float(total_reward),
            "trajectory": trajectory,
            "objective": objective,
            "query": query_text,
            "steps": steps,
            "episode_length": episode_length,
            "gamefile": gamefile,
            "retrieved_memories": list(retrieved_memories or [])
        }
    except Exception as e:
        print("ALFWORLD BATCH B ENV EVAL FAILED: ", e)
        raise
    finally:
        try:
            env.close()
        except Exception as e:
            print(e)
            pass
