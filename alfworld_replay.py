"""
Replay ALFWorld episodes using expert_plan and save trajectories.

Usage:
  python alfworld_replay.py --split train --batch-size 8 \
    --output ./data/alfworld_expert_train.json

Note: batch-size controls the number of worker threads.
"""
import argparse
import ast
import json
import os
import re
import threading
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple

from tqdm import tqdm

import textworld
import textworld.gym

from alfworld.agents.environment.alfred_tw_env import (
    AlfredDemangler,
    AlfredExpert,
    AlfredExpertType,
    AlfredInfos,
)
from alfworld.info import ALFWORLD_DATA


TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}


def _data_path_from_split(split: str) -> str:
    if split == "train":
        return os.path.join(ALFWORLD_DATA, "json_2.1.1", "train")
    if split == "eval_in_distribution":
        return os.path.join(ALFWORLD_DATA, "json_2.1.1", "valid_seen")
    if split == "eval_out_of_distribution":
        return os.path.join(ALFWORLD_DATA, "json_2.1.1", "valid_unseen")
    raise ValueError(f"Unknown split: {split}")


def _collect_game_files(split: str) -> Dict[str, List[str]]:
    data_path = _data_path_from_split(split)

    game_files_by_type: Dict[str, List[str]] = {}

    for root, _, files in os.walk(data_path):
        if "traj_data.json" not in files:
            continue
        if "movable" in root or "Sliced" in root:
            continue

        json_path = os.path.join(root, "traj_data.json")
        game_path = os.path.join(root, "game.tw-pddl")

        try:
            with open(json_path, "r") as f:
                traj_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if traj_data.get("task_type") not in TASK_TYPES.values():
            continue

        if not os.path.exists(game_path):
            continue

        try:
            with open(game_path, "r") as f:
                gamedata = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if not gamedata.get("solvable", False):
            continue

        game_files_by_type.setdefault(traj_data.get("task_type"), []).append(game_path)

    return game_files_by_type


def _extract_expert_plan(info: Any) -> List[str]:
    if isinstance(info, list) and info:
        info = info[0]
    if isinstance(info, dict):
        value = info.get("extra.expert_plan")
        if value is None:
            value = info.get("expert_plan")
        if isinstance(value, (list, tuple)) and value:
            if len(value) == 1 and isinstance(value[0], str):
                parsed = _parse_plan_string(value[0])
                if parsed:
                    return parsed
            items: List[str] = []
            for item in value:
                if isinstance(item, str):
                    parsed = _parse_plan_string(item)
                    if parsed:
                        items.extend(parsed)
                    elif item.strip():
                        items.append(item.strip())
                elif isinstance(item, (list, tuple)):
                    items.extend([str(entry) for entry in item if entry])
                elif item is not None:
                    items.append(str(item))
            if items:
                return items
        if isinstance(value, str) and value.strip():
            parsed = _parse_plan_string(value)
            return parsed if parsed else [value.strip()]
    return []


def _parse_plan_string(value: str, depth: int = 2) -> Optional[List[str]]:
    if depth <= 0:
        return None
    text = value.strip()
    if not text:
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        inner = text[1:-1].strip()
        nested = _parse_plan_string(inner, depth - 1)
        if nested:
            return nested
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed if item]
        if isinstance(parsed, str) and parsed.strip():
            nested = _parse_plan_string(parsed, depth - 1)
            return nested if nested else [parsed.strip()]
    return None


def _ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _build_step(action: Optional[str], observation: str, reward: float, done: bool,
                info: Any, step_idx: int) -> Dict[str, Any]:
    return {
        "step": step_idx,
        "action": action,
        "observation": observation,
        "reward": reward,
        "done": done,
        "expert_plan": _extract_expert_plan(info),
    }

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


def _save_output(path: str, payload: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _run_single_game(task_type: str,
                     gamefile: str,
                     max_steps: int) -> Tuple[str, str, Dict[str, Any]]:
    request_infos = textworld.EnvInfos(
        feedback=True,
        description=True,
        inventory=True,
        command_templates=True,
        intermediate_reward=True,
        location=True,
        objective=True,
        admissible_commands=True,
        extras=["gamefile", "expert_plan"]
    )
    wrappers = [
        AlfredDemangler(),
        AlfredInfos,
        AlfredExpert(expert_type=AlfredExpertType.PLANNER),
    ]
    env_id = textworld.gym.register_games(
        [gamefile],
        request_infos,
        batch_size=1,
        auto_reset=False,
        max_episode_steps=max_steps,
        asynchronous=False,
        name=f"alfworld-replay-{uuid.uuid4().hex}",
        wrappers=wrappers
    )
    env = textworld.gym.make(env_id)
    error: Optional[str] = None
    try:
        reset_result: Dict[str, Any] = {}

        def _do_reset() -> None:
            try:
                reset_result["value"] = env.reset()
            except Exception as exc:
                reset_result["error"] = exc

        reset_thread = threading.Thread(target=_do_reset, daemon=True)
        reset_thread.start()
        reset_thread.join(600.0)
        if reset_thread.is_alive():
            print("Timeout. Skipping...")
            return task_type, gamefile, {}
        if "error" in reset_result:
            raise reset_result["error"]
        obs_batch, info_batch = reset_result.get("value", ("", {}))
        # print(obs_batch)
        obs_list = _ensure_list(obs_batch)
        info_list = _ensure_list(info_batch)
        obs = obs_list[0] if obs_list else ""
        info = info_list[0] if info_list else {}
        plan = _extract_expert_plan(info)
        # print("expert plan: ", plan)
        if not plan:
            error = "Missing expert_plan at reset"

        steps: List[Dict[str, Any]] = []
        steps.append(_build_step(None, str(obs or ""), 0.0, False, info, 0))
        total_reward = 0.0
        step_ptr = 0
        done = False

        while not done and error is None:
            if step_ptr >= len(plan):
                error = f"Expert plan exhausted at step {step_ptr}"
                break
            action = str(plan[step_ptr])
            # print(action)
            obs_batch, score_batch, done_batch, info_batch = env.step([action])
            # print(info_batch)
            obs_list = _ensure_list(obs_batch)
            score_list = _ensure_list(score_batch)
            done_list = _ensure_list(done_batch)
            info_list = _ensure_list(info_batch)
            # print(info_list)
            obs = obs_list[0] if obs_list else ""
            score = score_list[0] if score_list else 0.0
            done_flag = bool(done_list[0]) if done_list else False
            info = info_list[0] if info_list else {}
            reward = float(score or 0.0)
            total_reward += reward
            step_ptr += 1
            steps.append(
                _build_step(action, str(obs or ""), reward, done_flag, info, step_ptr)
            )
            if done_flag or step_ptr >= max_steps:
                done = True

        if not done or total_reward != 1.0:
            print("Failed Trajectory!!!")

        traj_path = os.path.join(os.path.dirname(gamefile), "traj_data.json")
        first_observation = str(steps[0].get("observation") or "") if steps else ""
        objective = _extract_objective(first_observation)
        result = {
            "traj_path": traj_path,
            "first_observation": first_observation,
            "objective": objective,
            "total_reward": total_reward,
            "steps": steps,
            "trajectory": _build_trajectory_text(steps),
        }
        if error:
            result["error"] = error
        return task_type, gamefile, result
    finally:
        try:
            env.close()
        except Exception:
            pass


'''
python alfworld_replay.py
'''
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="eval_out_of_distribution",
                        choices=["train", "eval_in_distribution", "eval_out_of_distribution"])
    parser.add_argument("--batch-size", type=int, default=32, help="Worker thread count")
    parser.add_argument("--output", type=str, default="./data/alfworld_expert_eval_out_of_distribution.json")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save after every N completed games")
    args = parser.parse_args()

    game_files_by_type = _collect_game_files(args.split)
    total_games = sum(len(files) for files in game_files_by_type.values())
    print(total_games)

    max_steps = 150

    output: Dict[str, Dict[str, Any]] = {
        task_type: {} for task_type in game_files_by_type.keys()
    }
    progress = tqdm(total=total_games, desc="Collecting trajectories")
    completed_since_save = 0

    all_games: List[Tuple[str, str]] = []
    for task_type, games in game_files_by_type.items():
        for game in games:
            all_games.append((task_type, game))

    max_workers = max(1, int(args.batch_size))

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_single_game, task_type, game, max_steps): (task_type, game)
                for task_type, game in all_games
            }
            for future in as_completed(futures):
                task_type, game = futures[future]
                try:
                    task_type_res, game_res, result = future.result()
                except Exception as exc:
                    print(f"Task type {task_type} failed with exception {exc}")
                    traj_path = os.path.join(os.path.dirname(game), "traj_data.json")
                    result = {"traj_path": traj_path, "error": repr(exc)}
                    task_type_res, game_res = task_type, game
                output.setdefault(task_type_res, {})[game_res] = result
                progress.update(1)
                completed_since_save += 1
                if args.save_every > 0 and completed_since_save >= args.save_every:
                    _save_output(args.output, output)
                    completed_since_save = 0
    finally:
        _save_output(args.output, output)
        progress.close()


if __name__ == "__main__":
    main()
