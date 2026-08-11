#!/usr/bin/env python3
"""Closed-loop LIBERO evaluation for the five aligned robot policies."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from libero_policy_profile import _load_lerobot_policy


MODEL_CHECKPOINTS = {
    "pi05": "pi05_libero_finetuned_v044",
    "smolvla_libero": "SmolVLA-LIBERO",
    "groot_libero": "GR00T-N1.7-LIBERO",
    "vla_jepa_libero": "VLA-JEPA-LIBERO",
    "molmoact2_libero": "MolmoAct2-LIBERO-LeRobot",
}
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def percentile(q: float) -> float:
        return ordered[int(q * (len(ordered) - 1))]

    return {
        "n": len(values),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def _episode_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    total_steps = sum(int(episode["steps"]) for episode in episodes)
    total_wall_seconds = sum(float(episode["wall_seconds"]) for episode in episodes)
    successes = sum(bool(episode["success"]) for episode in episodes)
    successful_steps = [
        int(episode["steps"]) for episode in episodes if episode["success"]
    ]
    timed_steps = sum(
        int(episode["observation_to_action_ms"]["n"]) for episode in episodes
    )
    inference_ms_sum = sum(
        float(episode["observation_to_action_ms"]["mean"])
        * int(episode["observation_to_action_ms"]["n"])
        for episode in episodes
    )
    return {
        "episodes": len(episodes),
        "successes": successes,
        "success_rate_percent": 100.0 * successes / len(episodes),
        "total_steps": total_steps,
        "total_wall_seconds": total_wall_seconds,
        "control_frequency_hz": total_steps / total_wall_seconds,
        "successful_episode_mean_steps": (
            statistics.fmean(successful_steps) if successful_steps else None
        ),
        "observation_to_action_mean_ms": inference_ms_sum / timed_steps,
    }


def _nvidia_smi(field: str) -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            text=True,
        ).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def _successes(info: dict[str, Any], batch_size: int) -> list[bool]:
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict):
            value = final_info.get("is_success", [False] * batch_size)
            return value.tolist() if hasattr(value, "tolist") else [bool(value)] * batch_size
        return [
            bool(item.get("is_success", False)) if isinstance(item, dict) else False
            for item in final_info
        ]
    value = info.get("is_success", [False] * batch_size)
    return value.tolist() if hasattr(value, "tolist") else [bool(value)] * batch_size


class Pi05Adapter:
    def __init__(self, checkpoint: Path):
        import flash_rt

        extension_dir = os.environ.get("FLASHRT_EXTENSION_DIR")
        if extension_dir and extension_dir not in flash_rt.__path__:
            flash_rt.__path__.append(extension_dir)
        self.policy = flash_rt.load_model(
            checkpoint,
            framework="torch",
            config="pi05",
            hardware="rtx_sm120",
            num_views=2,
            autotune=0,
            num_steps=10,
            cache_frames=1,
            use_fp8=True,
            use_fp16=False,
        )
        self.actions: deque[torch.Tensor] = deque()

    def reset(self) -> None:
        self.actions.clear()

    def select_action(self, observation: dict[str, Any]) -> torch.Tensor:
        if not self.actions:
            images = []
            for key in ("observation.images.image", "observation.images.image2"):
                image = F.interpolate(observation[key], size=(224, 224), mode="bilinear")
                image = image.mul(255).clamp(0, 255).byte()[0].permute(1, 2, 0)
                images.append(image.cpu().numpy())
            native = torch.as_tensor(
                self.policy.predict(images, prompt=observation["task"][0]),
                dtype=torch.float32,
            )
            self.actions.extend(native[:, :7])
        return self.actions.popleft().unsqueeze(0)


class LeRobotAdapter:
    def __init__(self, args: argparse.Namespace, checkpoint: Path):
        from lerobot.policies.factory import make_pre_post_processors

        load_args = argparse.Namespace(**vars(args), checkpoint=str(checkpoint))
        self.model_name = args.model
        self.policy = _load_lerobot_policy(load_args)
        overrides: dict[str, dict[str, Any]] = {
            "device_processor": {"device": "cuda"},
        }
        if args.model == "smolvla_libero":
            overrides["tokenizer_processor"] = {"tokenizer_name": args.smolvlm_base}
        elif args.model == "groot_libero":
            overrides["rename_observations_processor"] = {
                "rename_map": {
                    "observation.images.image2": "observation.images.wrist_image"
                }
            }
            overrides["groot_n1_7_vlm_encode_v1"] = {
                "model_name": args.groot_vlm_base
            }
        elif args.model == "molmoact2_libero":
            overrides["rename_observations_processor"] = {
                "rename_map": {
                    "observation.images.image2": "observation.images.wrist_image"
                }
            }
            overrides["molmoact2_pack_inputs"] = {
                "checkpoint_path": args.molmoact2_base,
                "checkpoint_revision": None,
                "checkpoint_force_download": False,
                "discrete_action_tokenizer": args.molmoact2_action_tokenizer,
            }
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            checkpoint,
            preprocessor_overrides=overrides,
        )

    def reset(self) -> None:
        self.policy.reset()

    def select_action(self, observation: dict[str, Any]) -> torch.Tensor:
        observation = dict(observation)
        if self.model_name == "vla_jepa_libero":
            for key in ("observation.images.image", "observation.images.image2"):
                observation[key] = F.interpolate(
                    observation[key], size=(224, 224), mode="bilinear"
                )
        batch = self.preprocessor(observation)
        kwargs = (
            {"inference_action_mode": "continuous"}
            if self.model_name == "molmoact2_libero"
            else {}
        )
        action = self.policy.select_action(batch, **kwargs)
        return self.postprocessor(action)


def _make_adapter(args: argparse.Namespace, checkpoint: Path):
    if args.model == "pi05":
        return Pi05Adapter(checkpoint)
    return LeRobotAdapter(args, checkpoint)


def _checkpoint_for_case(args: argparse.Namespace, suite: str) -> Path:
    checkpoint = Path(args.model_root) / MODEL_CHECKPOINTS[args.model]
    if args.model == "groot_libero":
        return checkpoint / suite
    return checkpoint


def _evaluate_task(
    args: argparse.Namespace,
    adapter: Pi05Adapter | LeRobotAdapter,
    suite: str,
    task_id: int,
) -> list[dict[str, Any]]:
    from lerobot.envs import (
        close_envs,
        make_env,
        make_env_config,
        make_env_pre_post_processors,
        preprocess_observation,
    )

    env_cfg = make_env_config(
        "libero",
        task=suite,
        task_ids=[task_id],
        observation_height=256,
        observation_width=256,
        control_mode="relative",
        max_parallel_tasks=1,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[suite][task_id]
    policy_cfg = getattr(getattr(adapter, "policy", None), "config", None)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )
    episodes = []
    try:
        for episode_index in range(args.episodes):
            seed = args.seed + episode_index
            np.random.seed(seed)
            observation, _ = env.reset(seed=seed)
            adapter.reset()
            max_steps = int(env.call("_max_episode_steps")[0])
            success = False
            failure_reason = None
            reward_sum = 0.0
            inference_ms: list[float] = []
            started = time.perf_counter()
            steps = 0
            for step in range(max_steps):
                processed = preprocess_observation(observation)
                processed["task"] = list(env.call("task_description"))
                processed = env_preprocessor(processed)
                torch.cuda.synchronize()
                infer_started = time.perf_counter()
                with torch.inference_mode():
                    action = adapter.select_action(processed)
                    action = env_postprocessor({"action": action})["action"][:, :7]
                torch.cuda.synchronize()
                inference_ms.append((time.perf_counter() - infer_started) * 1000)
                if not bool(torch.isfinite(action).all().item()):
                    failure_reason = "non_finite_action"
                    steps = step + 1
                    break
                action_numpy = action.detach().cpu().numpy()
                observation, reward, terminated, truncated, info = env.step(action_numpy)
                reward_sum += float(np.asarray(reward)[0])
                success = success or any(_successes(info, 1))
                steps = step + 1
                if bool(np.asarray(terminated)[0] or np.asarray(truncated)[0]):
                    break
            episodes.append(
                {
                    "episode": episode_index,
                    "seed": seed,
                    "success": success,
                    "failure_reason": failure_reason,
                    "steps": steps,
                    "sum_reward": reward_sum,
                    "wall_seconds": time.perf_counter() - started,
                    "observation_to_action_ms": _stats(inference_ms),
                }
            )
            print(
                f"[{args.model}] {suite} task={task_id} "
                f"episode={episode_index} seed={seed}: "
                f"{'success' if success else failure_reason or 'failed'} "
                f"steps={steps}",
                flush=True,
            )
    finally:
        close_envs(envs)
    return episodes


def _parse_task_ids(value: str) -> list[int]:
    if value == "all":
        return list(range(10))
    ids = sorted({int(item) for item in value.split(",")})
    if not ids or any(item < 0 or item > 9 for item in ids):
        raise argparse.ArgumentTypeError("task ids must be 'all' or comma-separated 0..9")
    return ids


def _parse_cases(value: str) -> list[tuple[str, int]]:
    cases: list[tuple[str, int]] = []
    for item in value.split(","):
        suite, separator, task_id_text = item.strip().partition(":")
        if not separator or suite not in SUITES:
            raise argparse.ArgumentTypeError(
                "cases must use SUITE:TASK_ID with a supported LIBERO suite"
            )
        try:
            task_id = int(task_id_text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid task id in case: {item}"
            ) from error
        if task_id < 0 or task_id > 9:
            raise argparse.ArgumentTypeError("case task ids must be in 0..9")
        case = (suite, task_id)
        if case not in cases:
            cases.append(case)
    if not cases:
        raise argparse.ArgumentTypeError("at least one case is required")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CHECKPOINTS, required=True)
    parser.add_argument("--model-root", default="/workspace/models")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suites", default=",".join(SUITES))
    parser.add_argument("--task-ids", type=_parse_task_ids, default=[0])
    parser.add_argument(
        "--cases",
        type=_parse_cases,
        help="comma-separated SUITE:TASK_ID pairs; overrides --suites/--task-ids",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smolvlm-base",
        default="/workspace/models/SmolVLM2-500M-Video-Instruct",
    )
    parser.add_argument(
        "--groot-base", default="/workspace/models/GR00T-N1.7-3B"
    )
    parser.add_argument(
        "--groot-vlm-base",
        default="/workspace/models/VLA-JEPA-deps/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument(
        "--vla-jepa-qwen-base",
        default="/workspace/models/VLA-JEPA-deps/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument(
        "--vla-jepa-encoder-base",
        default="/workspace/models/VLA-JEPA-deps/vjepa2-vitl-fpc64-256",
    )
    parser.add_argument(
        "--molmoact2-base", default="/workspace/models/MolmoAct2-LIBERO-Base"
    )
    parser.add_argument(
        "--molmoact2-action-tokenizer",
        default="/workspace/models/MolmoAct2-FAST-Tokenizer",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.cases:
        cases = args.cases
    else:
        suites = tuple(item.strip() for item in args.suites.split(",") if item.strip())
        if not suites or any(item not in SUITES for item in suites):
            parser.error(f"--suites must contain only: {', '.join(SUITES)}")
        cases = [(suite, task_id) for suite in suites for task_id in args.task_ids]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    task_results: list[dict[str, Any]] = []
    adapter: Pi05Adapter | LeRobotAdapter | None = None
    loaded_checkpoint: Path | None = None
    for suite, task_id in cases:
        checkpoint = _checkpoint_for_case(args, suite)
        if checkpoint != loaded_checkpoint:
            if adapter is not None:
                del adapter
                gc.collect()
                torch.cuda.empty_cache()
            adapter = _make_adapter(args, checkpoint)
            loaded_checkpoint = checkpoint
        assert adapter is not None
        episodes = _evaluate_task(args, adapter, suite, task_id)
        task_results.append(
            {
                "suite": suite,
                "task_id": task_id,
                "checkpoint": str(checkpoint),
                "summary": _episode_summary(episodes),
                "episodes": episodes,
            }
        )
        print(
            f"[{args.model}] {suite} task={task_id}: "
            f"{sum(ep['success'] for ep in episodes)}/{len(episodes)}",
            flush=True,
        )

    flat = [episode for task in task_results for episode in task["episodes"]]
    result = {
        "schema_version": 1,
        "evaluation_scope": "closed_loop_raw_libero_observation_to_environment_action",
        "model": args.model,
        "checkpoint": (
            {
                suite: str(_checkpoint_for_case(args, suite))
                for suite, _ in cases
            }
            if args.model == "groot_libero"
            else str(_checkpoint_for_case(args, cases[0][0]))
        ),
        "protocol": {
            "cases": [
                {"suite": suite, "task_id": task_id} for suite, task_id in cases
            ],
            "episodes_per_task": args.episodes,
            "control_mode": "relative",
            "camera_resolution": [256, 256],
            "seed_start": args.seed,
            "environment_seed_per_episode": True,
            "policy_rng_mode": "continuous_stream_seeded_once_per_model_process",
        },
        "summary": {
            **_episode_summary(flat),
            "mean_episode_seconds": statistics.fmean(
                episode["wall_seconds"] for episode in flat
            ),
        },
        "tasks": task_results,
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": _nvidia_smi("name"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Result: {args.output}", flush=True)


if __name__ == "__main__":
    main()
