#!/usr/bin/env python3
"""Aligned hot-path benchmark for five LIBERO robot policies."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


MODEL_INFO = {
    "pi05": {
        "architecture": "pi05_flow_matching",
        "revision": "local_flashrt_checkpoint",
    },
    "smolvla_libero": {
        "architecture": "smolvla_flow_matching",
        "revision": "31d453f7edd78c839a8bbc39744a292686daf0de",
    },
    "groot_libero": {
        "architecture": "groot_n1_7_flow_matching",
        "revision": "32a6ec786d6509df31b40392b4e4dcdda78c0f11",
    },
    "vla_jepa_libero": {
        "architecture": "vla_jepa_world_model_policy",
        "revision": "735d9f692981e286ade093b5046627eda876e5d0",
    },
    "molmoact2_libero": {
        "architecture": "molmoact2_continuous_flow_matching",
        "revision": "f0c5a9567c2b72faadec901e16e055c8b098c2f5",
    },
}


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def _nvidia_smi(field: str) -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            text=True,
        ).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def _checkpoint_size(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
        and not item.is_symlink()
        and ".cache" not in item.relative_to(path).parts
    )


def _canonical_images(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.rand((1, 3, 224, 224), generator=generator),
        torch.rand((1, 3, 224, 224), generator=generator),
    )


def _image_pair_sha256(image_a: torch.Tensor, image_b: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(image_a.contiguous().numpy().tobytes())
    digest.update(image_b.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _align_first_libero_action(native: torch.Tensor) -> torch.Tensor:
    if native.ndim == 2:
        native = native.unsqueeze(0)
    if native.ndim != 3:
        raise ValueError(f"expected an action chunk, got shape {tuple(native.shape)}")
    if native.shape[-1] < 7:
        raise ValueError(f"expected at least 7 action dimensions, got {native.shape[-1]}")
    return native[:, 0, :7].float()


def _load_pi05_case(args, image_a: torch.Tensor, image_b: torch.Tensor):
    import flash_rt

    extension_dir = os.environ.get("FLASHRT_EXTENSION_DIR")
    if extension_dir and extension_dir not in flash_rt.__path__:
        flash_rt.__path__.append(extension_dir)
    policy = flash_rt.load_model(
        args.checkpoint,
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
    native_holder: dict[str, Any] = {}

    def infer() -> torch.Tensor:
        images = [
            image.mul(255).byte().permute(0, 2, 3, 1).numpy()[0]
            for image in (image_a, image_b)
        ]
        native = policy.predict(images, prompt=args.task)
        native = torch.as_tensor(native)
        native_holder["shape"] = list(native.shape)
        native_holder["dtype"] = str(native.dtype)
        return _align_first_libero_action(native)

    return infer, native_holder, None, {
        "native_image_shape": [3, 224, 224],
        "state_dimension": 0,
        "preprocessing_in_timed_region": True,
        "denoising_steps": 10,
        "native_action_chunk_size": 10,
        "native_action_dimension": 7,
        "api_boundary": "FlashRTPi05.predict",
    }


def _load_lerobot_policy(args):
    checkpoint = Path(args.checkpoint)
    if args.model == "smolvla_libero":
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        config = SmolVLAConfig.from_pretrained(checkpoint, local_files_only=True)
        config.vlm_model_name = args.smolvlm_base
        policy = SmolVLAPolicy.from_pretrained(
            checkpoint, config=config, local_files_only=True, strict=False
        )
    elif args.model == "groot_libero":
        from lerobot.policies.groot.configuration_groot import GrootConfig
        from lerobot.policies.groot.modeling_groot import GrootPolicy

        config = GrootConfig.from_pretrained(checkpoint, local_files_only=True)
        config.base_model_path = args.groot_base
        policy = GrootPolicy.from_pretrained(
            checkpoint, config=config, local_files_only=True, strict=False
        )
    elif args.model == "vla_jepa_libero":
        from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
        from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

        config = VLAJEPAConfig.from_pretrained(checkpoint, local_files_only=True)
        config.qwen_model_name = args.vla_jepa_qwen_base
        config.jepa_encoder_name = args.vla_jepa_encoder_base
        policy = VLAJEPAPolicy.from_pretrained(
            checkpoint, config=config, local_files_only=True, strict=False
        )
    else:
        from lerobot.policies.molmoact2.configuration_molmoact2 import (
            MolmoAct2Config,
        )
        from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

        config = MolmoAct2Config.from_pretrained(checkpoint, local_files_only=True)
        config.checkpoint_path = args.molmoact2_base
        config.checkpoint_revision = None
        config.checkpoint_force_download = False
        config.inference_action_mode = "continuous"
        policy = MolmoAct2Policy.from_pretrained(
            checkpoint, config=config, local_files_only=True, strict=False
        )
    return policy.cuda().eval()


def _make_lerobot_case(
    args,
    policy,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    state: torch.Tensor | None = None,
):
    native_holder: dict[str, Any] = {}
    state_dim = policy.config.robot_state_feature.shape[0]
    if state is None:
        state = torch.zeros((1, state_dim))
    elif state.ndim != 2 or state.shape[0] != 1 or state.shape[1] < state_dim:
        raise ValueError(
            f"expected at least a 1x{state_dim} robot state, got {tuple(state.shape)}"
        )
    else:
        state = state[:, :state_dim]

    if args.model == "smolvla_libero":
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )

        tokenizer = policy.model.vlm_with_expert.processor.tokenizer
        noise = torch.randn(
            (1, policy.config.chunk_size, policy.config.max_action_dim),
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            device="cuda",
        )

        def native_infer() -> torch.Tensor:
            resized_a = F.interpolate(image_a, size=(256, 256), mode="bilinear")
            resized_b = F.interpolate(image_b, size=(256, 256), mode="bilinear")
            tokens = tokenizer(
                args.task,
                padding="max_length",
                max_length=policy.config.tokenizer_max_length,
                truncation=True,
                return_tensors="pt",
            )
            batch: dict[str, Any] = {
                "observation.images.camera1": resized_a.cuda(),
                "observation.images.camera2": resized_b.cuda(),
                "observation.state": state.cuda(),
                OBS_LANGUAGE_TOKENS: tokens["input_ids"].cuda(),
                OBS_LANGUAGE_ATTENTION_MASK: tokens["attention_mask"].bool().cuda(),
            }
            return policy.predict_action_chunk(batch, noise=noise)

        timed_preprocessing = True
        native_resolution = 256
    elif args.model == "groot_libero":
        from lerobot.policies.factory import make_pre_post_processors

        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            args.checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
                "rename_observations_processor": {
                    "rename_map": {
                        "observation.images.image2": "observation.images.wrist_image"
                    }
                },
                "groot_n1_7_vlm_encode_v1": {
                    "model_name": args.groot_vlm_base
                },
            },
        )

        def native_infer() -> torch.Tensor:
            resized_a = F.interpolate(image_a, size=(256, 256), mode="bilinear")
            resized_b = F.interpolate(image_b, size=(256, 256), mode="bilinear")
            raw_batch = {
                "observation.images.image": resized_a[0],
                "observation.images.image2": resized_b[0],
                "observation.state": state[0],
                "task": args.task,
            }
            batch = preprocessor(raw_batch)
            return postprocessor(policy.predict_action_chunk(batch))

        timed_preprocessing = True
        native_resolution = 256
    elif args.model == "vla_jepa_libero":
        def native_infer() -> torch.Tensor:
            batch = {
                "observation.images.image": image_a.cuda(),
                "observation.images.image2": image_b.cuda(),
                "observation.state": state.cuda(),
                "task": [args.task],
            }
            return policy.predict_action_chunk(batch)

        timed_preprocessing = True
        native_resolution = 224
    else:
        from lerobot.policies.factory import make_pre_post_processors

        preprocessor, _ = make_pre_post_processors(
            policy.config,
            args.checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
                "molmoact2_pack_inputs": {
                    "checkpoint_path": args.molmoact2_base,
                    "checkpoint_revision": None,
                    "checkpoint_force_download": False,
                    "discrete_action_tokenizer": args.molmoact2_action_tokenizer,
                },
            },
        )

        def native_infer() -> torch.Tensor:
            resized_a = F.interpolate(image_a, size=(256, 256), mode="bilinear")
            resized_b = F.interpolate(image_b, size=(256, 256), mode="bilinear")
            raw_batch = {
                "observation.images.image": resized_a[0],
                "observation.images.wrist_image": resized_b[0],
                "observation.state": state[0],
                "task": args.task,
            }
            batch = preprocessor(raw_batch)
            return policy.predict_action_chunk(
                batch, inference_action_mode="continuous"
            )

        timed_preprocessing = True
        native_resolution = 256

    def infer() -> torch.Tensor:
        native = native_infer()
        native_holder["shape"] = list(native.shape)
        native_holder["dtype"] = str(native.dtype)
        return _align_first_libero_action(native)

    settings = {
        "native_image_shape": [3, native_resolution, native_resolution],
        "state_dimension": state_dim,
        "preprocessing_in_timed_region": timed_preprocessing,
        "denoising_steps": getattr(
            policy.config,
            "num_steps",
            getattr(
                policy.config,
                "num_denoising_steps",
                getattr(
                    policy.config,
                    "num_flow_timesteps",
                    getattr(policy.config, "num_inference_timesteps", None),
                ),
            ),
        ),
        "native_action_chunk_size": policy.config.chunk_size,
        "native_action_dimension": policy.config.action_feature.shape[0],
        "api_boundary": f"{type(policy).__name__}.predict_action_chunk",
    }
    return infer, native_holder, policy, settings


def _load_libero_sample(
    suite: str,
    task_id: int,
    seed: int,
    policy_cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Load one real, policy-aligned initial observation from LIBERO."""
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
    try:
        observation, _ = env.reset(seed=seed)
        processed = preprocess_observation(observation)
        env_preprocessor, _ = make_env_pre_post_processors(
            env_cfg=env_cfg,
            policy_cfg=policy_cfg,
        )
        processed = env_preprocessor(processed)
        task = str(env.call("task_description")[0])
        return (
            processed["observation.images.image"].cpu(),
            processed["observation.images.image2"].cpu(),
            processed["observation.state"].cpu(),
            task,
        )
    finally:
        close_envs(envs)


def _profile_range(name: str, fn: Callable[[], Any], iters: int) -> Any:
    if iters <= 0:
        return None
    cudart = ctypes.CDLL("libcudart.so")
    rc = cudart.cudaProfilerStart()
    if rc != 0:
        raise RuntimeError(f"cudaProfilerStart failed with code {rc}")
    output = None
    try:
        torch.cuda.nvtx.range_push(f"libero_aligned::{name}::replan")
        for _ in range(iters):
            output = fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    finally:
        rc = cudart.cudaProfilerStop()
        if rc != 0:
            raise RuntimeError(f"cudaProfilerStop failed with code {rc}")
    return output


def _measure(fn: Callable[[], Any], iters: int) -> tuple[Any, dict, dict]:
    wall_ms: list[float] = []
    cuda_ms: list[float] = []
    output = None
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start.record()
        output = fn()
        end.record()
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - t0) * 1000.0)
        cuda_ms.append(start.elapsed_time(end))
    return output, _stats(wall_ms), _stats(cuda_ms)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure five LIBERO policies with a shared first-7D-action output."
    )
    parser.add_argument("--model", choices=sorted(MODEL_INFO), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="pick up the red block")
    parser.add_argument(
        "--libero-suite",
        choices=("libero_spatial", "libero_object", "libero_goal", "libero_10"),
        help="use a real initial observation from this LIBERO suite",
    )
    parser.add_argument("--libero-task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--profile-iters", type=int, default=0, help=argparse.SUPPRESS)
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

    for name in ("warmup", "iters", "profile_iters"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not 0 <= args.libero_task_id <= 9:
        parser.error("--libero-task-id must be in 0..9")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    load_t0 = time.perf_counter()
    if args.model == "pi05":
        if args.libero_suite:
            image_a, image_b, state, args.task = _load_libero_sample(
                args.libero_suite, args.libero_task_id, args.seed, None
            )
        else:
            image_a, image_b = _canonical_images(args.seed)
            state = None
        infer, native_holder, policy, settings = _load_pi05_case(
            args, image_a, image_b
        )
    else:
        policy = _load_lerobot_policy(args)
        if args.libero_suite:
            image_a, image_b, state, args.task = _load_libero_sample(
                args.libero_suite, args.libero_task_id, args.seed, policy.config
            )
        else:
            image_a, image_b = _canonical_images(args.seed)
            state = None
        infer, native_holder, policy, settings = _make_lerobot_case(
            args, policy, image_a, image_b, state
        )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_t0

    first_t0 = time.perf_counter()
    output = infer()
    torch.cuda.synchronize()
    first_seconds = time.perf_counter() - first_t0
    for _ in range(args.warmup):
        output = infer()
    torch.cuda.synchronize()

    profile_output = _profile_range(args.model, infer, args.profile_iters)
    if profile_output is not None:
        output = profile_output
    measured_output, wall, cuda = _measure(infer, args.iters)
    if measured_output is not None:
        output = measured_output

    parameter_dtypes: dict[str, int] = {}
    parameters = None
    if policy is not None:
        parameters = sum(parameter.numel() for parameter in policy.parameters())
        for parameter in policy.parameters():
            dtype = str(parameter.dtype)
            parameter_dtypes[dtype] = parameter_dtypes.get(dtype, 0) + parameter.numel()
    p50_ms = float(wall["p50"]) if wall["n"] else None
    info = MODEL_INFO[args.model]
    result = {
        "schema_version": 2,
        "model": args.model,
        "architecture": info["architecture"],
        "task_family": "LIBERO",
        "comparison_scope": "aligned_two_view_replan_to_first_7d_action",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_revision": info["revision"],
        "checkpoint_size_bytes": _checkpoint_size(checkpoint),
        "parameters": parameters,
        "parameter_dtypes": parameter_dtypes,
        "input": {
            "source": "libero_initial_observation" if args.libero_suite else "canonical_random",
            "libero_suite": args.libero_suite,
            "libero_task_id": args.libero_task_id if args.libero_suite else None,
            "canonical_camera_count": 2,
            "canonical_raw_image_shape": list(image_a.shape[1:]),
            "canonical_image_pair_sha256": _image_pair_sha256(image_a, image_b),
            "task": args.task,
            "seed": args.seed,
            **settings,
        },
        "native_output": native_holder,
        "aligned_output_definition": "batch x first action x first 7 LIBERO dimensions",
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "driver": _nvidia_smi("driver_version"),
            "power_limit_w": _nvidia_smi("power.limit"),
        },
        "cold": {
            "load_and_prepare_seconds": load_seconds,
            "first_infer_seconds": first_seconds,
        },
        "steady_state": {
            "warmup": args.warmup,
            "wall_ms": wall,
            "cuda_event_ms": cuda,
            "replans_per_second_p50": 1000.0 / p50_ms if p50_ms else None,
        },
        "memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "output": {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": bool(torch.isfinite(output).all().item()),
            "numel": output.numel(),
        },
        "profiler_range_iterations": args.profile_iters,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
