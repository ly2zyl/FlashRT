"""FlashRT Pi0.5 frontend for Houmo M50 (XH2).

Unlike the NVIDIA backends, M50 kernels are ahead-of-time compiled HMM
graphs.  This frontend treats the HMM directory as a FlashRT device bundle
and preserves the public ``set_prompt``/``infer`` contract used by
``flash_rt.load_model``.

The bundle contains six executable graphs plus the token embedding table::

    siglip.hmm
    gemma_2b_prefill.hmm
    gemma_expert_300m_decode.hmm
    action_in_proj.hmm
    action_out_proj.hmm
    time_mlp.hmm
    embedding.pt

The host is responsible for using HMMs compiled for the installed Dadao,
driver, and firmware version.  ``tcim_lite`` owns all device allocation and
synchronisation; PyTorch is used only for CPU preprocessing and small host
side tensor operations.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from flash_rt.core.utils.actions import transform_feature
from flash_rt.core.utils.norm_stats import load_norm_stats, pi05_candidates
from flash_rt.core.utils.pi05_prompt import format_pi05_prompt

logger = logging.getLogger(__name__)


M50_REQUIRED_FILES = (
    "siglip.hmm",
    "gemma_2b_prefill.hmm",
    "gemma_expert_300m_decode.hmm",
    "action_in_proj.hmm",
    "action_out_proj.hmm",
    "time_mlp.hmm",
    "embedding.pt",
)


class M50ArtifactError(RuntimeError):
    """Raised when an M50 Pi0.5 device bundle is incomplete."""


@dataclass(frozen=True)
class M50Pi05Artifacts:
    """Resolved paths for one M50 Pi0.5 deployment bundle."""

    model_dir: Path

    @classmethod
    def resolve(
        cls,
        checkpoint_dir: Path,
        compiled_model_dir: str | os.PathLike[str] | None,
    ) -> "M50Pi05Artifacts":
        candidates: list[Path] = []
        if compiled_model_dir is not None:
            candidates.append(Path(compiled_model_dir).expanduser())
        env_dir = os.environ.get("FLASHRT_M50_MODEL_DIR")
        if env_dir:
            candidates.append(Path(env_dir).expanduser())
        candidates.extend(
            (
                checkpoint_dir / "m50_xh2",
                checkpoint_dir / "xh2",
                checkpoint_dir / "outputs" / "pi05" / "xh2",
            )
        )

        # Prefer a complete bundle, but retain the first explicit candidate so
        # the eventual error lists the exact missing files at the user's path.
        chosen = candidates[0]
        for candidate in candidates:
            if all((candidate / name).is_file() for name in M50_REQUIRED_FILES):
                chosen = candidate
                break

        missing = [name for name in M50_REQUIRED_FILES if not (chosen / name).is_file()]
        if missing:
            details = "\n".join(f"  - {chosen / name}" for name in missing)
            raise M50ArtifactError(
                "Incomplete FlashRT M50 Pi0.5 bundle. Missing:\n"
                f"{details}\n"
                "Pass compiled_model_dir=..., or set FLASHRT_M50_MODEL_DIR."
            )
        return cls(model_dir=chosen.resolve())

    def path(self, filename: str) -> Path:
        return self.model_dir / filename


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on deploy image
        raise RuntimeError(
            "FlashRT M50 Pi0.5 requires PyTorch for CPU preprocessing. "
            "Install the robot-assistant runtime requirements inside the "
            "Houmo deployment container."
        ) from exc
    return torch


def _import_tcim():
    try:
        import tcim_lite as tcim
    except ImportError as exc:  # pragma: no cover - depends on Houmo image
        raise RuntimeError(
            "FlashRT M50 backend requires tcim_lite from the Houmo Dadao "
            "deployment image. Run this process inside that container."
        ) from exc
    return tcim


def _to_numpy(tensor, *, dtype=None) -> np.ndarray:
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor.detach().numpy()


def _make_attention_masks(torch, pad_masks, att_masks):
    cumsum = torch.cumsum(att_masks, dim=1)
    causal_blocks = cumsum[:, None, :] <= cumsum[:, :, None]
    valid = pad_masks[:, None, :] * pad_masks[:, :, None]
    return causal_blocks & valid


def _mask_to_additive(torch, mask):
    return torch.where(mask[:, None, :, :], 0.0, -2.3820e38)


class _M50Pi05Runtime:
    """Low-level HMM scheduler for the Pi0.5 static graph bundle."""

    def __init__(self, artifacts: M50Pi05Artifacts, config: dict, device_id: int):
        self.torch = _import_torch()
        self.tcim = _import_tcim()
        self.device_id = int(device_id)
        self.config = config
        self.chunk_size = int(config.get("chunk_size", 50))
        self.max_action_dim = int(config.get("max_action_dim", 32))
        self.num_steps = int(config.get("num_inference_steps", 10))
        self.min_period = float(config.get("min_period", 4e-3))
        self.max_period = float(config.get("max_period", 4.0))

        get_device_num = getattr(self.tcim.runtime, "get_device_num", None)
        if callable(get_device_num):
            count = int(get_device_num("Xh2HalBackend"))
            if self.device_id < 0 or self.device_id >= count:
                raise RuntimeError(
                    f"M50 device_id={self.device_id} is unavailable; "
                    f"tcim_lite reports {count} XH2 device(s)."
                )

        # WeightManager lifetime is coupled to each loaded executable. Keep
        # every manager and option alive for the entire frontend lifetime.
        self._weight_managers: list[Any] = []
        self._options: list[Any] = []

        self.siglip = self._load(artifacts.path("siglip.hmm"))
        self.gemma_prefill = self._load(artifacts.path("gemma_2b_prefill.hmm"))
        self.gemma_decode = self._load(artifacts.path("gemma_expert_300m_decode.hmm"))
        self.action_in = self._load(artifacts.path("action_in_proj.hmm"))
        self.action_out = self._load(artifacts.path("action_out_proj.hmm"))
        self.time_mlp_graph = self._load(artifacts.path("time_mlp.hmm"))

        # The prefill graph owns the shared KV-cache allocation. Bind the
        # expert decode graph's cache inputs directly to those device buffers.
        for name in self._prefill_cache_names():
            self.gemma_decode.set_input(name, self.gemma_prefill.get_dev_input(name))

        state = self.torch.load(
            str(artifacts.path("embedding.pt")),
            map_location="cpu",
            weights_only=True,
        )
        weight = state["weight"] if isinstance(state, dict) else state
        self.embedding = self.torch.nn.Embedding(weight.shape[0], weight.shape[1])
        self.embedding.load_state_dict({"weight": weight})
        self.embedding.eval()
        self._embedding_scale = math.sqrt(int(weight.shape[1]))

        output_info = self.action_in.get_output_info("action_in_proj_out")
        self.action_embed_dim = int(output_info.shape[-1])
        self._valid_length = np.zeros((1,), dtype=np.int32)
        self._current_length = np.zeros((1,), dtype=np.int32)
        self.valid_kv_length = 0
        self._suffix_att_mask = self.torch.tensor(
            [1] + [0] * (self.chunk_size - 1), dtype=self.torch.bool
        )
        self._missing_image_embedding = None

    def _load(self, path: Path):
        manager = self.tcim.runtime.WeightManager(device=self.device_id)
        option = self.tcim.runtime.Option(manager)
        executable = self.tcim.runtime.load(str(path), option)
        self._weight_managers.append(manager)
        self._options.append(option)
        return executable

    def _prefill_cache_names(self) -> list[str]:
        names: list[str] = []
        for index in range(self.gemma_prefill.get_num_inputs()):
            name = self.gemma_prefill.get_input_name(index)
            if "model_layer" in name and "cache" in name:
                names.append(name)
        return names

    @staticmethod
    def _run(executable) -> None:
        executable.run()
        executable.sync()

    def _set_lengths(self, current: int) -> None:
        self._valid_length[0] = self.valid_kv_length
        self._current_length[0] = int(current)

    def embed_image(self, image):
        torch = self.torch
        self.siglip.set_input("pixel_values", _to_numpy(image, dtype=torch.float16))
        self._run(self.siglip)
        return torch.from_numpy(self.siglip.get_output("image_features").numpy())

    def embed_tokens(self, tokens):
        return self.embedding(tokens) * self._embedding_scale

    def prefill(self, embeddings, attention_mask) -> None:
        torch = self.torch
        embeddings = embeddings.to(torch.float16)
        attention_mask = attention_mask.to(torch.float16)
        pad = 1024 - int(attention_mask.shape[-1])
        if pad < 0:
            raise ValueError(
                f"M50 Pi0.5 prefix length {attention_mask.shape[-1]} exceeds 1024"
            )
        attention_mask = torch.nn.functional.pad(
            attention_mask, (0, pad), value=-2.3820e38
        )
        self._set_lengths(int(embeddings.shape[1]))
        self.gemma_prefill.set_input("input_1", _to_numpy(embeddings))
        self.gemma_prefill.set_input("valid_length", self._valid_length)
        self.gemma_prefill.set_input("current_length", self._current_length)
        self.gemma_prefill.set_input("attention_mask", _to_numpy(attention_mask))
        self._run(self.gemma_prefill)
        self.valid_kv_length = int(embeddings.shape[1])

    def decode(self, embeddings, attention_mask, condition):
        torch = self.torch
        embeddings = embeddings.to(torch.float16)
        attention_mask = attention_mask.to(torch.float16)
        condition = condition.to(torch.float16)
        self._set_lengths(int(embeddings.shape[1]))
        self.gemma_decode.set_input("input_1", _to_numpy(embeddings))
        self.gemma_decode.set_input("valid_length", self._valid_length)
        self.gemma_decode.set_input("current_length", self._current_length)
        self.gemma_decode.set_input("cond", _to_numpy(condition))
        self.gemma_decode.set_input("attention_mask", _to_numpy(attention_mask))
        self._run(self.gemma_decode)
        name = self.gemma_decode.get_output_name(0)
        return torch.from_numpy(self.gemma_decode.get_output(name).numpy())

    def _graph_call(self, executable, input_name: str, value, output_name: str):
        torch = self.torch
        executable.set_input(input_name, _to_numpy(value.to(torch.float16)))
        self._run(executable)
        return torch.from_numpy(executable.get_output(output_name).numpy())

    def action_in_projection(self, actions):
        return self._graph_call(
            self.action_in, "action_in", actions, "action_in_proj_out"
        )

    def action_out_projection(self, hidden):
        return self._graph_call(
            self.action_out, "action_out", hidden, "action_out_proj_out"
        )

    def time_projection(self, embedding):
        return self._graph_call(
            self.time_mlp_graph, "time_emb", embedding, "time_mlp_out"
        )

    def _time_embedding(self, timestep):
        torch = self.torch
        half = self.action_embed_dim // 2
        fraction = torch.linspace(0.0, 1.0, half, dtype=torch.float32)
        period = self.min_period * (self.max_period / self.min_period) ** fraction
        angle = timestep.to(torch.float32)[:, None] * ((2 * math.pi) / period)[None, :]
        return torch.cat((torch.sin(angle), torch.cos(angle)), dim=1)

    def _decode_mask(self, prefix_pad_masks):
        torch = self.torch
        batch = prefix_pad_masks.shape[0]
        prefix = prefix_pad_masks[:, None, :].expand(
            batch, self.chunk_size, prefix_pad_masks.shape[1]
        )
        suffix_blocks = self._suffix_att_mask[None, :].expand(batch, self.chunk_size)
        suffix_valid = torch.ones(
            batch, self.chunk_size, dtype=torch.bool
        )
        suffix = _make_attention_masks(torch, suffix_valid, suffix_blocks)
        mask = _mask_to_additive(torch, torch.cat((prefix, suffix), dim=2))
        pad = 1024 - int(mask.shape[-1])
        if pad < 0:
            raise ValueError(
                f"M50 Pi0.5 decode KV length {mask.shape[-1]} exceeds 1024"
            )
        return torch.nn.functional.pad(mask, (0, pad), value=-2.3820e38)

    def infer(self, images, image_masks, tokens, token_masks, noise=None):
        torch = self.torch
        embeddings = []
        pad_masks = []
        for image, present in zip(images, image_masks, strict=True):
            is_present = bool(present.reshape(-1)[0].item())
            if not is_present and self._missing_image_embedding is not None:
                image_embedding = self._missing_image_embedding
            else:
                image_embedding = self.embed_image(image)
                if not is_present:
                    # Every absent view uses the same all-minus-one image. Its
                    # SigLIP embedding is invariant and can be reused across
                    # cameras and action chunks.
                    self._missing_image_embedding = image_embedding
            embeddings.append(image_embedding)
            pad_masks.append(
                present[:, None].expand(image_embedding.shape[0], image_embedding.shape[1])
            )

        language = self.embed_tokens(tokens)
        embeddings.append(language)
        pad_masks.append(token_masks)
        prefix_embeddings = torch.cat(embeddings, dim=1)
        prefix_pad = torch.cat(pad_masks, dim=1)
        prefix_blocks = torch.zeros_like(prefix_pad, dtype=torch.bool)
        prefix_mask = _mask_to_additive(
            torch, _make_attention_masks(torch, prefix_pad, prefix_blocks)
        )
        decode_mask = self._decode_mask(prefix_pad)

        self.valid_kv_length = 0
        self.prefill(prefix_embeddings, prefix_mask)

        if noise is None:
            x_t = torch.normal(
                mean=0.0,
                std=1.0,
                size=(1, self.chunk_size, self.max_action_dim),
                dtype=torch.float32,
            )
        else:
            x_t = torch.as_tensor(noise, dtype=torch.float32)
            if x_t.ndim == 2:
                x_t = x_t[None, ...]
            expected = (1, self.chunk_size, self.max_action_dim)
            if tuple(x_t.shape) != expected:
                raise ValueError(f"noise shape must be {expected}, got {tuple(x_t.shape)}")

        dt = -1.0 / self.num_steps
        times = torch.linspace(
            1.0,
            1.0 + (self.num_steps - 1) * dt,
            steps=self.num_steps,
            dtype=torch.float32,
        )
        for value in times:
            action_embedding = self.action_in_projection(x_t)
            condition = self.time_projection(self._time_embedding(value.expand(1)))
            hidden = self.decode(action_embedding, decode_mask, condition)
            velocity = self.action_out_projection(hidden[:, -self.chunk_size :])
            x_t = x_t + dt * velocity.float()
        return x_t


class Pi05M50Frontend:
    """Public FlashRT frontend backed by Houmo ``tcim_lite`` and HMM graphs."""

    def __init__(
        self,
        checkpoint: str | os.PathLike[str],
        num_views: int = 2,
        *,
        compiled_model_dir: str | os.PathLike[str] | None = None,
        tokenizer_path: str | os.PathLike[str] | None = None,
        device_id: int = 0,
        hardware: str = "m50_hmm_compat",
    ):
        if hardware != "m50_hmm_compat":
            raise ValueError(
                "Pi05M50Frontend is the precompiled-HMM compatibility adapter; "
                f"expected hardware='m50_hmm_compat', got {hardware!r}"
            )
        self.checkpoint_dir = Path(checkpoint).expanduser().resolve()
        config_path = self.checkpoint_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Pi0.5 config not found: {config_path}")
        with open(config_path, encoding="utf-8") as stream:
            self.config = json.load(stream)

        self.artifacts = M50Pi05Artifacts.resolve(
            self.checkpoint_dir, compiled_model_dir
        )
        self.tokenizer_path = self._resolve_tokenizer(tokenizer_path)
        self.tokenizer = self._load_tokenizer(self.tokenizer_path)
        self.norm_stats = load_norm_stats(
            pi05_candidates(self.checkpoint_dir),
            checkpoint_dir=self.checkpoint_dir,
        )
        normalization = self.config.get("normalization_mapping", {})
        self.state_normalization = str(normalization.get("STATE", "QUANTILES"))
        self.action_normalization = str(normalization.get("ACTION", "QUANTILES"))

        self.max_state_dim = int(self.config.get("max_state_dim", 32))
        self.max_prompt_len = int(self.config.get("tokenizer_max_length", 200))
        self.output_dim = self._output_dim(self.config)
        self.expected_views = self._expected_views(self.config)
        if num_views < 1 or num_views > self.expected_views:
            raise ValueError(
                f"num_views must be in [1, {self.expected_views}], got {num_views}"
            )
        self.num_views = int(num_views)
        self.runtime = _M50Pi05Runtime(self.artifacts, self.config, device_id)
        self._tokens = None
        self._token_masks = None
        self._prompt = None
        self.latency_records: list[float] = []
        logger.info(
            "Pi0.5 M50 frontend ready (device=%d, input_views=%d, compiled_views=%d)",
            device_id,
            self.num_views,
            self.expected_views,
        )

    def _resolve_tokenizer(self, value) -> Path:
        candidates: list[Path] = []
        if value is not None:
            candidates.append(Path(value).expanduser())
        env_path = os.environ.get("FLASHRT_M50_TOKENIZER_DIR")
        if env_path:
            candidates.append(Path(env_path).expanduser())
        candidates.extend(
            (
                self.checkpoint_dir / "tokenizer",
                self.checkpoint_dir.parent / "paligemma-3b-pt-224",
            )
        )
        for path in candidates:
            if (path / "tokenizer_config.json").is_file():
                return path.resolve()
        raise FileNotFoundError(
            "PaliGemma tokenizer not found. Pass tokenizer_path=..., or set "
            "FLASHRT_M50_TOKENIZER_DIR. Tried:\n  "
            + "\n  ".join(str(path) for path in candidates)
        )

    @staticmethod
    def _load_tokenizer(path: Path):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - deploy dependency
            raise RuntimeError("transformers is required by the M50 frontend") from exc
        return AutoTokenizer.from_pretrained(str(path), local_files_only=True)

    @staticmethod
    def _output_dim(config: dict) -> int:
        feature = config.get("output_features", {}).get("action", {})
        shape = feature.get("shape", [7])
        return int(shape[0])

    @staticmethod
    def _expected_views(config: dict) -> int:
        features = config.get("input_features", {})
        count = sum(
            1
            for feature in features.values()
            if str(feature.get("type", "")).upper() == "VISUAL"
        )
        return count or 3

    def set_prompt(self, prompt_text: str, state=None) -> None:
        if state is not None:
            state = np.asarray(state, dtype=np.float32).reshape(-1)
            if "state" not in self.norm_stats:
                raise ValueError("checkpoint normalization stats do not contain state")
            state = transform_feature(
                state,
                self.norm_stats["state"],
                self.state_normalization,
            )
            if state.size > self.max_state_dim:
                raise ValueError(
                    f"Pi0.5 state has {state.size} values; max is {self.max_state_dim}"
                )
            state = np.pad(state, (0, self.max_state_dim - state.size))
        formatted = format_pi05_prompt(prompt_text, state=state)
        encoded = self.tokenizer(
            formatted,
            max_length=self.max_prompt_len,
            truncation=True,
            padding="max_length",
            padding_side="right",
            return_tensors="pt",
        )
        self._tokens = encoded["input_ids"].to(dtype=self.runtime.torch.long)
        self._token_masks = encoded["attention_mask"].to(
            dtype=self.runtime.torch.bool
        )
        self._prompt = prompt_text

    def _collect_images(self, observation: dict) -> list[np.ndarray]:
        if "images" in observation:
            values = list(observation["images"])
        else:
            values = []
            for key in ("image", "wrist_image", "wrist_image_right"):
                if key in observation:
                    values.append(observation[key])
        if not values:
            raise ValueError("M50 Pi0.5 requires at least one image")
        if len(values) > self.expected_views:
            raise ValueError(
                f"M50 Pi0.5 bundle expects at most {self.expected_views} views; "
                f"got {len(values)}"
            )
        return [np.asarray(value) for value in values]

    def _prepare_image(self, image: np.ndarray):
        torch = self.runtime.torch
        tensor = torch.from_numpy(np.ascontiguousarray(image))
        if tensor.ndim != 3:
            raise ValueError(f"image must have 3 dimensions, got {tuple(tensor.shape)}")
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        elif tensor.shape[0] != 3:
            raise ValueError(f"image must be HWC or CHW RGB, got {tuple(tensor.shape)}")
        tensor = tensor.to(torch.float32)
        if image.dtype == np.uint8 or float(tensor.max()) > 1.5:
            tensor = tensor / 255.0
        tensor = tensor[None, ...]

        height, width = 224, 224
        old_h, old_w = int(tensor.shape[-2]), int(tensor.shape[-1])
        ratio = max(old_w / width, old_h / height)
        new_h, new_w = int(old_h / ratio), int(old_w / ratio)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        top, rem_h = divmod(height - new_h, 2)
        left, rem_w = divmod(width - new_w, 2)
        tensor = torch.nn.functional.pad(
            tensor,
            (left, left + rem_w, top, top + rem_h),
            value=0.0,
        )
        return tensor * 2.0 - 1.0

    def infer(self, observation: dict, debug: bool = False) -> dict:
        if self._tokens is None:
            raise RuntimeError("set_prompt must be called before infer")
        raw_images = self._collect_images(observation)
        torch = self.runtime.torch
        images = [self._prepare_image(value) for value in raw_images]
        masks = [torch.ones(1, dtype=torch.bool) for _ in images]
        while len(images) < self.expected_views:
            images.append(torch.full((1, 3, 224, 224), -1.0, dtype=torch.float32))
            masks.append(torch.zeros(1, dtype=torch.bool))

        started = time.perf_counter()
        normalized = self.runtime.infer(
            images,
            masks,
            self._tokens,
            self._token_masks,
            noise=observation.get("noise"),
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.latency_records.append(latency_ms)
        raw_actions = normalized[0].float().cpu().numpy()
        actions = transform_feature(
            raw_actions,
            self.norm_stats["actions"],
            self.action_normalization,
            inverse=True,
        )[:, : self.output_dim]
        if debug:
            logger.info("M50 Pi0.5 latency %.3f ms", latency_ms)
        return {"actions": actions, "latency_ms": latency_ms}

    def latency_stats(self) -> dict[str, float]:
        if not self.latency_records:
            return {}
        values = np.asarray(self.latency_records, dtype=np.float64)
        return {
            "count": float(values.size),
            "mean_ms": float(values.mean()),
            "min_ms": float(values.min()),
            "max_ms": float(values.max()),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "hz": float(1000.0 / values.mean()),
        }


__all__ = [
    "M50ArtifactError",
    "M50Pi05Artifacts",
    "M50_REQUIRED_FILES",
    "Pi05M50Frontend",
]
