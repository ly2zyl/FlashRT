"""FlashRT-native Qwen inference on M50 through ``tcim_lite``.

No llama.cpp/HLIELLama symbols are imported or loaded.  FlashRT owns prompt
processing, embedding lookup, prefill/decode scheduling, KV-cache binding,
sampling, and text decoding.  TCIM only executes the compiled HMM graphs.
"""

from __future__ import annotations

import hashlib
import os
import struct
import time
from pathlib import Path

import numpy as np

from .gguf_assets import GGUFAssetIndex


_TOKENIZER_ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
)


class QwenM50Frontend:
    """Direct FlashRT -> TCIM -> M50 Qwen runtime for Houmo GGUF assets."""

    def __init__(self, checkpoint, *, device_id=0, max_tokens=512,
                 temp=0.0, top_k=0, top_p=0.0, seed=1,
                 tokenizer_cache_dir=None, zero_kv_on_reset=False, **_unused):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        if int(device_id) < 0:
            raise ValueError("device_id must be non-negative")
        if float(temp) < 0.0:
            raise ValueError("temp must be non-negative")
        if int(top_k) < 0:
            raise ValueError("top_k must be non-negative")
        if not 0.0 <= float(top_p) <= 1.0:
            raise ValueError("top_p must be in the range [0, 1]")
        self.model_path = Path(checkpoint).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Houmo Qwen GGUF not found: {checkpoint}")

        self.max_tokens = int(max_tokens)
        self.temp = float(temp)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self._rng = np.random.default_rng(int(seed))
        self.zero_kv_on_reset = bool(zero_kv_on_reset)
        self._index = GGUFAssetIndex(self.model_path)
        if not self._index.metadata.get("is_hmm", False):
            raise ValueError("GGUF is not marked as a Houmo HMM container")

        self._tokenizer_dir = self._prepare_tokenizer(tokenizer_cache_dir)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._tokenizer_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        self._embedding = self._open_embedding()
        self.vocab_size, self.hidden_size = self._embedding.shape
        self._sampling_logits = np.empty(self.vocab_size, dtype=np.float32)

        try:
            import tcim_lite as tcim
        except ImportError as exc:
            raise RuntimeError(
                "FlashRT M50 Qwen requires the Houmo tcim_lite runtime") from exc
        self._tcim = tcim
        if tcim.runtime.get_device_num("Xh2HalBackend") <= int(device_id):
            raise RuntimeError(f"M50 device {device_id} is not available")

        weight_manager = tcim.runtime.WeightManager(int(device_id))
        prefill_asset = self._index.require("prefill.hmm")
        prefill_option = tcim.runtime.Option(weight_manager)
        prefill_option.set_model_offset(prefill_asset.offset, prefill_asset.size)
        self._prefill = tcim.runtime.load(
            str(self.model_path), option=prefill_option)

        self._prefill_input_names = tuple(
            self._prefill.get_input_name(i)
            for i in range(self._prefill.get_num_inputs()))
        self._prefill_output_name = self._prefill.get_output_name(0)
        cache_names = [
            name for name in self._prefill_input_names
            if "model_layers_" in name
            and ("kcache_input" in name or "vcache_input" in name)
        ]
        if not cache_names:
            raise RuntimeError("prefill HMM exposes no KV-cache inputs")

        decode_asset = self._index.require("decoder.hmm")
        decode_option = tcim.runtime.Option(weight_manager)
        decode_option.set_model_offset(decode_asset.offset, decode_asset.size)
        decode_option.set_dummy_tensors(cache_names)
        self._decode = tcim.runtime.load(
            str(self.model_path), option=decode_option)
        self._decode_input_names = tuple(
            self._decode.get_input_name(i)
            for i in range(self._decode.get_num_inputs()))
        self._decode_output_name = self._decode.get_output_name(0)

        self._cache_tensors = []
        for name in cache_names:
            # The prefill module owns the allocated KV cache.  Decode marks
            # these inputs as dummy tensors and borrows the same device
            # buffers, which is the ownership direction required by TCIM.
            cache = self._prefill.get_dev_input(name)
            self._decode.set_input(name, cache)
            self._cache_tensors.append(cache)

        prefill_info = self._prefill.get_input_info(
            self._prefill_input_names[0])
        self.prefill_length = int(prefill_info.shape[1])
        if int(prefill_info.shape[2]) != self.hidden_size:
            raise RuntimeError("embedding width does not match prefill HMM")
        self.context_length = int(
            self._prefill.get_input_info(cache_names[0]).shape[2])
        self._prefill_embeddings = np.zeros(
            (1, self.prefill_length, self.hidden_size), dtype=np.float16)
        self._prefill_valid_length = np.zeros(1, dtype=np.int32)
        self._prefill_current_length = np.zeros(1, dtype=np.int32)
        self._decode_embedding = np.empty(
            (1, 1, self.hidden_size), dtype=np.float16)
        self._decode_valid_length = np.zeros(1, dtype=np.int32)
        self._decode_current_length = np.ones(1, dtype=np.int32)
        self.eos_token_ids = set()
        for value in (
                self.tokenizer.eos_token_id,
                getattr(self.tokenizer, "pad_token_id", None)):
            if value is not None:
                self.eos_token_ids.add(int(value))

        self._logits = None
        self._generated_ids = []
        self._valid_length = 0
        self._session_started = False
        self._finished = False
        self._closed = False
        self.last_prefill_ms = 0.0
        self.last_decode_ms = 0.0

    def _prepare_tokenizer(self, cache_dir):
        stat = self.model_path.stat()
        key = hashlib.sha256(
            f"{self.model_path}:{stat.st_size}:{stat.st_mtime_ns}".encode()
        ).hexdigest()[:16]
        if cache_dir is None:
            repo_dir = Path(__file__).resolve().parents[3]
            cache_dir = repo_dir / ".cache" / "m50" / "tokenizer"
        target = Path(cache_dir).resolve() / key
        target.mkdir(parents=True, exist_ok=True)
        for name in _TOKENIZER_ASSETS:
            if name not in self._index.assets:
                continue
            output = target / name
            if (output.exists() and
                    output.stat().st_size == self._index.assets[name].size):
                continue
            temporary = output.with_suffix(
                output.suffix + f".{os.getpid()}.tmp")
            temporary.write_bytes(self._index.read_bytes(name))
            os.replace(temporary, output)
        if not (target / "tokenizer.json").is_file():
            raise ValueError("GGUF contains no tokenizer.json")
        return str(target)

    def _open_embedding(self):
        asset = self._index.require("quant_embedding.bin")
        with self.model_path.open("rb") as handle:
            handle.seek(asset.offset)
            dimensions = struct.unpack("<I", handle.read(4))[0]
            if dimensions != 2:
                raise ValueError("Qwen embedding must be a 2-D tensor")
            shape = struct.unpack("<II", handle.read(8))
            dtype_code = struct.unpack("<I", handle.read(4))[0]
        if dtype_code != 1:
            raise ValueError("M50 native Qwen currently requires FP16 embedding")
        expected = 16 + int(np.prod(shape)) * np.dtype("<f2").itemsize
        if expected != asset.size:
            raise ValueError("invalid quant_embedding.bin payload size")
        return np.memmap(
            self.model_path,
            mode="r",
            dtype="<f2",
            offset=asset.offset + 16,
            shape=shape,
        )

    @staticmethod
    def _numpy(tensor):
        return tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)

    def _sample(self, logits):
        source = np.asarray(logits).reshape(-1)
        if source.size != self.vocab_size:
            raise RuntimeError(
                f"logits size {source.size} does not match vocabulary "
                f"size {self.vocab_size}")
        np.copyto(self._sampling_logits, source, casting="unsafe")
        values = self._sampling_logits
        if self.temp <= 0.0:
            return int(np.argmax(values))
        values /= self.temp
        candidates = np.arange(values.size)
        if 0 < self.top_k < values.size:
            selected = np.argpartition(values, -self.top_k)[-self.top_k:]
            candidates = candidates[selected]
            values = values[selected]
        values -= np.max(values)
        probabilities = np.exp(values)
        probabilities /= probabilities.sum()
        if 0.0 < self.top_p < 1.0:
            order = np.argsort(probabilities)[::-1]
            cumulative = np.cumsum(probabilities[order])
            keep = cumulative - probabilities[order] < self.top_p
            order = order[keep]
            candidates = candidates[order]
            probabilities = probabilities[order]
            probabilities /= probabilities.sum()
        return int(self._rng.choice(candidates, p=probabilities))

    def _prompt_tokens(self, prompt):
        messages = [{"role": "user", "content": str(prompt)}]
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return np.asarray(ids, dtype=np.int64).reshape(-1)

    def reset(self):
        if self._closed:
            raise RuntimeError("QwenM50Frontend is closed")
        if self._session_started and self.zero_kv_on_reset:
            for cache in self._cache_tensors:
                cache.set_zero()
        self._logits = None
        self._generated_ids = []
        self._valid_length = 0
        self._session_started = False
        self._finished = False
        self.last_decode_ms = 0.0

    def prefill(self, prompt=None, *, tokens=None, return_logits=True):
        if (prompt is None) == (tokens is None):
            raise ValueError("exactly one of prompt or tokens is required")
        self.reset()
        input_ids = self._prompt_tokens(prompt) if tokens is None else np.asarray(
            tokens, dtype=np.int64).reshape(-1)
        if input_ids.size == 0 or input_ids.size > self.prefill_length:
            raise ValueError(
                f"prompt requires 1..{self.prefill_length} tokens, got "
                f"{input_ids.size}")
        if np.any(input_ids < 0) or np.any(input_ids >= self.vocab_size):
            raise ValueError("prompt contains token IDs outside the vocabulary")

        self._prefill_embeddings.fill(0)
        self._prefill_embeddings[0, :input_ids.size] = self._embedding[input_ids]
        self._prefill_valid_length[0] = 0
        self._prefill_current_length[0] = input_ids.size
        self._prefill.set_input(
            self._prefill_input_names[0], self._prefill_embeddings)
        self._prefill.set_input(
            self._prefill_input_names[1], self._prefill_valid_length)
        self._prefill.set_input(
            self._prefill_input_names[2], self._prefill_current_length)

        started = time.perf_counter()
        self._prefill.run()
        self._prefill.sync()
        self.last_prefill_ms = (time.perf_counter() - started) * 1000.0
        self._logits = self._numpy(
            self._prefill.get_output(self._prefill_output_name))
        self._valid_length = int(input_ids.size)
        self._session_started = True
        return self._logits.copy() if return_logits else None

    def decode(self, *, return_text=True):
        if self._logits is None:
            raise RuntimeError("decode requires a successful prefill")
        if self._finished:
            raise RuntimeError("decode called after generation finished")
        if len(self._generated_ids) >= self.max_tokens:
            raise RuntimeError("decode exceeds configured max_tokens")

        token = self._sample(self._logits)
        is_eog = token in self.eos_token_ids
        self._generated_ids.append(token)
        budget_exhausted = len(self._generated_ids) >= self.max_tokens
        self._finished = is_eog or budget_exhausted
        decode_ms = 0.0
        if not is_eog and not budget_exhausted:
            if self._valid_length >= self.context_length:
                raise RuntimeError("decode exceeds the HMM context length")
            self._decode_embedding[0, 0] = self._embedding[token]
            self._decode_valid_length[0] = self._valid_length
            self._decode.set_input(
                self._decode_input_names[0], self._decode_embedding)
            self._decode.set_input(
                self._decode_input_names[1],
                self._decode_valid_length,
            )
            self._decode.set_input(
                self._decode_input_names[2],
                self._decode_current_length,
            )
            started = time.perf_counter()
            self._decode.run()
            self._decode.sync()
            decode_ms = (time.perf_counter() - started) * 1000.0
            self.last_decode_ms += decode_ms
            self._logits = self._numpy(
                self._decode.get_output(self._decode_output_name))
            self._valid_length += 1

        result = {"token": token, "is_eog": is_eog, "decode_ms": decode_ms}
        if return_text:
            result["text"] = self.get_text()
        return result

    def get_logits(self):
        if self._logits is None:
            raise RuntimeError("logits are not ready")
        return self._logits.copy()

    def get_text(self):
        ids = [token for token in self._generated_ids
               if token not in self.eos_token_ids]
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def generate(self, prompt):
        self.prefill(prompt, return_logits=False)
        for _ in range(self.max_tokens):
            result = self.decode(return_text=False)
            if result["is_eog"]:
                break
        return self.get_text()

    def close(self):
        if self._closed:
            return
        self._cache_tensors = []
        self._decode = None
        self._prefill = None
        mmap = getattr(self._embedding, "_mmap", None)
        if mmap is not None:
            mmap.close()
        self._embedding = None
        self._sampling_logits = None
        self._prefill_embeddings = None
        self._decode_embedding = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    @property
    def uses_hliellama(self):
        return False
