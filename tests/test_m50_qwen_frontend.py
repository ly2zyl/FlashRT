import struct
from pathlib import Path

import numpy as np
import pytest

import flash_rt.api as api
from flash_rt.frontends.m50 import qwen as m50_qwen
from flash_rt.frontends.m50.gguf_assets import GGUFAssetIndex


def _gguf_string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def test_gguf_asset_index_locates_embedded_byte_tensor(tmp_path: Path):
    header = bytearray(b"GGUF")
    header += struct.pack("<IQQ", 3, 1, 2)
    header += _gguf_string("general.alignment")
    header += struct.pack("<II", 4, 32)
    header += _gguf_string("is_hmm")
    header += struct.pack("<I?", 7, True)
    header += _gguf_string("payload.bin")
    header += struct.pack("<IQIQ", 1, 5, 24, 0)
    header += b"\0" * ((32 - len(header) % 32) % 32)
    payload_offset = len(header)
    model = tmp_path / "model.gguf"
    model.write_bytes(header + b"abcde")

    index = GGUFAssetIndex(model)
    asset = index.require("payload.bin")

    assert index.version == 3
    assert index.metadata["is_hmm"] is True
    assert asset.offset == payload_offset
    assert asset.size == 5
    assert index.read_bytes("payload.bin") == b"abcde"


def test_load_model_forwards_native_tcim_qwen_options(monkeypatch, tmp_path):
    captured = {}
    sentinel = object()

    def fake_frontend(checkpoint, **kwargs):
        captured["checkpoint"] = checkpoint
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(m50_qwen, "QwenM50Frontend", fake_frontend)
    checkpoint = tmp_path / "qwen.gguf"
    model = api.load_model(
        checkpoint,
        framework="tcim",
        config="qwen",
        device_id=0,
        temp=0.0,
        top_k=0,
        top_p=0.0,
        seed=7,
        max_tokens=32,
        tokenizer_cache_dir=tmp_path / "tokenizer",
        zero_kv_on_reset=True,
    )

    assert model is sentinel
    assert captured == {
        "checkpoint": checkpoint,
        "device_id": 0,
        "temp": 0.0,
        "top_k": 0,
        "top_p": 0.0,
        "seed": 7,
        "max_tokens": 32,
        "tokenizer_cache_dir": tmp_path / "tokenizer",
        "zero_kv_on_reset": True,
    }


def test_tcim_framework_rejects_non_qwen_config():
    try:
        api.load_model("model.gguf", framework="tcim", config="llm")
    except ValueError as exc:
        assert "Supported: qwen" in str(exc)
    else:
        raise AssertionError("tcim framework accepted an unsupported config")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"device_id": -1}, "device_id must be non-negative"),
        ({"max_tokens": 0}, "max_tokens must be greater than zero"),
        ({"temp": -0.1}, "temp must be non-negative"),
        ({"top_k": -1}, "top_k must be non-negative"),
        ({"top_p": 1.1}, "top_p must be in the range"),
    ],
)
def test_qwen_frontend_rejects_invalid_runtime_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        m50_qwen.QwenM50Frontend("missing.gguf", **kwargs)


def test_decode_rejects_steps_after_eog():
    frontend = object.__new__(m50_qwen.QwenM50Frontend)
    frontend._logits = np.array([0.0, 1.0], dtype=np.float16)
    frontend._generated_ids = []
    frontend._finished = False
    frontend.max_tokens = 4
    frontend.temp = 0.0
    frontend.vocab_size = 2
    frontend._sampling_logits = np.empty(2, dtype=np.float32)
    frontend.eos_token_ids = {1}

    result = frontend.decode(return_text=False)

    assert result == {"token": 1, "is_eog": True, "decode_ms": 0.0}
    with pytest.raises(RuntimeError, match="after generation finished"):
        frontend.decode(return_text=False)


def test_close_releases_embedding_memmap():
    class FakeMmap:
        closed = False

        def close(self):
            self.closed = True

    class FakeEmbedding:
        _mmap = FakeMmap()

    frontend = object.__new__(m50_qwen.QwenM50Frontend)
    frontend._closed = False
    frontend._cache_tensors = [object()]
    frontend._decode = object()
    frontend._prefill = object()
    frontend._embedding = FakeEmbedding()
    mmap = frontend._embedding._mmap

    frontend.close()

    assert mmap.closed is True
    assert frontend._embedding is None
    assert frontend._closed is True
