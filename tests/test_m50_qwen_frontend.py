import struct
from pathlib import Path

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
