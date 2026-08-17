from pathlib import Path

import flash_rt.api as api
from flash_rt.frontends.houmo_llama import llm as houmo_llm


def test_load_model_forwards_houmo_qwen_options(monkeypatch, tmp_path: Path):
    captured = {}
    sentinel = object()

    def fake_frontend(checkpoint, **kwargs):
        captured["checkpoint"] = checkpoint
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(houmo_llm, "LlmHoumoFrontend", fake_frontend)
    model_path = tmp_path / "qwen.gguf"
    identity = "a" * 64

    model = api.load_model(
        model_path,
        framework="houmo_llama",
        config="llm",
        backend="houmo",
        n_ctx=512,
        n_threads=8,
        temp=0.0,
        top_k=0,
        top_p=0.0,
        seed=7,
        max_tokens=32,
        model_identity=identity,
        lib_path="/tmp/provider.so",
    )

    assert model is sentinel
    assert captured == {
        "checkpoint": model_path,
        "backend": "houmo",
        "n_ctx": 512,
        "n_threads": 8,
        "temp": 0.0,
        "top_k": 0,
        "top_p": 0.0,
        "seed": 7,
        "max_tokens": 32,
        "model_identity": identity,
        "lib_path": "/tmp/provider.so",
    }


def test_houmo_default_backend_is_normalized(monkeypatch):
    captured = {}

    def fake_frontend(_checkpoint, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(houmo_llm, "LlmHoumoFrontend", fake_frontend)
    api.load_model("qwen.gguf", framework="houmo_llama", config="llm")

    assert captured["backend"] == "houmo"
