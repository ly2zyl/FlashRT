"""Validate FlashRT Qwen session state and resource handling on M50."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import flash_rt
import numpy as np


def _model_fd_count(model_path: Path) -> int:
    count = 0
    for descriptor in Path("/proc/self/fd").iterdir():
        try:
            if descriptor.resolve() == model_path:
                count += 1
        except FileNotFoundError:
            continue
    return count


def _generate(model, prompt: str) -> dict:
    model.prefill(prompt, return_logits=False)
    token_ids = []
    is_eog = False
    for _ in range(model.max_tokens):
        result = model.decode(return_text=False)
        token_ids.append(result["token"])
        is_eog = result["is_eog"]
        if is_eog:
            break
    return {
        "prompt": prompt,
        "text": model.get_text(),
        "token_ids": token_ids,
        "eog": is_eog,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    file_descriptors = {"before_load": _model_fd_count(model_path)}
    model = flash_rt.load_model(
        model_path,
        framework="tcim",
        config="qwen",
        device_id=args.device_id,
        max_tokens=args.max_tokens,
        temp=0.0,
        top_k=0,
        top_p=0.0,
        zero_kv_on_reset=True,
    )
    file_descriptors["while_loaded"] = _model_fd_count(model_path)
    graph_contract = {
        "prefill_length": model.prefill_length,
        "context_length": model.context_length,
        "vocab_size": model.vocab_size,
        "hidden_size": model.hidden_size,
        "kv_cache_tensors": len(model._cache_tensors),
    }

    try:
        model.zero_kv_on_reset = False
        session_runs = [
            _generate(model, "请只输出数字2，不要输出其他内容。 /no_think"),
            _generate(model, "请只输出数字3，不要输出其他内容。 /no_think"),
            _generate(model, "请只输出数字2，不要输出其他内容。 /no_think"),
        ]
        model.zero_kv_on_reset = True
        reset_runs = [
            _generate(model, "请只输出数字2，不要输出其他内容。 /no_think"),
            _generate(model, "请只输出数字2，不要输出其他内容。 /no_think"),
        ]
        maps = Path("/proc/self/maps").read_text(encoding="utf-8")
        try:
            model.decode(return_text=False)
        except RuntimeError as exc:
            eog_guard = "after generation finished" in str(exc)
        else:
            eog_guard = False

        boundary_tokens = np.zeros(model.prefill_length, dtype=np.int64)
        model.prefill(tokens=boundary_tokens, return_logits=False)
        boundary_logits = model.get_logits()
        prefill_boundary_passed = bool(
            boundary_logits.shape == (1, 1, model.vocab_size)
            and np.isfinite(boundary_logits).all()
        )
        try:
            model.prefill(
                tokens=np.zeros(model.prefill_length + 1, dtype=np.int64),
                return_logits=False,
            )
        except ValueError as exc:
            overlength_guard = "prompt requires" in str(exc)
        else:
            overlength_guard = False
    finally:
        model.close()

    file_descriptors["after_close"] = _model_fd_count(model_path)
    summary = {
        "model": str(model_path),
        "session_runs": session_runs,
        "session_outputs": [run["text"] for run in session_runs],
        "session_isolation_passed": [
            run["text"].rstrip().endswith(expected)
            for run, expected in zip(session_runs, ("2", "3", "2"))
        ] == [True, True, True],
        "logical_reset_test_zero_kv_on_reset": False,
        "physical_reset_test_zero_kv_on_reset": True,
        "physical_reset_outputs_repeatable": (
            reset_runs[0]["token_ids"] == reset_runs[1]["token_ids"]
        ),
        "eog_guard_passed": eog_guard,
        "graph_contract": graph_contract,
        "prefill_256_token_boundary_passed": prefill_boundary_passed,
        "prefill_257_token_rejected": overlength_guard,
        "model_file_descriptors": file_descriptors,
        "libllama_mapped": "libllama.so" in maps,
        "tcim_runtime_mapped": "libtcim_runtime_lite.so" in maps,
    }
    serialized = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    print(serialized, end="")
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
