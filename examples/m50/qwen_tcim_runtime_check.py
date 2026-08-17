"""Validate FlashRT Qwen session state and resource handling on M50."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import flash_rt


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

    try:
        session_runs = [
            _generate(model, "请只输出数字2，不要输出其他内容。 /no_think"),
            _generate(model, "请只输出数字3，不要输出其他内容。 /no_think"),
            _generate(model, "请只输出数字2，不要输出其他内容。 /no_think"),
        ]
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
        "physical_kv_reset_repeatable": (
            reset_runs[0]["token_ids"] == reset_runs[1]["token_ids"]
        ),
        "eog_guard_passed": eog_guard,
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
