"""Run and benchmark FlashRT-native Qwen inference through TCIM on M50."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import flash_rt


def run_staged(model, prompt, max_tokens):
    started = time.perf_counter()
    model.prefill(prompt, return_logits=False)
    prefill_total_ms = (time.perf_counter() - started) * 1000.0
    tokens = []
    eog = False
    decode_wall_started = time.perf_counter()
    for _ in range(max_tokens):
        result = model.decode(return_text=False)
        tokens.append(result["token"])
        eog = result["is_eog"]
        if eog:
            break
    decode_wall_ms = (time.perf_counter() - decode_wall_started) * 1000.0
    return {
        "mode": "staged",
        "prefill_total_ms": prefill_total_ms,
        "prefill_hmm_ms": model.last_prefill_ms,
        "decode_wall_ms": decode_wall_ms,
        "decode_hmm_ms": model.last_decode_ms,
        "decode_tokens": len(tokens),
        "decode_tokens_per_second": (
            len(tokens) * 1000.0 / decode_wall_ms if decode_wall_ms else 0.0
        ),
        "eog": eog,
        "token_ids": tokens,
        "text": model.get_text(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="请只回答一个数字：一加一等于多少？")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output-json")
    parser.add_argument("--zero-kv-on-reset", action="store_true")
    args = parser.parse_args()

    load_started = time.perf_counter()
    model = flash_rt.load_model(
        args.model,
        framework="tcim",
        config="qwen",
        device_id=args.device_id,
        max_tokens=args.max_tokens,
        temp=0.0,
        top_k=0,
        top_p=0.0,
        zero_kv_on_reset=args.zero_kv_on_reset,
    )
    load_ms = (time.perf_counter() - load_started) * 1000.0

    runs = []
    try:
        for repeat in range(1, args.repeat + 1):
            result = run_staged(model, args.prompt, args.max_tokens)
            result["repeat"] = repeat
            runs.append(result)
        maps = Path("/proc/self/maps").read_text(encoding="utf-8")
    finally:
        model.close()

    summary = {
        "model": str(Path(args.model).resolve()),
        "framework": "tcim",
        "execution_path": "FlashRT -> tcim_lite -> TCIM Runtime -> M50",
        "model_load_ms": load_ms,
        "max_tokens": args.max_tokens,
        "repeat": args.repeat,
        "zero_kv_on_reset": args.zero_kv_on_reset,
        "libllama_mapped": "libllama.so" in maps,
        "tcim_runtime_mapped": "libtcim_runtime_lite.so" in maps,
        "runs": runs,
        "prefill_hmm_ms_median": statistics.median(
            run["prefill_hmm_ms"] for run in runs),
        "decode_tokens_per_second_median": statistics.median(
            run["decode_tokens_per_second"] for run in runs),
        "token_ids_repeatable": all(
            run["token_ids"] == runs[0]["token_ids"] for run in runs[1:]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
