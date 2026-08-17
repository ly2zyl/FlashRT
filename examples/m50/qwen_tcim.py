"""Run and benchmark FlashRT-native Qwen inference through TCIM on M50."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import flash_rt


def run_staged(model, prompt, max_tokens):
    started = time.perf_counter()
    model.prefill(prompt, return_logits=False)
    prefill_total_ms = (time.perf_counter() - started) * 1000.0
    tokens = []
    decode_hmm_steps = 0
    first_decode_call_ms = None
    eog = False
    decode_wall_started = time.perf_counter()
    for _ in range(max_tokens):
        decode_call_started = time.perf_counter()
        result = model.decode(return_text=False)
        decode_call_ms = (time.perf_counter() - decode_call_started) * 1000.0
        if first_decode_call_ms is None:
            first_decode_call_ms = decode_call_ms
        tokens.append(result["token"])
        if result["decode_ms"] > 0.0:
            decode_hmm_steps += 1
        eog = result["is_eog"]
        if eog:
            break
    decode_wall_ms = (time.perf_counter() - decode_wall_started) * 1000.0
    request_wall_ms = prefill_total_ms + decode_wall_ms
    return {
        "mode": "staged",
        "prefill_total_ms": prefill_total_ms,
        "prefill_hmm_ms": model.last_prefill_ms,
        "caller_visible_first_token_ms": (
            prefill_total_ms + first_decode_call_ms
            if first_decode_call_ms is not None else None
        ),
        "decode_wall_ms": decode_wall_ms,
        "decode_hmm_ms": model.last_decode_ms,
        "request_wall_ms": request_wall_ms,
        "output_tokens": len(tokens),
        "post_prefill_output_tokens_per_second": (
            len(tokens) * 1000.0 / decode_wall_ms if decode_wall_ms else 0.0
        ),
        "request_output_tokens_per_second": (
            len(tokens) * 1000.0 / request_wall_ms if request_wall_ms else 0.0
        ),
        "decode_hmm_steps": decode_hmm_steps,
        "decode_hmm_steps_per_second": (
            decode_hmm_steps * 1000.0 / model.last_decode_ms
            if model.last_decode_ms else 0.0
        ),
        "host_orchestration_ms": decode_wall_ms - model.last_decode_ms,
        "eog": eog,
        "token_ids": tokens,
        "text": model.get_text(),
    }


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="请只回答一个数字：一加一等于多少？")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output-json")
    parser.add_argument("--zero-kv-on-reset", action="store_true")
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be greater than zero")
    if args.repeat <= 0:
        parser.error("--repeat must be greater than zero")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

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
    prompt_tokens = int(model._prompt_tokens(args.prompt).size)

    runs = []
    try:
        for _ in range(args.warmup):
            run_staged(model, args.prompt, args.max_tokens)
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
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "prompt_tokens": prompt_tokens,
        "max_tokens": args.max_tokens,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "zero_kv_on_reset": args.zero_kv_on_reset,
        "sampling": {"temperature": 0.0, "top_k": 0, "top_p": 0.0},
        "libllama_mapped": "libllama.so" in maps,
        "tcim_runtime_mapped": "libtcim_runtime_lite.so" in maps,
        "runs": runs,
        "prefill_total_ms_p50": statistics.median(
            run["prefill_total_ms"] for run in runs),
        "prefill_total_ms_p95": percentile(
            [run["prefill_total_ms"] for run in runs], 0.95),
        "prefill_hmm_ms_p50": statistics.median(
            run["prefill_hmm_ms"] for run in runs),
        "prefill_hmm_ms_p95": percentile(
            [run["prefill_hmm_ms"] for run in runs], 0.95),
        "caller_visible_first_token_ms_p50": statistics.median(
            run["caller_visible_first_token_ms"] for run in runs),
        "caller_visible_first_token_ms_p95": percentile(
            [run["caller_visible_first_token_ms"] for run in runs], 0.95),
        "request_wall_ms_p50": statistics.median(
            run["request_wall_ms"] for run in runs),
        "request_wall_ms_p95": percentile(
            [run["request_wall_ms"] for run in runs], 0.95),
        "post_prefill_output_tokens_per_second_p50": statistics.median(
            run["post_prefill_output_tokens_per_second"] for run in runs),
        "request_output_tokens_per_second_p50": statistics.median(
            run["request_output_tokens_per_second"] for run in runs),
        "decode_hmm_steps_per_second_p50": statistics.median(
            run["decode_hmm_steps_per_second"] for run in runs),
        "host_orchestration_ms_p50": statistics.median(
            run["host_orchestration_ms"] for run in runs),
        "host_orchestration_ms_p95": percentile(
            [run["host_orchestration_ms"] for run in runs], 0.95),
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
