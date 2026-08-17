"""Deploy and benchmark a Houmo-quantized Qwen GGUF through FlashRT on M50.

The staged path keeps the model and KV cache resident, lets FlashRT schedule
prefill/decode explicitly, avoids copying the full vocabulary logits to
Python, and copies accumulated text only once after decoding.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import flash_rt


def run_one_shot(model, prompt: str) -> dict:
    started = time.perf_counter()
    text = model.generate(prompt)
    return {
        "mode": "one_shot",
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "text": text,
    }


def run_staged(model, prompt: str, max_tokens: int) -> dict:
    prefill_started = time.perf_counter()
    model.prefill(prompt, return_logits=False)
    prefill_ms = (time.perf_counter() - prefill_started) * 1000.0

    decode_ms = 0.0
    tokens = []
    is_eog = False
    for _ in range(max_tokens):
        decode_started = time.perf_counter()
        step = model.decode(return_text=False)
        decode_ms += (time.perf_counter() - decode_started) * 1000.0
        tokens.append(step["token"])
        is_eog = step["is_eog"]
        if is_eog:
            break

    text = model.get_text() if tokens and not (len(tokens) == 1 and is_eog) else ""
    return {
        "mode": "staged",
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "decode_tokens": len(tokens),
        "decode_tokens_per_second": (
            len(tokens) * 1000.0 / decode_ms if decode_ms > 0.0 else 0.0
        ),
        "eog": is_eog,
        "token_ids": tokens,
        "text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Houmo Qwen GGUF")
    parser.add_argument("--provider-lib", required=True,
                        help="FlashRT Houmo provider DSO")
    parser.add_argument("--prompt", default="请只回答一个数字：一加一等于多少？")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--mode", choices=("one-shot", "staged", "both"),
                        default="both")
    parser.add_argument("--output-json")
    parser.add_argument(
        "--model-sha256-file",
        help="file whose first field is the pre-verified model SHA-256",
    )
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.repeat <= 0:
        parser.error("--max-tokens and --repeat must be greater than zero")

    model_identity = None
    if args.model_sha256_file:
        fields = Path(args.model_sha256_file).read_text(encoding="utf-8").split()
        if not fields:
            parser.error("--model-sha256-file is empty")
        model_identity = fields[0]

    load_started = time.perf_counter()
    model = flash_rt.load_model(
        args.model,
        framework="houmo_llama",
        config="llm",
        backend="houmo",
        n_ctx=args.ctx_size,
        n_threads=args.threads,
        temp=0.0,
        top_k=0,
        top_p=0.0,
        max_tokens=args.max_tokens,
        model_identity=model_identity,
        lib_path=args.provider_lib,
    )
    load_ms = (time.perf_counter() - load_started) * 1000.0

    runs = []
    try:
        for repeat in range(1, args.repeat + 1):
            if args.mode in ("one-shot", "both"):
                result = run_one_shot(model, args.prompt)
                result["repeat"] = repeat
                runs.append(result)
            if args.mode in ("staged", "both"):
                result = run_staged(model, args.prompt, args.max_tokens)
                result["repeat"] = repeat
                runs.append(result)
    finally:
        model.close()

    staged = [run for run in runs if run["mode"] == "staged"]
    one_shot = [run for run in runs if run["mode"] == "one_shot"]
    summary = {
        "model": str(Path(args.model).resolve()),
        "provider_lib": str(Path(args.provider_lib).resolve()),
        "backend": "houmo",
        "model_identity": model_identity,
        "ctx_size": args.ctx_size,
        "max_tokens": args.max_tokens,
        "repeat": args.repeat,
        "model_load_ms": load_ms,
        "runs": runs,
    }
    if one_shot:
        summary["one_shot_latency_ms_median"] = statistics.median(
            run["latency_ms"] for run in one_shot)
    if staged:
        summary.update({
            "staged_prefill_ms_median": statistics.median(
                run["prefill_ms"] for run in staged),
            "staged_decode_ms_median": statistics.median(
                run["decode_ms"] for run in staged),
            "staged_decode_tokens_per_second_median": statistics.median(
                run["decode_tokens_per_second"] for run in staged),
        })
        summary["staged_token_ids_repeatable"] = all(
            run["token_ids"] == staged[0]["token_ids"] for run in staged[1:]
        )
    if one_shot and staged:
        staged_by_repeat = {run["repeat"]: run for run in staged}
        summary["one_shot_staged_text_match_all"] = all(
            run["text"] == staged_by_repeat[run["repeat"]]["text"]
            for run in one_shot
        )
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
