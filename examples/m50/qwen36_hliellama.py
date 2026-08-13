"""Minimal Qwen3.6 text generation through FlashRT on Houmo M50."""

from __future__ import annotations

import argparse

import flash_rt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Houmo-quantized GGUF")
    parser.add_argument("--provider-lib", required=True,
                        help="libflashrt_cpp_llama_cpp_provider_c.so")
    parser.add_argument("--prompt", default="请只回答一个数字：一加一等于多少？")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--ctx-size", type=int, default=512)
    args = parser.parse_args()

    model = flash_rt.load_model(
        args.model,
        framework="houmo_llama",
        config="llm",
        backend="houmo",
        n_ctx=args.ctx_size,
        temp=0.0,
        top_k=0,
        top_p=0.0,
        max_tokens=args.max_tokens,
        lib_path=args.provider_lib,
    )
    print(model.generate(args.prompt))
    model.close()


if __name__ == "__main__":
    main()
