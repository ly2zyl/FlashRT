#!/usr/bin/env python3
"""Check whether the local host/container can run FlashRT Pi0.5 on M50."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys


REQUIRED_HMM_BUNDLE = (
    "siglip.hmm",
    "gemma_2b_prefill.hmm",
    "gemma_expert_300m_decode.hmm",
    "action_in_proj.hmm",
    "action_out_proj.hmm",
    "time_mlp.hmm",
    "embedding.pt",
)


def report(ok: bool, label: str, details: str) -> None:
    marker = "OK" if ok else "FAIL"
    print(f"[{marker}] {label}: {details}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--compiled-model-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    failures = 0
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    compiled = Path(args.compiled_model_dir).expanduser().resolve()
    tokenizer = Path(args.tokenizer).expanduser().resolve()

    checks = (
        (checkpoint / "config.json", "Pi0.5 config"),
        (tokenizer / "tokenizer_config.json", "PaliGemma tokenizer"),
    )
    for path, label in checks:
        ok = path.is_file()
        report(ok, label, str(path))
        failures += int(not ok)

    missing = [name for name in REQUIRED_HMM_BUNDLE if not (compiled / name).is_file()]
    ok = not missing
    report(ok, "M50 compiled bundle", str(compiled))
    for name in missing:
        print(f"       missing: {compiled / name}")
    failures += int(not ok)

    for module in ("numpy", "torch", "transformers", "safetensors", "tcim_lite"):
        ok = importlib.util.find_spec(module) is not None
        report(ok, f"Python module {module}", sys.executable)
        failures += int(not ok)

    hm_smi = shutil.which("hm_smi")
    if hm_smi:
        result = subprocess.run(
            [hm_smi], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        report(result.returncode == 0, "hm_smi", hm_smi)
        failures += int(result.returncode != 0)
    else:
        report(False, "hm_smi", "not found on PATH")
        failures += 1

    try:
        import tcim_lite as tcim

        count = int(tcim.runtime.get_device_num("Xh2HalBackend"))
        ok = 0 <= args.device_id < count
        report(ok, "XH2 device", f"requested={args.device_id}, visible={count}")
        failures += int(not ok)
    except Exception as exc:
        report(False, "XH2 device", str(exc))
        failures += 1

    if failures:
        print(f"\nM50 Pi0.5 environment is not ready ({failures} failed checks).")
        return 1
    print("\nM50 Pi0.5 environment is ready for the FlashRT smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
