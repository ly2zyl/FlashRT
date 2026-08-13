"""Plain-text GGUF frontend backed by Houmo HLIELLama on M50/XH2.

The provider DSO implements FlashRT's stable ``frt_model_runtime_v1`` ABI and
links to the vendor ``libllama.so``.  The HLIELLama installation and GGUF are
treated as read-only external deployment inputs.
"""

from __future__ import annotations

import os
from pathlib import Path

from flash_rt.frontends.jetson_pi.llm import LlmJetsonPiFrontend


class LlmHoumoFrontend(LlmJetsonPiFrontend):
    """FlashRT text-generation frontend for Houmo-quantized GGUF models."""

    def __init__(self, checkpoint, *, backend="houmo", lib_path=None,
                 **kwargs):
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Houmo GGUF model not found: {checkpoint}")
        if backend not in ("houmo", "m50"):
            raise ValueError("Houmo Llama backend must be 'houmo' or 'm50'")
        if lib_path is None:
            lib_path = os.environ.get("FLASHRT_HOUMO_LLAMA_LIB")
        if not lib_path:
            raise RuntimeError(
                "Houmo Llama provider DSO is required; pass lib_path=... or "
                "set FLASHRT_HOUMO_LLAMA_LIB")
        super().__init__(
            str(checkpoint_path), backend=backend, lib_path=lib_path, **kwargs)
