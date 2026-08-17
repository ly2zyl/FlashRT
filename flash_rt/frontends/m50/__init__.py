"""Houmo M50/XH2 frontends.

The M50 backend executes compiler-produced HMM graphs through ``tcim_lite``.
It deliberately keeps that optional dependency lazy so the rest of FlashRT
can still be imported on development hosts without the Houmo runtime.
"""

from .pi05 import Pi05M50Frontend
from .qwen import QwenM50Frontend

__all__ = ["Pi05M50Frontend", "QwenM50Frontend"]
