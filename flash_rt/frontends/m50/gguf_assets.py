"""Minimal GGUF V2/V3 reader for byte-packed M50 deployment assets.

This module intentionally does not import llama.cpp or ``libllama.so``.  It
only locates byte tensors embedded in a GGUF container so TCIM can load HMM
subranges directly from the original file.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path


_VALUE_SIZES = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}


@dataclass(frozen=True)
class GGUFAsset:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int
    size: int


class GGUFAssetIndex:
    """Index byte tensors without depending on a model inference framework."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.version = 0
        self.metadata = {}
        self.assets = {}
        self._parse()

    @staticmethod
    def _read_exact(handle, size):
        data = handle.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF file")
        return data

    @classmethod
    def _u32(cls, handle):
        return struct.unpack("<I", cls._read_exact(handle, 4))[0]

    @classmethod
    def _u64(cls, handle):
        return struct.unpack("<Q", cls._read_exact(handle, 8))[0]

    @classmethod
    def _string(cls, handle):
        size = cls._u64(handle)
        return cls._read_exact(handle, size).decode("utf-8")

    @classmethod
    def _value(cls, handle, value_type, *, keep=False):
        if value_type in _VALUE_SIZES:
            size = _VALUE_SIZES[value_type]
            data = cls._read_exact(handle, size)
            if not keep:
                return None
            formats = {
                0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I",
                5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q",
                12: "<d",
            }
            return struct.unpack(formats[value_type], data)[0]
        if value_type == 8:  # STRING
            value = cls._string(handle)
            return value if keep else None
        if value_type == 9:  # ARRAY
            element_type = cls._u32(handle)
            count = cls._u64(handle)
            if element_type in _VALUE_SIZES and not keep:
                handle.seek(_VALUE_SIZES[element_type] * count, 1)
                return None
            values = [] if keep else None
            for _ in range(count):
                value = cls._value(handle, element_type, keep=keep)
                if keep:
                    values.append(value)
            return values
        raise ValueError(f"unsupported GGUF metadata value type: {value_type}")

    def _parse(self):
        file_size = self.path.stat().st_size
        with self.path.open("rb") as handle:
            if self._read_exact(handle, 4) != b"GGUF":
                raise ValueError(f"not a GGUF file: {self.path}")
            self.version = self._u32(handle)
            if self.version not in (2, 3):
                raise ValueError(f"unsupported GGUF version: {self.version}")
            tensor_count = self._u64(handle)
            metadata_count = self._u64(handle)

            retained = {
                "general.alignment",
                "general.architecture",
                "general.name",
                "is_hmm",
                "hmm.info",
            }
            for _ in range(metadata_count):
                key = self._string(handle)
                value_type = self._u32(handle)
                value = self._value(handle, value_type, keep=key in retained)
                if key in retained:
                    self.metadata[key] = value

            entries = []
            for _ in range(tensor_count):
                name = self._string(handle)
                dimensions = self._u32(handle)
                shape = tuple(self._u64(handle) for _ in range(dimensions))
                ggml_type = self._u32(handle)
                relative_offset = self._u64(handle)
                entries.append((name, shape, ggml_type, relative_offset))

            alignment = int(self.metadata.get("general.alignment", 32))
            data_start = (handle.tell() + alignment - 1) // alignment * alignment
            for name, shape, ggml_type, relative_offset in entries:
                # Houmo stores deployment files as custom I8 byte tensors.  Their
                # tensor shape is the exact payload size, excluding GGUF padding.
                if ggml_type != 24:
                    continue
                size = math.prod(shape)
                offset = data_start + relative_offset
                if offset < data_start or offset + size > file_size:
                    raise ValueError(f"GGUF tensor {name!r} exceeds file bounds")
                self.assets[name] = GGUFAsset(
                    name=name,
                    shape=shape,
                    ggml_type=ggml_type,
                    offset=offset,
                    size=size,
                )

    def require(self, name):
        try:
            return self.assets[name]
        except KeyError as exc:
            raise ValueError(f"GGUF does not contain required asset {name!r}") from exc

    def read_bytes(self, name):
        asset = self.require(name)
        with self.path.open("rb") as handle:
            handle.seek(asset.offset)
            return self._read_exact(handle, asset.size)
