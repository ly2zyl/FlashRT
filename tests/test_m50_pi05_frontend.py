from pathlib import Path

import pytest

from flash_rt.frontends.m50.pi05 import (
    M50ArtifactError,
    M50Pi05Artifacts,
    M50_REQUIRED_FILES,
)
from flash_rt.hardware import resolve_pipeline_class


def test_m50_pi05_dispatch_is_registered():
    cls = resolve_pipeline_class("pi05", "torch", "m50_hmm_compat")
    assert cls.__name__ == "Pi05M50Frontend"
    assert cls.__module__ == "flash_rt.frontends.m50.pi05"


def test_m50_artifact_bundle_resolves_explicit_directory(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in M50_REQUIRED_FILES:
        (bundle / name).touch()

    artifacts = M50Pi05Artifacts.resolve(tmp_path / "checkpoint", bundle)
    assert artifacts.model_dir == bundle.resolve()
    assert artifacts.path("siglip.hmm") == bundle.resolve() / "siglip.hmm"


def test_m50_artifact_bundle_reports_every_missing_file(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "siglip.hmm").touch()

    with pytest.raises(M50ArtifactError) as exc_info:
        M50Pi05Artifacts.resolve(tmp_path / "checkpoint", bundle)

    message = str(exc_info.value)
    assert "siglip.hmm" not in message
    for name in M50_REQUIRED_FILES[1:]:
        assert name in message
