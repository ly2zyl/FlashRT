import numpy as np
import pytest

from flash_rt.core.utils.actions import transform_feature


def test_mean_std_round_trip_and_padded_dimensions():
    stats = {"mean": [1.0, -2.0], "std": [2.0, 4.0]}
    raw = np.array([3.0, 2.0, 9.0], dtype=np.float32)

    normalized = transform_feature(raw, stats, "MEAN_STD")
    np.testing.assert_allclose(normalized, [1.0, 1.0, 9.0])
    restored = transform_feature(normalized, stats, "MEAN_STD", inverse=True)
    np.testing.assert_allclose(restored, raw)


@pytest.mark.parametrize(
    ("mode", "stats"),
    [
        ("MIN_MAX", {"min": [-2.0], "max": [2.0]}),
        ("QUANTILES", {"q01": [-2.0], "q99": [2.0]}),
        ("QUANTILE10", {"q10": [-2.0], "q90": [2.0]}),
    ],
)
def test_range_modes_round_trip(mode, stats):
    normalized = transform_feature(np.array([1.0]), stats, mode)
    np.testing.assert_allclose(normalized, [0.5])
    restored = transform_feature(normalized, stats, mode, inverse=True)
    np.testing.assert_allclose(restored, [1.0])


def test_unknown_normalization_mode_is_rejected():
    with pytest.raises(ValueError, match="unsupported normalization mode"):
        transform_feature(np.array([0.0]), {}, "mystery")
