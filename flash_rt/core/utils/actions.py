"""FlashRT — Action post-processing utilities."""

import numpy as np

LIBERO_ACTION_DIM = 7


def _stat_array(stats, name: str) -> np.ndarray:
    if name not in stats:
        raise ValueError(f"normalization statistics do not contain {name!r}")
    return np.asarray(stats[name], dtype=np.float32).reshape(-1)


def transform_feature(values, stats, mode: str, *, inverse: bool = False):
    """Apply a LeRobot feature normalization transform using NumPy.

    Only dimensions covered by ``stats`` are transformed. This is useful for
    Pi0.5, whose HMM graphs operate on 32 padded action/state dimensions while
    LIBERO statistics cover the seven action or eight state dimensions.
    """
    result = np.asarray(values, dtype=np.float32).copy()
    normalized_mode = str(mode).upper()
    if normalized_mode == "IDENTITY":
        return result

    if normalized_mode == "MEAN_STD":
        first = _stat_array(stats, "mean")
        second = _stat_array(stats, "std")
    elif normalized_mode == "MIN_MAX":
        first = _stat_array(stats, "min")
        second = _stat_array(stats, "max")
    elif normalized_mode == "QUANTILES":
        first = _stat_array(stats, "q01")
        second = _stat_array(stats, "q99")
    elif normalized_mode == "QUANTILE10":
        first = _stat_array(stats, "q10")
        second = _stat_array(stats, "q90")
    else:
        raise ValueError(f"unsupported normalization mode: {mode!r}")

    dim = min(result.shape[-1], first.size, second.size)
    target = result[..., :dim]
    first = first[:dim]
    second = second[:dim]
    if normalized_mode == "MEAN_STD":
        if inverse:
            result[..., :dim] = target * second + first
        else:
            result[..., :dim] = target - first
            result[..., :dim] /= second + 1e-8
        return result

    denominator = second - first
    denominator = np.where(denominator == 0, 1e-8, denominator)
    if inverse:
        result[..., :dim] = (target + 1.0) * denominator / 2.0 + first
    else:
        result[..., :dim] = 2.0 * (target - first) / denominator - 1.0
    return result


def unnormalize_actions(actions, norm_stats):
    """Unnormalize actions using q01/q99 statistics (pure numpy)."""
    q01 = np.array(norm_stats["actions"]["q01"], dtype=np.float32)
    q99 = np.array(norm_stats["actions"]["q99"], dtype=np.float32)
    dim = min(actions.shape[-1], len(q01))
    clipped = np.clip(actions, -1.0, 1.0)
    unnorm = clipped.copy()
    unnorm[..., :dim] = (
        (clipped[..., :dim] + 1.0) / 2.0 * (q99[:dim] - q01[:dim] + 1e-6)
        + q01[:dim]
    )
    return unnorm
