import numpy as np

MODES = ("artifact_only", "independent", "shared_pair")


def normalize_pair(artifact, ground_truth, mode="artifact_only"):
    if mode not in MODES:
        raise ValueError(f"Unknown normalization mode: {mode}")

    a = np.asarray(artifact, dtype=np.float32)
    g = np.asarray(ground_truth, dtype=np.float32)

    if a.shape != g.shape:
        raise ValueError("Artifact and GT shapes differ")

    if not np.isfinite(a).all() or not np.isfinite(g).all():
        raise ValueError("Nonfinite image values")

    al, ah = np.percentile(a, [1, 99])
    gl, gh = np.percentile(g, [1, 99])

    def scale(x, lo, hi):
        return np.clip(
            (x - lo) / max(float(hi - lo), 1e-6),
            0, 1
        ).astype(np.float32)

    if mode == "artifact_only":
        return scale(a, al, ah), scale(g, al, ah)

    if mode == "independent":
        return scale(a, al, ah), scale(g, gl, gh)

    lo = min(al, gl)
    hi = max(ah, gh)
    return scale(a, lo, hi), scale(g, lo, hi)
