from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import sobel

MANIFEST = "manifests/dataset_manifest.csv"
OUTPUT = Path("outputs/active_pair_quality.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)
df = df[df["split"].isin(["train", "val", "test"])].copy()

rows = []

def corr(x, y):
    x = x.ravel().astype(np.float64)
    y = y.ravel().astype(np.float64)

    x -= x.mean()
    y -= y.mean()

    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / den) if den > 1e-12 else np.nan

for i, row in enumerate(df.itertuples(index=False), 1):
    result = {
        "volume_id": row.volume_id,
        "group_id": row.group_id,
        "split": row.split,
        "plane": row.plane,
    }

    try:
        a = nib.load(row.artifact_path).get_fdata(dtype=np.float32)
        g = nib.load(row.ground_truth_path).get_fdata(dtype=np.float32)

        result["finite_fraction_artifact"] = float(np.isfinite(a).mean())
        result["finite_fraction_gt"] = float(np.isfinite(g).mean())

        if a.shape != g.shape:
            raise ValueError("Shape mismatch")

        if not np.isfinite(a).all() or not np.isfinite(g).all():
            raise ValueError("Nonfinite values")

        # Three representative slices from each volume
        indices = sorted(set([
            max(0, a.shape[2] // 4),
            a.shape[2] // 2,
            min(a.shape[2] - 1, 3 * a.shape[2] // 4),
        ]))

        correlations = []
        edge_correlations = []
        clipped_low = []
        clipped_high = []
        dynamic_ranges = []

        for z in indices:
            x = a[:, :, z]
            y = g[:, :, z]

            lo, hi = np.percentile(x, [1, 99])
            scale = max(float(hi - lo), 1e-6)

            xn = np.clip((x - lo) / scale, 0, 1)
            yn = np.clip((y - lo) / scale, 0, 1)

            correlations.append(corr(xn, yn))

            gx = np.hypot(sobel(xn, axis=0), sobel(xn, axis=1))
            gy = np.hypot(sobel(yn, axis=0), sobel(yn, axis=1))

            edge_correlations.append(corr(gx, gy))

            clipped_low.append(float(np.mean(y < lo)))
            clipped_high.append(float(np.mean(y > hi)))
            dynamic_ranges.append(float(hi - lo))

        result.update({
            "status": "ok",
            "intensity_corr": float(np.nanmean(correlations)),
            "edge_corr": float(np.nanmean(edge_correlations)),
            "gt_clip_low": float(np.mean(clipped_low)),
            "gt_clip_high": float(np.mean(clipped_high)),
            "artifact_dynamic_range": float(np.mean(dynamic_ranges)),
        })

    except Exception as exc:
        result["status"] = f"error: {exc}"

    rows.append(result)

    if i % 100 == 0:
        print(f"Audited {i}/{len(df)} volumes", flush=True)

result_df = pd.DataFrame(rows)
result_df.to_csv(OUTPUT, index=False)

print("\n========== AUDIT SUMMARY ==========")
print("Volumes:", len(result_df))
print("\nStatus:")
print(result_df["status"].value_counts())

good = result_df[result_df["status"] == "ok"]

if len(good):
    print("\nQuality metrics:")
    print(good[[
        "intensity_corr",
        "edge_corr",
        "gt_clip_low",
        "gt_clip_high",
        "artifact_dynamic_range",
    ]].describe().to_string())

    print("\nLowest edge correlations:")
    print(good.nsmallest(10, "edge_corr")[
        ["volume_id", "split", "plane",
         "intensity_corr", "edge_corr"]
    ].to_string(index=False))

    print("\nHighest GT clipping:")
    good = good.copy()
    good["total_gt_clipping"] = (
        good["gt_clip_low"] + good["gt_clip_high"]
    )
    print(good.nlargest(10, "total_gt_clipping")[
        ["volume_id", "split", "plane",
         "gt_clip_low", "gt_clip_high"]
    ].to_string(index=False))

print("\nSaved:", OUTPUT)
