import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from skimage.metrics import structural_similarity

df = pd.read_csv("manifests/dataset_manifest.csv")
df = df[df["split"].isin(["train", "val"])]

out = Path("outputs/alignment_audit")
out.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
selected = pd.concat([
    part.sample(
        n=min(5, len(part)),
        random_state=42
    )
    for _, part in df.groupby("split")
])

rows = []

for _, row in selected.iterrows():
    a = nib.load(row["artifact_path"]).get_fdata(dtype=np.float32)
    g = nib.load(row["ground_truth_path"]).get_fdata(dtype=np.float32)

    z = a.shape[2] // 2
    x = a[:, :, z]
    y = g[:, :, z]

    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.all():
        rows.append({
            "volume_id": row["volume_id"],
            "split": row["split"],
            "status": "nonfinite",
        })
        continue

    lo, hi = np.percentile(x, [1, 99])
    scale = max(float(hi - lo), 1e-6)

    xn = np.clip((x - lo) / scale, 0, 1)
    yn = np.clip((y - lo) / scale, 0, 1)

    score = structural_similarity(
        xn, yn, data_range=1.0
    )

    clipped_low = float(np.mean(y < lo))
    clipped_high = float(np.mean(y > hi))

    rows.append({
        "volume_id": row["volume_id"],
        "split": row["split"],
        "status": "ok",
        "middle_slice_ssim": score,
        "gt_clipped_low_fraction": clipped_low,
        "gt_clipped_high_fraction": clipped_high,
    })

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, image, title in zip(
        axes,
        [xn, yn, np.abs(xn - yn)],
        ["Artifact", "Ground truth", "Absolute difference"]
    ):
        ax.imshow(image.T, cmap="gray", origin="lower",
                  vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(row["volume_id"])
    fig.tight_layout()

    safe_id = row["volume_id"].replace("/", "_")
    fig.savefig(out / f"{safe_id}.png", dpi=120)
    plt.close(fig)

result = pd.DataFrame(rows)
result.to_csv(out / "audit.csv", index=False)

print("Pairs inspected:", len(result))
print("\nSummary:")
print(result.describe(include="all").to_string())
print("\nSaved images:", out)
