import numpy as np
import pandas as pd
import nibabel as nib

df = pd.read_csv("manifests/dataset_manifest.csv")

cases = [
    "2021_8-XXX_MR1202101060308_sagittal_0.0",
    "2021_105-XX_MR1202104080076_transection_0.0",
    "2021_258-20210722004267_sagittal_0.0",
    "2021_33-XXX_MR1202102200156_coronal_0.0",
]

def normalize(x, lo, hi):
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)

for case in cases:
    row = df[df.volume_id == case].iloc[0]

    a = nib.load(row.artifact_path).get_fdata(dtype=np.float32)
    g = nib.load(row.ground_truth_path).get_fdata(dtype=np.float32)

    # Volume-level percentiles, rather than slice-level.
    al, ah = np.percentile(a, [1, 99])
    gl, gh = np.percentile(g, [1, 99])

    methods = {
        "artifact_only": (
            normalize(a, al, ah),
            normalize(g, al, ah)
        ),
        "shared_pair": (
            normalize(a, min(al, gl), max(ah, gh)),
            normalize(g, min(al, gl), max(ah, gh))
        ),
        "independent": (
            normalize(a, al, ah),
            normalize(g, gl, gh)
        ),
    }

    print("\n", case)

    for name, (x, y) in methods.items():
        print(
            f"{name:15s} "
            f"artifact_mean={x.mean():.4f} "
            f"gt_mean={y.mean():.4f} "
            f"gt_zero={(y == 0).mean():.4f} "
            f"gt_one={(y == 1).mean():.4f}"
        )
