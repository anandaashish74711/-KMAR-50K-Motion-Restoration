import pandas as pd
import numpy as np
import nibabel as nib
from collections import Counter

df = pd.read_csv("manifests/dataset_manifest.csv")
bad = df[df["split"] == "geometry_review"]

rows = []

for _, row in bad.iterrows():
    a = nib.load(row["artifact_path"])
    g = nib.load(row["ground_truth_path"])

    a_spacing = np.array(a.header.get_zooms()[:3])
    g_spacing = np.array(g.header.get_zooms()[:3])

    a_shape = np.array(a.shape[:3])
    g_shape = np.array(g.shape[:3])

    a_corners = np.array(np.meshgrid(
        [0, a_shape[0]-1],
        [0, a_shape[1]-1],
        [0, a_shape[2]-1],
        indexing="ij"
    )).reshape(3, -1).T

    g_corners = np.array(np.meshgrid(
        [0, g_shape[0]-1],
        [0, g_shape[1]-1],
        [0, g_shape[2]-1],
        indexing="ij"
    )).reshape(3, -1).T

    a_world = nib.affines.apply_affine(a.affine, a_corners)
    g_world = nib.affines.apply_affine(g.affine, g_corners)

    a_min, a_max = a_world.min(axis=0), a_world.max(axis=0)
    g_min, g_max = g_world.min(axis=0), g_world.max(axis=0)

    intersection = np.maximum(
        0, np.minimum(a_max, g_max) - np.maximum(a_min, g_min)
    )

    union_extent = np.maximum(a_max, g_max) - np.minimum(a_min, g_min)

    intersection_volume = np.prod(intersection)
    union_volume = (
        np.prod(a_max - a_min)
        + np.prod(g_max - g_min)
        - intersection_volume
    )

    bbox_iou = (
        intersection_volume / union_volume
        if union_volume > 0 else 0
    )

    rows.append({
        "volume_id": row["volume_id"],
        "group_id": row["group_id"],
        "plane": row["plane"],
        "artifact_shape": str(tuple(a.shape)),
        "gt_shape": str(tuple(g.shape)),
        "same_shape": a.shape == g.shape,
        "artifact_spacing": str(tuple(a_spacing)),
        "gt_spacing": str(tuple(g_spacing)),
        "artifact_orientation": "".join(
            nib.aff2axcodes(a.affine)
        ),
        "gt_orientation": "".join(
            nib.aff2axcodes(g.affine)
        ),
        "affine_max_abs_diff": float(
            np.max(np.abs(a.affine - g.affine))
        ),
        "bbox_iou": bbox_iou,
    })

result = pd.DataFrame(rows)
result.to_csv("outputs/geometry_audit.csv", index=False)

print("Audited:", len(result))
print("\nSame shape:")
print(result["same_shape"].value_counts())

print("\nOrientation pairs:")
print(
    result.groupby([
        "artifact_orientation",
        "gt_orientation"
    ]).size().sort_values(ascending=False).head(10)
)

print("\nBounding-box IoU:")
print(result["bbox_iou"].describe())

print("\nAffine difference:")
print(result["affine_max_abs_diff"].describe())

print("\nLowest-overlap examples:")
print(
    result.nsmallest(10, "bbox_iou")[
        ["volume_id", "artifact_shape",
         "gt_shape", "bbox_iou"]
    ].to_string(index=False)
)

print("\nSaved: outputs/geometry_audit.csv")
