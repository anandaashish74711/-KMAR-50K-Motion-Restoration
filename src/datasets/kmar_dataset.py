import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


from src.datasets.normalization import normalize_pair

class KMARDataset(Dataset):
    def __init__(self, manifest_path, split="train", size=256, normalization="artifact_only"):
        self.normalization = normalization
        df = pd.read_csv(manifest_path)
        df = df[
            (df["split"] == split) &
            (df["geometry_compatible"] == True)
        ].reset_index(drop=True)

        self.size = size
        self.samples = []

        for row in df.itertuples():
            shape = json.loads(row.artifact_shape)

            for z in range(shape[2]):
                self.samples.append((
                    row.artifact_path,
                    row.ground_truth_path,
                    z
                ))

        # Cache only the most recently accessed volume pair.
        self.cached_key = None
        self.cached_pair = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        art_path, gt_path, z = self.samples[idx]
        key = (art_path, gt_path)

        if self.cached_key != key:
            art = nib.load(art_path).get_fdata(dtype=np.float32)
            gt = nib.load(gt_path).get_fdata(dtype=np.float32)
            self.cached_pair = (art, gt)
            self.cached_key = key

        art, gt = self.cached_pair

        x = np.asarray(art[:, :, z], dtype=np.float32)
        y = np.asarray(gt[:, :, z], dtype=np.float32)

        # Use the artifact image to determine a shared intensity scale.
        x, y = normalize_pair(
            x, y, mode=self.normalization
        )

        x = torch.from_numpy(x.astype(np.float32))[None, None]
        y = torch.from_numpy(y.astype(np.float32))[None, None]

        x = F.interpolate(
            x, size=(self.size, self.size),
            mode="bilinear", align_corners=False
        )[0]

        y = F.interpolate(
            y, size=(self.size, self.size),
            mode="bilinear", align_corners=False
        )[0]

        return {
            "artifact": x,
            "ground_truth": y,
            "slice_index": z,
        }
