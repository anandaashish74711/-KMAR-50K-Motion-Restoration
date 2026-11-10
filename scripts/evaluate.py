import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.kmar_dataset import KMARDataset
from src.models.unet import UNet
from src.models.frequency_unet import FrequencyGuidedUNet
from scripts.train_unet import ssim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["unet", "frequency_unet"],
                   required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["train", "val", "test"],
                   default="val")
    p.add_argument("--max-slices", type=int, default=0)
    p.add_argument("--output", default="outputs/evaluation.csv")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(
        args.checkpoint, map_location="cpu",
        weights_only=False
    )

    base = ckpt.get("base", 16)

    model = (
        UNet(base=base)
        if args.model == "unet"
        else FrequencyGuidedUNet(base=base)
    ).to(device)

    normalization = ckpt.get("normalization", "artifact_only")
    if normalization not in ("artifact_only", "independent"):
        raise ValueError(
            f"Unsupported checkpoint normalization: {normalization}"
        )
    print(f"Checkpoint normalization: {normalization}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = KMARDataset(
        "manifests/dataset_manifest.csv",
        split=args.split,
        normalization=normalization
    )

    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0
    )

    volume_metrics = defaultdict(list)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if args.max_slices and idx >= args.max_slices:
                break

            x = batch["artifact"].to(device)
            y = batch["ground_truth"].to(device)
            pred = model(x)

            mse = F.mse_loss(pred, y).clamp_min(1e-10)
            psnr = (-10 * torch.log10(mse)).item()
            score = ssim(pred, y).item()

            volume_id = ds.samples[idx][0]
            input_mse = F.mse_loss(x, y).clamp_min(1e-10)
            input_psnr = (-10 * torch.log10(input_mse)).item()
            input_ssim = ssim(x, y).item()

            volume_metrics[volume_id].append(
                (psnr, score, input_psnr, input_ssim)
            )

            if (idx + 1) % 100 == 0:
                print("Evaluated slices:", idx + 1, flush=True)

    if not volume_metrics:
        raise RuntimeError("No slices evaluated")

    rows = []
    for volume_id, metrics in volume_metrics.items():
        mean_psnr = sum(m[0] for m in metrics) / len(metrics)
        mean_ssim = sum(m[1] for m in metrics) / len(metrics)
        input_psnr = sum(m[2] for m in metrics) / len(metrics)
        input_ssim = sum(m[3] for m in metrics) / len(metrics)
        rows.append((
            volume_id, len(metrics), mean_psnr, mean_ssim,
            input_psnr, input_ssim,
            mean_psnr - input_psnr,
            mean_ssim - input_ssim
        ))

    mean_psnr = sum(r[2] for r in rows) / len(rows)
    mean_ssim = sum(r[3] for r in rows) / len(rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "volume_id", "slices", "psnr", "ssim",
            "input_psnr", "input_ssim",
            "delta_psnr", "delta_ssim"
        ])
        writer.writerows(rows)

    print("Model:", args.model)
    print("Split:", args.split)
    print("Volumes evaluated:", len(rows))
    print("Mean volume PSNR:", round(mean_psnr, 4))
    print("Mean volume SSIM:", round(mean_ssim, 4))
    print("Mean input PSNR:",
          round(sum(r[4] for r in rows) / len(rows), 4))
    print("Mean input SSIM:",
          round(sum(r[5] for r in rows) / len(rows), 4))
    print("Mean PSNR improvement:",
          round(sum(r[6] for r in rows) / len(rows), 4))
    print("Mean SSIM improvement:",
          round(sum(r[7] for r in rows) / len(rows), 4))
    print("Saved:", output)


if __name__ == "__main__":
    main()
