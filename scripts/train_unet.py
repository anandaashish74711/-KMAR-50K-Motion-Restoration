import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.kmar_dataset import KMARDataset
from src.models.unet import UNet
from src.models.frequency_unet import FrequencyGuidedUNet


def ssim(x, y):
    # Differentiable, single-scale SSIM for normalized grayscale MRI.
    k = 7
    sigma = 1.5
    coords = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).view(1, 1, k, k)

    def blur(z):
        return F.conv2d(F.pad(z, (3, 3, 3, 3), mode="reflect"), kernel)

    ux, uy = blur(x), blur(y)
    vx = (blur(x*x) - ux*ux).clamp_min(0)
    vy = (blur(y*y) - uy*uy).clamp_min(0)
    cov = blur(x*y) - ux*uy

    c1, c2 = 0.01**2, 0.03**2
    score = ((2*ux*uy+c1)*(2*cov+c2)) / (
        (ux*ux+uy*uy+c1)*(vx+vy+c2)
    )
    return score.mean()


def compute_metrics(pred, target):
    mse = F.mse_loss(pred, target).clamp_min(1e-10)
    psnr = -10 * torch.log10(mse)
    return psnr.item(), ssim(pred, target).item()


def run_epoch(model, loader, optimizer, device, max_steps):
    training = optimizer is not None
    model.train(training)

    total_loss = total_psnr = total_ssim = 0.0
    steps = 0

    for batch in loader:
        x = batch["artifact"].to(device)
        y = batch["ground_truth"].to(device)

        with torch.set_grad_enabled(training):
            pred = model(x)
            l1 = F.l1_loss(pred, y)
            score = ssim(pred, y)
            loss = l1 + 0.2 * (1 - score)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            psnr, _ = compute_metrics(pred.detach(), y)
            total_loss += loss.item()
            total_psnr += psnr
            total_ssim += score.item()

        steps += 1
        if max_steps and steps >= max_steps:
            break

    if steps == 0:
        raise RuntimeError("No batches available")

    return (
        total_loss / steps,
        total_psnr / steps,
        total_ssim / steps
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--base", type=int, default=16)
    p.add_argument(
        "--model",
        choices=["unet", "frequency_unet"],
        default="unet"
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-train-steps", type=int, default=2)
    p.add_argument("--max-val-steps", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="checkpoints/unet_smoke.pt")
    p.add_argument(
        "--normalization",
        choices=["artifact_only", "independent"],
        default="artifact_only"
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import random
    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    print("Device:", device, flush=True)

    manifest = "manifests/dataset_manifest.csv"
    train_ds = KMARDataset(manifest, split="train", normalization=args.normalization)
    val_ds = KMARDataset(manifest, split="val", normalization=args.normalization)

    # Sequential access is intentional: the current Dataset caches
    # only one volume pair, so random access would be very expensive.
    from src.data.samplers import VolumeShuffleSampler

    train_sampler = VolumeShuffleSampler(train_ds, seed=args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0
    )

    model = (
        UNet(base=args.base)
        if args.model == "unet"
        else FrequencyGuidedUNet(base=args.base)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history_path = output.with_suffix(".csv")

    best_loss = float("inf")

    with history_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_psnr", "train_ssim",
            "val_loss", "val_psnr", "val_ssim"
        ])

        for epoch in range(1, args.epochs + 1):
            train_sampler.set_epoch(epoch)
            tr = run_epoch(
                model, train_loader, optimizer, device,
                args.max_train_steps
            )
            va = run_epoch(
                model, val_loader, None, device,
                args.max_val_steps
            )

            writer.writerow([epoch, *tr, *va])
            f.flush()

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={tr[0]:.4f} | "
                f"val_loss={va[0]:.4f} | "
                f"val_PSNR={va[1]:.2f} | "
                f"val_SSIM={va[2]:.4f}",
                flush=True
            )

            if va[0] < best_loss:
                best_loss = va[0]
                torch.save({
                    "epoch": epoch,
                    "normalization": args.normalization,
                    "seed": args.seed,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": va[0],
                    "base": args.base
                }, output)

    print("Checkpoint:", output)
    print("History:", history_path)


if __name__ == "__main__":
    main()
