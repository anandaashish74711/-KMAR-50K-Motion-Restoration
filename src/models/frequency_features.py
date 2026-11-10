import torch
from torch import nn
import torch.nn.functional as F


class SobelFeatures(nn.Module):
    def __init__(self):
        super().__init__()

        kx = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32) / 8.0

        ky = kx.T.contiguous()

        kernels = torch.stack([kx, ky]).unsqueeze(1)
        self.register_buffer("kernels", kernels)

    def forward(self, x):
        # [B, 1, H, W] -> [B, 2, H, W]
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(x, self.kernels)


class HaarFeatures(nn.Module):
    def __init__(self, levels=2):
        super().__init__()
        self.levels = levels

        low = torch.tensor([1., 1.]) / (2 ** 0.5)
        high = torch.tensor([1., -1.]) / (2 ** 0.5)

        filters = torch.stack([
            torch.outer(low, low),
            torch.outer(low, high),
            torch.outer(high, low),
            torch.outer(high, high)
        ]).unsqueeze(1)

        self.register_buffer("filters", filters)

    def forward(self, x):
        # Extract LH, HL, HH bands at multiple scales.
        features = []
        current = x
        target_size = x.shape[-2:]

        for _ in range(self.levels):
            h, w = current.shape[-2:]
            if h % 2 or w % 2:
                current = F.pad(
                    current, (0, w % 2, 0, h % 2),
                    mode="replicate"
                )

            bands = F.conv2d(
                current, self.filters, stride=2
            )

            ll = bands[:, 0:1]
            details = bands[:, 1:4]

            details = F.interpolate(
                details,
                size=target_size,
                mode="bilinear",
                align_corners=False
            )
            features.append(details)
            current = ll

        return torch.cat(features, dim=1)


class LearnedGating(nn.Module):
    def __init__(self, in_channels, hidden=32):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        weights = self.gate(features)
        return features * weights
