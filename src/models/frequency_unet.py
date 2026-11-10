import torch
from torch import nn

from .frequency_features import (
    SobelFeatures,
    HaarFeatures,
    LearnedGating
)
from .unet import UNet


class FrequencyGuidedUNet(nn.Module):
    def __init__(self, base=16, wavelet_levels=2):
        super().__init__()

        self.sobel = SobelFeatures()
        self.haar = HaarFeatures(levels=wavelet_levels)

        feature_channels = 1 + 2 + 3 * wavelet_levels

        self.gating = LearnedGating(
            in_channels=feature_channels,
            hidden=32
        )

        # U-Net receives fused multichannel features.
        self.unet = UNet(
            in_channels=feature_channels,
            base=base
        )

        # UNet uses residual addition internally. Its input
        # must be a single-channel MRI for restoration, so we
        # replace the output head behavior below.
        self.unet.output = nn.Conv2d(base, 1, 1)
        nn.init.zeros_(self.unet.output.weight)
        nn.init.zeros_(self.unet.output.bias)

    def forward(self, x):
        sobel = self.sobel(x)
        haar = self.haar(x)

        features = torch.cat(
            [x, sobel, haar], dim=1
        )

        fused = self.gating(features)

        # The existing UNet adds its input to the output.
        # Instead, call its encoder/decoder blocks directly.
        e1 = self.unet.enc1(fused)
        e2 = self.unet.enc2(self.unet.pool(e1))
        e3 = self.unet.enc3(self.unet.pool(e2))
        b = self.unet.bottleneck(self.unet.pool(e3))

        import torch.nn.functional as F

        d3 = F.interpolate(
            b, size=e3.shape[-2:],
            mode="bilinear", align_corners=False
        )
        d3 = self.unet.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(
            d3, size=e2.shape[-2:],
            mode="bilinear", align_corners=False
        )
        d2 = self.unet.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(
            d2, size=e1.shape[-2:],
            mode="bilinear", align_corners=False
        )
        d1 = self.unet.dec1(torch.cat([d1, e1], dim=1))

        correction = self.unet.output(d1)

        return torch.clamp(x + correction, 0, 1)
