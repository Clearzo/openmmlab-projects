"""Semantic segmentation head for UniverSat features."""

import torch
import torch.nn as nn

try:
    from mmcv.cnn import ConvModule
    HAS_MMCV = True
except ImportError:  # pragma: no cover
    HAS_MMCV = False

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class _PlainConvBlock(nn.Module):
    """Fallback conv block when mmcv is not available (standalone tests)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


@HEADS.register_module()
class UniverSatSegHead(BaseDecodeHead):
    """Lightweight segmentation head for UniverSat backbone outputs.

    UniverSat returns a single-scale feature map of shape ``(B, C, H, W)``
    where ``H x W`` corresponds to the requested ``output_grid``. This head
    applies a few convolutions and bilinear-upsamples to the target resolution.

    Args:
        in_channels (int): Number of input channels. Default 768 (Base).
        channels (int): Number of intermediate channels. Default 256.
        num_convs (int): Number of conv layers before classifier. Default 2.
        output_size (tuple | None): Target output size ``(H, W)``.
        *args, **kwargs: forwarded to ``BaseDecodeHead``.
    """

    def __init__(
        self,
        in_channels: int = 768,
        channels: int = 256,
        num_convs: int = 2,
        output_size=None,
        *args,
        **kwargs,
    ):
        super().__init__(in_channels, channels, *args, **kwargs)
        self.output_size = output_size

        convs = []
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else channels
            if HAS_MMCV:
                convs.append(
                    ConvModule(
                        in_ch,
                        channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    )
                )
            else:
                convs.append(_PlainConvBlock(in_ch, channels))
        self.convs = nn.Sequential(*convs)

    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (list[Tensor]): list of feature maps from backbone.

        Returns:
            Tensor: segmentation logits of shape ``(B, num_classes, H, W)``.
        """
        x = inputs[self.in_index]
        x = self.convs(x)
        x = self.cls_seg(x)
        if self.output_size is not None:
            x = torch.nn.functional.interpolate(
                x, size=self.output_size, mode="bilinear", align_corners=False
            )
        return x
