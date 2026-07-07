"""Linear probe heads for segmentation/regression on UniverSat features."""

import torch
import torch.nn as nn


class LayerNormLinearClassifier(nn.Module):
    """CAPI-style probe: LayerNorm + Linear."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.ln = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, out_features)
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.ln(x))


class BatchedLayerNormLinearProbes(nn.Module):
    """Vectorized forward over many probe heads.

    Returns stacked logits of shape ``[H, B, C]`` where H is the number of
    heads. This is used to sweep many (lr, weight_decay) combinations in
    parallel.
    """

    def __init__(self, heads: list[nn.Module]):
        super().__init__()
        if len(heads) == 0:
            raise ValueError("heads must be non-empty")
        self.heads = nn.ModuleList(heads)
        self.in_features = self.heads[0].linear.in_features
        self.out_features = self.heads[0].linear.out_features
        self.ln_eps = self.heads[0].ln.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        W = torch.stack([h.linear.weight for h in self.heads], dim=0)  # [H, C, D]
        b = torch.stack([h.linear.bias for h in self.heads], dim=0)   # [H, C]

        mean = x_fp32.mean(dim=-1, keepdim=True)
        var = x_fp32.var(dim=-1, unbiased=False, keepdim=True)
        x_hat = (x_fp32 - mean) / torch.sqrt(var + self.ln_eps)

        ln_weight = torch.stack([h.ln.weight for h in self.heads], dim=0)  # [H, D]
        ln_bias = torch.stack([h.ln.bias for h in self.heads], dim=0)      # [H, D]
        x_affine = x_hat.unsqueeze(0) * ln_weight[:, None, :] + ln_bias[:, None, :]
        return torch.bmm(x_affine, W.transpose(1, 2)) + b[:, None, :]
