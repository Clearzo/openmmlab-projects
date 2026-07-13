"""Linear-probe semantic segmentation on PASTIS-R with UniverSat.

Run from the MMSegmentation repository root::

    python projects/universat/PASTIS_R/tools/linear_probe.py \
        projects/universat/PASTIS_R/configs/linear_probe_universat_pastisr.py

The script:
  1. Loads a frozen UniverSat backbone (HuggingFace Hub or local checkpoint).
  2. Extracts per-patch features for train/val/test splits.
  3. Trains a LayerNorm + Linear probe head for each (lr, weight_decay) combo.
  4. Reports best val mIoU and corresponding test mIoU / micro-IoU.
"""

import argparse
import importlib.util
import itertools
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader

# Allow running from the UniverSat repo tree.
# This file lives at ``universat_run/mmseg_projects/PASTIS_R/tools/linear_probe.py``;
# ``PROJECT_ROOT`` points to ``universat_run/`` and ``MMSEG_PROJECTS`` points to
# ``universat_run/mmseg_projects/`` where ``universat/`` and ``PASTIS_R/`` are
# placed side-by-side as MMSegmentation custom projects.
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # universat_run/
MMSEG_PROJECTS = PROJECT_ROOT / "mmseg_projects"
sys.path.insert(0, str(MMSEG_PROJECTS))

from universat.models.backbones.universat import UniverSatBackbone
from PASTIS_R.datasets.pastisr_dataset import PASTISRDataset
from PASTIS_R.models.linear_probe_head import (
    BatchedLayerNormLinearProbes,
    LayerNormLinearClassifier,
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="UniverSat PASTIS-R linear probe")
    parser.add_argument("config", help="config python file path")
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none")
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def load_config(config_path: str):
    """Load a simple Python config (key = value) as a module."""
    spec = importlib.util.spec_from_file_location("cfg", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def build_backbone(cfg):
    """Build UniverSat backbone and load pretrained weights."""
    backbone = UniverSatBackbone(
        modalities=cfg.modalities,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        patch_size=cfg.patch_size,
        output_grid=cfg.output_grid,
        gating=cfg.gating,
        init_cfg=dict(type="Pretrained", checkpoint=cfg.checkpoint),
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


@torch.no_grad()
def extract_features(backbone, dataloader, device, patch_size=1):
    """Extract backbone features and labels.

    Args:
        backbone: frozen UniverSat backbone returning ``[(B, C, H, W)]``.
        dataloader: PASTIS-R dataloader.
        device: torch device.
        patch_size: side length (in label pixels) of the square region predicted
            by each output token. For PASTIS-R 128x128 labels with 128x128
            output tokens this is 1.
    """
    logits_list, labels_list = [], []

    for batch in dataloader:
        x = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        feats = backbone(x)[0]  # (B, C, H, W)

        B, C, H, W = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(B * H * W, C)

        labels = batch["label"]  # (B, H_img, W_img)
        if labels.ndim == 3:
            labels = rearrange(
                labels,
                "b (hp p1) (wp p2) -> b (hp wp) (p1 p2)",
                p1=patch_size,
                p2=patch_size,
            )
        labels = labels.reshape(-1, patch_size * patch_size)

        logits_list.append(feats.cpu().to(torch.float16))
        labels_list.append(labels.cpu().to(torch.int8))

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return logits, labels


# ---------------------------------------------------------------------------
# Linear probe training / evaluation
# ---------------------------------------------------------------------------

def evaluate_seg(logits, labels, probes, num_classes, patch_area, device):
    """Evaluate mIoU and micro-IoU for a batch of probe heads."""
    probes.eval()
    heads = len(probes.heads)
    conf = torch.zeros((heads, num_classes, num_classes), dtype=torch.float64, device=device)

    logits = logits.to(device)
    labels = labels.to(device)

    with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=torch.bfloat16):
        out = probes(logits)  # (H, N, num_classes * patch_area)

    out = out.view(heads, logits.shape[0], num_classes, patch_area)
    preds = out.argmax(dim=2)  # (H, N, patch_area)
    labels = labels.long()

    for h in range(heads):
        pred_h = preds[h].reshape(-1).long()
        label_h = labels.reshape(-1)
        valid = label_h != -1
        if not valid.any():
            continue
        y = label_h[valid]
        p = pred_h[valid].clamp(min=0, max=num_classes - 1)
        bins = torch.bincount(y * num_classes + p, minlength=num_classes * num_classes).to(torch.float64)
        conf[h] += bins.view(num_classes, num_classes)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(conf, op=torch.distributed.ReduceOp.SUM)

    miou_list, micro_iou_list = [], []
    for h in range(heads):
        cm = conf[h]
        inter = torch.diag(cm)
        union = cm.sum(dim=1) + cm.sum(dim=0) - inter
        valid = union > 0
        miou_h = (inter[valid] / (union[valid] + 1e-8)).mean() if valid.any() else torch.tensor(0.0)
        micro_h = inter.sum() / (union.sum() + 1e-8)
        miou_list.append(float(miou_h.item()))
        micro_iou_list.append(float(micro_h.item()))
    return miou_list, micro_iou_list


def train_probe(
    train_logits,
    train_labels,
    val_logits,
    val_labels,
    test_logits,
    test_labels,
    cfg,
    device,
):
    """Train linear probe heads and select best by validation mIoU."""
    in_features = train_logits.shape[-1]
    patch_area = cfg.patch_size_px * cfg.patch_size_px
    out_dim = cfg.num_classes * patch_area

    specs = list(itertools.product(cfg.lr_list, cfg.weight_decay_list))
    heads = [LayerNormLinearClassifier(in_features, out_dim) for _ in specs]
    container = BatchedLayerNormLinearProbes(heads).to(device)

    param_groups = {}
    for (lr, wd), head in zip(specs, heads):
        for name, p in head.named_parameters():
            if not p.requires_grad:
                continue
            key = (float(lr), 0.0 if "bias" in name else float(wd))
            if key not in param_groups:
                param_groups[key] = {"params": [], "base_lr": float(lr), "weight_decay": key[1]}
            param_groups[key]["params"].append(p)
    optimizer = torch.optim.AdamW(
        [{"params": g["params"], "lr": 0.0, "weight_decay": g["weight_decay"], "base_lr": g["base_lr"]} for g in param_groups.values()],
        betas=(0.9, 0.95),
    )

    train_ds = torch.utils.data.TensorDataset(train_logits, train_labels)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_ds = torch.utils.data.TensorDataset(val_logits, val_labels)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_ds = torch.utils.data.TensorDataset(test_logits, test_labels)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = cfg.max_epochs * steps_per_epoch
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    best_miou = torch.full((len(heads),), -float("inf"))
    best_state = [None] * len(heads)

    for epoch in range(cfg.max_epochs):
        container.train()
        for step, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=torch.bfloat16):
                logits_all = container(xb)  # (H, B, out_dim)
                losses = []
                for h in range(len(heads)):
                    logits_h = logits_all[h].view(xb.shape[0], cfg.num_classes, patch_area)
                    losses.append(ce_loss(logits_h, yb.long()))
                loss = torch.stack(losses).mean()
            loss.backward()
            rel_lr = 0.5 * (1.0 + math.cos(math.pi * (epoch * steps_per_epoch + step) / max(1, total_steps)))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * rel_lr
            optimizer.step()

        val_miou, _ = evaluate_seg(val_logits, val_labels, container, cfg.num_classes, patch_area, device)
        val_miou_t = torch.tensor(val_miou)
        improved = val_miou_t > best_miou
        for h in range(len(heads)):
            if improved[h]:
                best_miou[h] = val_miou_t[h]
                head = container.heads[h]
                best_state[h] = {
                    "linear_weight": head.linear.weight.detach().cpu().clone(),
                    "linear_bias": head.linear.bias.detach().cpu().clone(),
                    "ln_weight": head.ln.weight.detach().cpu().clone(),
                    "ln_bias": head.ln.bias.detach().cpu().clone(),
                }
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{cfg.max_epochs}, best val mIoU: {best_miou.max().item():.4f}")

    for h, state in enumerate(best_state):
        if state is None:
            continue
        head = container.heads[h]
        head.linear.weight.data.copy_(state["linear_weight"].to(device))
        head.linear.bias.data.copy_(state["linear_bias"].to(device))
        head.ln.weight.data.copy_(state["ln_weight"].to(device))
        head.ln.bias.data.copy_(state["ln_bias"].to(device))

    test_miou, test_micro_iou = evaluate_seg(test_logits, test_labels, container, cfg.num_classes, patch_area, device)
    best_h = int(best_miou.argmax().item())
    print(f"Best head: lr={specs[best_h][0]}, wd={specs[best_h][1]}")
    print(f"Best val mIoU: {best_miou[best_h].item():.4f}")
    print(f"Test mIoU:     {test_miou[best_h]:.4f}")
    print(f"Test micro-IoU:{test_micro_iou[best_h]:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.launcher == "pytorch":
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = build_backbone(cfg).to(device)

    def make_loader(split, folds):
        ds = PASTISRDataset(
            data_root=cfg.data_root,
            modalities=cfg.modalities,
            split=split,
            folds=folds,
            reference_date=cfg.reference_date,
            nb_split=cfg.nb_split,
            num_classes=cfg.num_classes,
            temporal_dropout=cfg.temporal_dropout if split == "train" else 0,
            normalize=cfg.normalize,
            norm_path=cfg.norm_path,
        )
        return DataLoader(
            ds,
            batch_size=cfg.global_batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.num_workers,
            collate_fn=PASTISRDataset.collate_fn,
            drop_last=False,
        )

    train_loader = make_loader("train", cfg.train_folds)
    val_loader = make_loader("val", cfg.val_folds)
    test_loader = make_loader("test", cfg.test_folds)

    print("Extracting train features...")
    train_logits, train_labels = extract_features(backbone, train_loader, device, patch_size=cfg.patch_size_px)
    print("Extracting val features...")
    val_logits, val_labels = extract_features(backbone, val_loader, device, patch_size=cfg.patch_size_px)
    print("Extracting test features...")
    test_logits, test_labels = extract_features(backbone, test_loader, device, patch_size=cfg.patch_size_px)

    print(f"Train features: {train_logits.shape}, labels: {train_labels.shape}")

    train_probe(
        train_logits,
        train_labels,
        val_logits,
        val_labels,
        test_logits,
        test_labels,
        cfg,
        device,
    )

    if args.launcher == "pytorch":
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
