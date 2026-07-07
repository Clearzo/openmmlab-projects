"""PASTIS-R dataset for linear-probe semantic segmentation.

The data layout follows the original PASTIS-R dataset:
    - metadata.geojson      : patch metadata (ID_PATCH, Fold, dates, ...)
    - DATA_S2/              : Sentinel-2 time-series .npy files
    - DATA_S1A/             : Sentinel-1 ascending time-series .npy files
    - ANNOTATIONS/TARGET_*  : per-pixel semantic labels .npy

This implementation is a standalone, minimal adaptation of
``UniverSat/src/data/Pastis.py`` for the MMSegmentation project.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


S2_BANDS = [0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.833, 0.865, 1.61, 2.19]
S1_BANDS = ["VV", "VH", "Ratio_VV_VH"]


def _prepare_dates(date_dict, reference_date):
    """Convert date strings to day-of-year relative to reference_date."""
    if isinstance(date_dict, str):
        date_dict = json.loads(date_dict)
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
            datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
            - reference_date
        ).days
    )
    return torch.tensor(d.values, dtype=torch.long)


def _split_image(image, nb_split, part):
    """Spatially split a large tile into quadrants."""
    if nb_split == 1:
        return image
    i1 = part // nb_split
    i2 = part % nb_split
    h, w = image.shape[-2:]
    h_step = h // nb_split
    w_step = w // nb_split
    if image.ndim == 4:
        return image[..., i1 * h_step:(i1 + 1) * h_step, i2 * w_step:(i2 + 1) * w_step]
    if image.ndim == 3:
        return image[:, i1 * h_step:(i1 + 1) * h_step, i2 * w_step:(i2 + 1) * w_step]
    return image[i1 * h_step:(i1 + 1) * h_step, i2 * w_step:(i2 + 1) * w_step]


class PASTISRDataset(Dataset):
    """PASTIS-R dataset for linear-probe semantic segmentation.

    Args:
        data_root (str): Path to PASTIS-R directory.
        modalities (list[str]): Modalities to load, e.g. ["s2", "s1"].
        split (str): One of "train", "val", "test".
        folds (list[int]): Fold indices to use. If None, uses split defaults.
        reference_date (str): Reference date for temporal encoding.
        nb_split (int): Split each tile into nb_split x nb_split sub-tiles.
        num_classes (int): Number of semantic classes (default 20 for PASTIS-R).
        temporal_dropout (int): Max number of timestamps kept per series.
            0 disables dropout. Only applied to train/val.
        normalize (bool): Whether to apply channel-wise normalization.
        norm_path (str | None): Directory with .npy normalization statistics.
    """

    def __init__(
        self,
        data_root: str,
        modalities: List[str] = ("s2", "s1"),
        split: str = "train",
        folds: Optional[List[int]] = None,
        reference_date: str = "2018-01-01",
        nb_split: int = 1,
        num_classes: int = 20,
        temporal_dropout: int = 0,
        normalize: bool = True,
        norm_path: Optional[str] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.modalities = list(modalities)
        self.split = split
        self.nb_split = nb_split
        self.num_classes = num_classes
        self.temporal_dropout = temporal_dropout if temporal_dropout > 0 else np.inf
        self.normalize = normalize
        self.norm_path = norm_path or data_root

        self.reference_date = datetime(*map(int, reference_date.split("-")))

        self.meta_patch = gpd.read_file(os.path.join(data_root, "metadata.geojson"))
        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )

        self.norm = self._load_norm()

    def _load_norm(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load per-modality mean/std for normalization."""
        norm = {}
        for mod in self.modalities:
            mean_path = os.path.join(self.norm_path, f"NORM_{mod.upper()}_mean.npy")
            std_path = os.path.join(self.norm_path, f"NORM_{mod.upper()}_std.npy")
            if os.path.exists(mean_path) and os.path.exists(std_path):
                mean = torch.from_numpy(np.load(mean_path)).float()
                std = torch.from_numpy(np.load(std_path)).float()
                norm[mod] = {"mean": mean, "std": std}
        return norm

    def _apply_norm(self, output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply stored normalization statistics."""
        if not self.normalize:
            return output
        for mod in self.modalities:
            if mod not in self.norm:
                continue
            mean = self.norm[mod]["mean"]
            std = self.norm[mod]["std"]
            tensor = output[mod]
            while mean.ndim < tensor.ndim:
                mean = mean.unsqueeze(-1)
            while std.ndim < tensor.ndim:
                std = std.unsqueeze(-1)
            output[mod] = (tensor - mean) / (std + 1e-8)
        return output

    def __len__(self):
        return len(self.meta_patch) * self.nb_split * self.nb_split

    def __getitem__(self, idx):
        line = self.meta_patch.iloc[idx // (self.nb_split * self.nb_split)]
        name = line["ID_PATCH"]
        part = idx % (self.nb_split * self.nb_split)

        label = torch.from_numpy(
            np.load(os.path.join(self.data_root, "ANNOTATIONS", f"TARGET_{name}.npy"))[0].astype(np.int32)
        )
        label = _split_image(label, self.nb_split, part).long()
        output = {"label": label, "name": name}

        for modality in self.modalities:
            if modality == "spot":
                path = os.path.join(
                    self.data_root,
                    "DATA_SPOT/PASTIS_SPOT6_RVB_1M00_2019",
                    f"SPOT6_RVB_1M00_2019_{name}.tif",
                )
                with rasterio.open(path) as f:
                    output["spot"] = _split_image(torch.FloatTensor(f.read()), self.nb_split, part)
            elif modality in ("s2", "s1"):
                modality_name = "s2" if modality == "s2" else "s1a"
                path = os.path.join(
                    self.data_root,
                    f"DATA_{modality_name.upper()}",
                    f"{modality_name.upper()}_{name}.npy",
                )
                images = _split_image(
                    torch.from_numpy(np.load(path)), self.nb_split, part
                ).to(torch.float32)
                output[modality] = images
                output[f"{modality}_dates"] = _prepare_dates(
                    line[f"dates-{modality_name.upper()}"], self.reference_date
                )
                N = len(images)
                if self.split in ("train", "val") and N > self.temporal_dropout:
                    indices = torch.randperm(N)[: int(self.temporal_dropout)]
                    output[modality] = output[modality][indices]
                    output[f"{modality}_dates"] = output[f"{modality}_dates"][indices]
            else:
                raise ValueError(f"Unsupported modality: {modality}")

        output = self._apply_norm(output)
        return output

    @staticmethod
    def collate_fn(batch):
        """Custom collate for variable-length time series."""
        keys = list(batch[0].keys())
        output = {}
        for key in ["s2", "s1"]:
            if key in keys:
                tensors = [x[key] for x in batch]
                max_t = max(t.size(0) for t in tensors)
                output[key] = torch.stack([
                    torch.nn.functional.pad(t, (0, 0, 0, 0, 0, 0, 0, max_t - t.size(0)))
                    for t in tensors
                ], dim=0).float()
                date_key = f"{key}_dates"
                date_tensors = [x[date_key] for x in batch]
                output[date_key] = torch.stack([
                    torch.nn.functional.pad(t, (0, max_t - t.size(0)))
                    for t in date_tensors
                ], dim=0).long()
                keys.remove(key)
                keys.remove(date_key)
        if "name" in keys:
            output["name"] = [x["name"] for x in batch]
            keys.remove("name")
        for key in keys:
            output[key] = torch.stack([x[key] for x in batch])
        return output
