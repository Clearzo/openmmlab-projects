"""PASTIS-R dataset for MMSegmentation.

This class adapts the original ``UniverSat/src/data/Pastis.py`` to the
MMSegmentation ``CustomDataset`` interface, enabling standard MMSeg training
and linear-probe experiments through config files.
"""

import json
import os
from datetime import datetime
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from mmseg.datasets import CustomDataset
from mmseg.datasets.builder import DATASETS


def _prepare_dates(date_dict, reference_date):
    """Convert date strings to days relative to reference_date."""
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


@DATASETS.register_module()
class PASTISRDataset(CustomDataset):
    """PASTIS-R dataset for MMSegmentation.

    Args:
        modalities (list[str]): Modalities to load, e.g. ["s2", "s1"].
        reference_date (str): Reference date for temporal encoding.
        temporal_dropout (int): Max number of timestamps kept per series
            during training/validation. 0 disables dropout.
        normalize (bool): Whether to apply stored normalization statistics.
        norm_path (str | None): Directory with NORM_*_mean.npy / NORM_*_std.npy.
        *args, **kwargs: forwarded to ``CustomDataset``.
    """

    CLASSES = (
        "Background",
        "Meadow",
        "Spring cereals",
        "Winter cereals",
        "Rapeseed",
        "Sunflower",
        "Soybeans",
        "Maize",
        "Beet",
        "Vegetables",
        "Orchards",
        "Vineyards",
        "Permanent grassland",
        "Forest",
        "Shrubland",
        "Water",
        "Urban fabric",
        "Rocks",
        "Built up",
        "Unknown",
    )

    PALETTE = None

    def __init__(
        self,
        modalities: List[str] = ("s2", "s1"),
        reference_date: str = "2018-01-01",
        temporal_dropout: int = 0,
        normalize: bool = True,
        norm_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        self.modalities = list(modalities)
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.temporal_dropout = temporal_dropout if temporal_dropout > 0 else np.inf
        self.normalize = normalize
        self.norm_path = norm_path
        super().__init__(*args, **kwargs)
        self.norm = self._load_norm()

    def _load_norm(self):
        """Load per-modality normalisation from ``NORM_{mod}_patch.json``.

        This mirrors ``src.data.utils.load_norm`` used by the original
        ``Pastis.py`` dataset.
        """
        norm = {}
        norm_path = self.norm_path or self.data_root
        if norm_path is None:
            return norm
        for mod in self.modalities:
            file_path = os.path.join(norm_path, f"NORM_{mod}_patch.json")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    vals = json.load(f)
                norm[mod] = (
                    torch.tensor(vals["mean"]).float(),
                    torch.tensor(vals["std"]).float(),
                )
        return norm

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        """Load PASTIS-R metadata from metadata.geojson and optional split folds."""
        meta_path = os.path.join(self.data_root, "metadata.geojson")
        meta = gpd.read_file(meta_path)

        if split is not None:
            with open(split, "r") as f:
                fold_info = json.load(f)
            folds = fold_info.get("folds", None)
            if folds is not None:
                meta = pd.concat([meta[meta["Fold"] == f] for f in folds])

        data_infos = []
        for _, line in meta.iterrows():
            id_patch = int(line["ID_PATCH"])
            modality_paths = {}
            dates = {}
            for mod in self.modalities:
                mod_upper = "S2" if mod == "s2" else "S1A"
                modality_paths[mod] = os.path.join(
                    self.data_root,
                    f"DATA_{mod_upper}",
                    f"{mod_upper}_{id_patch}.npy",
                )
                dates[mod] = _prepare_dates(
                    line[f"dates-{mod_upper}"], self.reference_date
                )

            data_infos.append({
                "filename": {"id_patch": id_patch},
                "modality_paths": modality_paths,
                "dates": dates,
                "ann": {"id_patch": id_patch},
            })
        return data_infos

    def get_seg_annotation(self, idx):
        """Load the dense semantic label map."""
        ann_info = self.data_infos[idx]["ann"]
        id_patch = ann_info["id_patch"]
        path = os.path.join(self.data_root, "ANNOTATIONS", f"TARGET_{id_patch}.npy")
        label = np.load(path)[0].astype(np.int64)
        return label

    def __getitem__(self, idx):
        """Inject dataset reference and index into the pipeline."""
        if not self.test_mode:
            return self.prepare_train_img(idx)
        return self.prepare_test_img(idx)

    def prepare_train_img(self, idx):
        info = self.data_infos[idx].copy()
        info["dataset"] = self
        info["idx"] = idx
        return self.pipeline(info)

    prepare_test_img = prepare_train_img
