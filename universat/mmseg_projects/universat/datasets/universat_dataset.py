"""UniverSat-compatible segmentation dataset for MMSegmentation."""

import os

import mmcv
import numpy as np
from mmseg.datasets import CustomDataset
from mmseg.datasets.builder import DATASETS


@DATASETS.register_module()
class UniverSatSegDataset(CustomDataset):
    """Custom segmentation dataset with multiple modality rasters.

    The annotation file is a JSON list of dicts::

        [
          {
            "filenames": {
              "s2": "s2/xxx.tif",
              "s1": "s1/xxx.tif"
            },
            "ann": {"seg_map": "masks/xxx.png"},
            "height": 360,
            "width": 360
          },
          ...
        ]

    Args:
        modalities (list[str]): List of modality names. Must match the backbone.
        *args, **kwargs: forwarded to ``CustomDataset``.
    """

    def __init__(self, modalities, *args, **kwargs):
        self.modalities = modalities
        super().__init__(*args, **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        """Load annotations from a JSON file or directory structure."""
        if split is not None:
            import json
            with open(split, "r") as f:
                samples = json.load(f)
            data_infos = []
            for sample in samples:
                info = {
                    "filename": sample["filenames"],
                    "height": sample.get("height", 0),
                    "width": sample.get("width", 0),
                }
                if "ann" in sample:
                    info["ann"] = dict(seg_map=sample["ann"]["seg_map"])
                data_infos.append(info)
            return data_infos

        # Fallback to directory-based discovery.
        return super().load_annotations(img_dir, img_suffix, ann_dir, seg_map_suffix, split)

    def get_seg_annotation(self, idx):
        """Get segmentation annotation by index."""
        ann_info = self.data_infos[idx].get("ann", {})
        seg_map = ann_info.get("seg_map", None)
        if seg_map is None:
            return None
        seg_map = os.path.join(self.ann_dir, seg_map)
        return mmcv.imread(seg_map, flag="unchanged").astype(np.uint8)
