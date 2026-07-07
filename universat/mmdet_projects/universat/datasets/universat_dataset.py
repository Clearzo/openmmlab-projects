"""UniverSat-compatible dataset for OpenMMLab detectors.

This is a minimal template. In practice you will subclass ``CustomDataset``
(or ``CocoDataset``) and adapt ``load_annotations`` to your multimodal data
layout.
"""

import os

from mmdet.datasets import CustomDataset
from mmdet.datasets.builder import DATASETS


@DATASETS.register_module()
class UniverSatDataset(CustomDataset):
    """Custom dataset where each sample has multiple modality rasters.

    The annotation file is a JSON list of dicts::

        [
          {
            "filenames": {
              "s2": "s2/xxx.tif",
              "s1": "s1/xxx.tif"
            },
            "height": 360,
            "width": 360,
            "bbox": [...],
            "labels": [...],
            "masks": [...]
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

    def load_annotations(self, ann_file):
        """Load annotations from a JSON file."""
        import json

        with open(ann_file, "r") as f:
            samples = json.load(f)

        data_infos = []
        for sample in samples:
            info = {
                "filename": sample["filenames"],  # dict for multimodal loader
                "height": sample["height"],
                "width": sample["width"],
            }
            if "bbox" in sample:
                info["ann"] = dict(
                    bboxes=sample["bbox"],
                    labels=sample["labels"],
                    masks=sample.get("masks", []),
                    bboxes_ignore=sample.get("bbox_ignore", []),
                    labels_ignore=sample.get("labels_ignore", []),
                )
            data_infos.append(info)
        return data_infos

    def get_ann_info(self, idx):
        """Get annotation by index."""
        return self.data_infos[idx].get("ann", dict(bboxes=[], labels=[]))
