"""Custom loading transforms for UniverSat multimodal inputs."""

import os

import numpy as np
import rasterio
import torch
from mmcv.parallel import DataContainer as DC


class LoadMultimodalFromFile:
    """Load multiple modality rasters and pack them into a dict.

    Expected annotation fields:
        - ``img_prefix``: root directory
        - ``img_info['filenames']``: dict ``{modality: relative_path}``

    Args:
        modalities (list[str]): Modality names, must match backbone config.
        to_float32 (bool): Convert loaded arrays to float32. Default True.
    """

    def __init__(self, modalities, to_float32=True):
        self.modalities = modalities
        self.to_float32 = to_float32

    def __call__(self, results):
        filenames = results["img_info"]["filenames"]
        prefix = results.get("img_prefix", "")

        imgs = {}
        for mod in self.modalities:
            path = filenames[mod] if isinstance(filenames, dict) else filenames
            if not os.path.isabs(path):
                path = os.path.join(prefix, path)

            with rasterio.open(path) as src:
                img = src.read()  # (C, H, W)

            if self.to_float32:
                img = img.astype(np.float32)
            imgs[mod] = img

        results["img"] = imgs  # dict of numpy arrays
        results["img_fields"] = ["img"]
        results["img_shape"] = imgs[self.modalities[0]].shape
        results["ori_shape"] = imgs[self.modalities[0]].shape
        return results


class PackMultimodalInputs:
    """Pack dict-style multimodal inputs into MMDet format.

    This transform should be placed at the end of the train/val pipeline.
    It converts the numpy dict into PyTorch tensors and wraps them in
    ``DataContainer`` for ``collate``.
    """

    def __init__(self, keys=("img", "gt_bboxes", "gt_labels", "gt_masks")):
        self.keys = keys

    def __call__(self, results):
        packed = {}
        for key in self.keys:
            if key not in results:
                continue
            val = results[key]
            if key == "img":
                # val is dict {modality: ndarray (C,H,W)}
                val = {
                    k: DC(torch.from_numpy(v), stack=True, pad_dims=None)
                    for k, v in val.items()
                }
            packed[key] = val
        return packed
