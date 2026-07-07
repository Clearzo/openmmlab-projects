"""Formatting transforms for multimodal inputs in MMSegmentation."""

import numpy as np
import torch
from mmcv.parallel import DataContainer as DC


class MultimodalDefaultFormatBundle:
    """Format multimodal inputs for DataContainer wrapping.

    Unlike the standard MMSeg ``DefaultFormatBundle`` which expects ``img``
    to be a single ndarray, this transform handles ``img`` as a dict of
    tensors/ndarrays (one per modality) plus auxiliary keys such as dates.

    Keys ending in ``_dates`` or ``_cloud_density`` are treated as auxiliary
    tensors and wrapped as well.
    """

    def __call__(self, results):
        if "img" in results:
            img = results["img"]
            if isinstance(img, dict):
                formatted = {}
                for key, value in img.items():
                    if isinstance(value, np.ndarray):
                        value = torch.from_numpy(value)
                    if isinstance(value, torch.Tensor):
                        if value.ndim == 3:
                            value = value.float()
                        formatted[key] = DC(value, stack=True, pad_dims=None)
                results["img"] = formatted
            else:
                results["img"] = DC(torch.from_numpy(img).float(), stack=True, pad_dims=None)

        if "gt_semantic_seg" in results:
            gt = results["gt_semantic_seg"]
            if isinstance(gt, np.ndarray):
                gt = torch.from_numpy(gt)
            results["gt_semantic_seg"] = DC(gt.long(), stack=True, pad_dims=None)

        if "img_metas" in results:
            results["img_metas"] = DC(results["img_metas"], cpu_only=True)

        return results
