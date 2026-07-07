"""Loading transforms for PASTIS-R in MMSegmentation."""

import os

import numpy as np
import torch


class LoadPASTISMultimodal:
    """Load PASTIS-R time-series tensors and auxiliaries from data_infos.

    Expected keys in ``results``:
        - modality_paths (dict): modality -> .npy path
        - dates (dict): modality -> tensor of relative day indices
    Keys added:
        - img (dict): modality -> tensor of shape (T, C, H, W)
        - modality_paths, dates
    """

    def __call__(self, results):
        modality_paths = results["modality_paths"]
        dates = results["dates"]
        img = {}
        for mod, path in modality_paths.items():
            tensor = torch.from_numpy(np.load(path).astype(np.float32))
            # Original files are typically (T, C, H, W) already.
            if tensor.ndim == 4:
                pass
            elif tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            img[mod] = tensor

        # Normalize (mirrors src.data.utils.apply_norm)
        dataset = results.get("dataset", None)
        if dataset is not None and dataset.normalize and hasattr(dataset, "norm"):
            for mod in img:
                if mod in dataset.norm:
                    mean, std = dataset.norm[mod]
                    data = img[mod]
                    if data.ndim == 3:  # (C, H, W)
                        m = mean.view(-1, 1, 1)
                        s = std.view(-1, 1, 1)
                    elif data.ndim == 4:  # (T, C, H, W)
                        m = mean.view(1, -1, 1, 1)
                        s = std.view(1, -1, 1, 1)
                    else:
                        continue
                    img[mod] = (data - m) / s.clamp_min(1e-6)

        # Temporal dropout (mirrors Pastis.__getitem__)
        dataset = results.get("dataset", None)
        if dataset is not None and not results.get("test_mode", False):
            if dataset.temporal_dropout != float("inf"):
                for mod in img:
                    t = img[mod].shape[0]
                    if t > dataset.temporal_dropout:
                        indices = torch.randperm(t)[: dataset.temporal_dropout]
                        img[mod] = img[mod][indices]
                        dates[mod] = dates[mod][indices]

        results["img"] = img
        # Dates are kept in meta for possible future use, but are not fed to
        # the backbone (UniverSat forward treats every key in x as a sensor).
        results["img_metas"] = {"dates": dates, "id_patch": results["filename"].get("id_patch")}
        results["modality_paths"] = modality_paths
        return results


class LoadPASTISAnnotations:
    """Load PASTIS-R semantic label maps."""

    def __call__(self, results):
        dataset = results["dataset"]
        results["gt_semantic_seg"] = dataset.get_seg_annotation(results["idx"])
        return results
