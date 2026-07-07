"""MMSegmentation-compatible backbone wrapper for UniverSat.

This file is a migration template for OpenMMLab projects. Copy the whole
``universat/`` project folder under ``mmsegmentation/projects/`` and run
training with the provided configs.

The underlying encoder code lives in ``universat_modules/`` and is kept as
close as possible to the original repository so that released checkpoints can
be loaded without key renaming beyond stripping the ``model.`` hub wrapper
prefix.
"""

import os
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# MMSeg registration (graceful fallback for standalone testing)
# ---------------------------------------------------------------------------
try:
    from mmcv.runner import BaseModule
    from mmseg.models.builder import BACKBONES
    _HAS_MMSEG = True
except ImportError:  # pragma: no cover
    BaseModule = nn.Module
    _HAS_MMSEG = False

    class _DummyRegistry:
        @staticmethod
        def register_module(cls=None, *, name=None, force=False):
            def _register(cls_):
                return cls_
            if cls is None:
                return _register
            return _register(cls)

    BACKBONES = _DummyRegistry()

# ---------------------------------------------------------------------------
# Core UniverSat components
# ---------------------------------------------------------------------------
from .universat_modules.UniverSat import UniverSat
from .universat_modules.UniversalPatchEncoder import UniversalPatchEncoder
from .universat_modules.utils.utils import RMSNorm
from .universat_modules.modality_registry import INPUT_RES, SUBPATCHES, WAVELENGTHS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unroll_block_list(blocks: List[str]) -> List[str]:
    """Same helper used by ``UniverSat.__init__`` for config compatibility."""
    out = []
    for b in blocks:
        if "x" in b:
            n = int(b.split("x")[1])
            bt = b.split("x")[0]
            out.extend([bt] * n)
        else:
            out.append(b)
    return out


def _load_checkpoint(path: str, map_location: str = "cpu"):
    """Load a checkpoint in either PyTorch or SafeTensors format."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "Loading .safetensors checkpoints requires `safetensors`. "
                "Install it with `pip install safetensors`."
            ) from exc
        return load_file(path, device=map_location)
    else:
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=map_location)


def _strip_hub_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove the ``model.`` prefix added by ``_UniverSatHub``."""
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            k = k[len("model."):]
        new_state[k] = v
    return new_state


DEFAULT_MODALITIES_DICT = {
    "flair": ["spotRGBN", "aerialflair", "s2flair", "s1flair", "dem"],
    "pastishd": ["spot", "s2", "s1"],
    "planted": ["s2", "s1", "l7", "alos", "modis"],
    "tsaits": ["aerial", "s2", "s1"],
    "s2naip": ["naip", "l8", "s2", "s1"],
    "hyperglobal": ["EO1"],
    "earthview": ["rgbneon", "ndemneon", "neon"],
    "spectralearth": ["enmap"],
}


@BACKBONES.register_module()
class UniverSatBackbone(BaseModule):
    """MMSegmentation backbone wrapping the UniverSat multimodal encoder.

    This is the same encoder as the MMDetection version but registered with
    ``mmseg.models.builder.BACKBONES``.
    """

    def __init__(
        self,
        modalities: Sequence[str],
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        patch_size: float = 40.0,
        output_grid: Optional[int] = None,
        block_type: Sequence[str] = ("Bi_ACA_in", "SAx12", "Bilinear_out", "CA_Sub"),
        n_registers: int = 4,
        gating: bool = True,
        proba_drop_modalities: float = 0.0,
        modalities_dict: Optional[Dict[str, List[str]]] = None,
        wavelengths: Optional[Dict[str, List[Union[float, str]]]] = None,
        input_res: Optional[Dict[str, float]] = None,
        subpatches: Optional[Dict[str, int]] = None,
        keep_intermediate: bool = False,
        out_indices: Tuple[int, ...] = (-1,),
        multi_grids: Optional[Sequence[int]] = None,
        frozen_stages: int = -1,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        if BaseModule is nn.Module:
            super().__init__()
            self.init_cfg = init_cfg
        else:
            super().__init__(init_cfg)
        self.frozen_stages = frozen_stages
        self.modalities = list(modalities)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.output_grid = output_grid
        self.block_type = _unroll_block_list(list(block_type))
        self.n_registers = n_registers
        self.gating = gating
        self.proba_drop_modalities = proba_drop_modalities
        self.modalities_dict = modalities_dict or DEFAULT_MODALITIES_DICT
        self.keep_intermediate = keep_intermediate
        self.out_indices = out_indices
        self.multi_grids = list(multi_grids) if multi_grids else None

        self.wavelengths = {}
        self.input_res = {}
        self.subpatches = {}
        for mod in self.modalities:
            if wavelengths is not None and mod in wavelengths:
                self.wavelengths[mod] = list(wavelengths[mod])
            elif mod in WAVELENGTHS:
                self.wavelengths[mod] = list(WAVELENGTHS[mod])
            else:
                raise KeyError(
                    f"Modality {mod!r} not in modality_registry and no "
                    f"`wavelengths` override provided."
                )

            if input_res is not None and mod in input_res:
                self.input_res[mod] = float(input_res[mod])
            elif mod in INPUT_RES:
                self.input_res[mod] = float(INPUT_RES[mod])
            else:
                raise KeyError(
                    f"Modality {mod!r} has no `input_res` in registry and no "
                    f"override provided."
                )

            if subpatches is not None and mod in subpatches:
                self.subpatches[mod] = int(subpatches[mod])
            else:
                self.subpatches[mod] = int(SUBPATCHES.get(mod, 1))

        spatial_encoder = partial(
            UniversalPatchEncoder,
            embed_dim=embed_dim // 8,
            final_dim=embed_dim,
            n_queries=[1, 1, 1, 1],
            expand_dim=[2, 2, 2, 2],
            order=["S1", "C", "T", "S"],
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop_rate=0.0,
            gating=gating,
        )

        self.model = UniverSat(
            spatial_encoder=spatial_encoder,
            block_type=self.block_type,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=False,
            n_registers=n_registers,
            pre_norm=False,
            drop_rate=0.0,
            drop_path_rate=0.0,
            attn_drop_rate=0.0,
            norm_layer=partial(RMSNorm),
            gating=gating,
            proba_drop_modalities=proba_drop_modalities,
            modalities_dict=self.modalities_dict,
        )

        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze stages according to ``self.frozen_stages``.

        Stages are defined as:
            - stage 0: the UPE (spatial_encoder)
            - stage 1..N: the i-th block in the trunk (1-indexed)
        ``frozen_stages=-1`` disables freezing. ``frozen_stages=0`` freezes
        the whole backbone (UPE + all trunk blocks). ``frozen_stages=1``
        freezes the UPE only, etc.
        """
        if self.frozen_stages < 0:
            return

        # Freeze UPE (stage 0)
        if self.frozen_stages >= 0:
            self.model.spatial_encoder.eval()
            for param in self.model.spatial_encoder.parameters():
                param.requires_grad = False

        # Freeze trunk blocks (stage 1..N)
        for i, block in enumerate(self.model.blocks):
            if i + 1 <= self.frozen_stages:
                block.eval()
                for param in block.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Override train so frozen stages stay in eval mode."""
        super().train(mode)
        self._freeze_stages()
        return self

    def init_weights(self) -> None:
        """Load pretrained weights from ``init_cfg['checkpoint']``."""
        if self.init_cfg is None:
            return

        checkpoint_path = self.init_cfg.get("checkpoint")
        if checkpoint_path is None:
            super().init_weights()
            return

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = _load_checkpoint(checkpoint_path, map_location="cpu")

        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict and isinstance(state_dict["model"], dict):
                state_dict = state_dict["model"]

        state_dict = _strip_hub_prefix(state_dict)

        ignored = [k for k in state_dict if k.startswith("projector__")]
        for k in ignored:
            del state_dict[k]
        if ignored:
            warnings.warn(f"Ignored {len(ignored)} projector weights.")

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            warnings.warn(f"Missing keys: {missing}")
        if unexpected:
            warnings.warn(f"Unexpected keys: {unexpected}")

    def _infer_latent_grid(self, x: Dict[str, torch.Tensor]) -> int:
        """Infer latent token count from the first spatial modality."""
        for mod in self.modalities:
            t = x[mod]
            if t.ndim >= 4:
                h = t.shape[-2]
                patch_px = max(int(self.patch_size / self.input_res[mod]), 1)
                side = max(h // patch_px, 1)
                return side * side
        raise ValueError("Cannot infer latent_grid from inputs.")

    def _compute_output_grid(self, latent_grid: int) -> int:
        """Return output side length (G of GxG)."""
        if self.output_grid is not None:
            return self.output_grid
        latent_side = int(latent_grid ** 0.5)
        max_sub = max(self.subpatches.values()) if self.subpatches else 1
        return latent_side * max_sub

    def _run_once(
        self,
        x: Dict[str, torch.Tensor],
        latent_grid: int,
        output_side: int,
    ) -> torch.Tensor:
        """Single forward pass returning (B, C, H, W) features."""
        output_tokens = output_side * output_side
        scale = self.patch_size / 10.0

        tokens, _ = self.model(
            x=x,
            wavelengths=self.wavelengths,
            input_res=self.input_res,
            scale=scale,
            latent_grid=latent_grid,
            output_grid=output_tokens,
            subpatches=self.subpatches,
            keep_intermediate=self.keep_intermediate,
            mask_in=None,
            mask_out=None,
        )

        tokens = tokens[:, self.n_registers:]
        b, n, c = tokens.shape
        h_w = int(round(n ** 0.5))
        features = tokens.permute(0, 2, 1).reshape(b, c, h_w, h_w)
        return features

    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
        """MMSeg-style forward.

        Args:
            x: Either a dict ``{modality: tensor}`` or a single tensor.

        Returns:
            list[torch.Tensor]: Feature maps.
        """
        if not isinstance(x, dict):
            if len(self.modalities) != 1:
                raise ValueError(
                    "Single-tensor input is only allowed when exactly one "
                    "modality is configured."
                )
            x = {self.modalities[0]: x}

        for mod in self.modalities:
            if mod not in x:
                raise KeyError(
                    f"Modality {mod!r} missing from input. Got keys: {list(x.keys())}"
                )

        latent_grid = self._infer_latent_grid(x)

        if self.multi_grids:
            features = [
                self._run_once(x, latent_grid, grid_side)
                for grid_side in self.multi_grids
            ]
            return [features[i] for i in self.out_indices]

        output_side = self._compute_output_grid(latent_grid)
        return [self._run_once(x, latent_grid, output_side)]


if __name__ == "__main__":
    backbone = UniverSatBackbone(
        modalities=["s2", "s1"],
        embed_dim=768,
        patch_size=40,
        output_grid=36,
    )
    x = {
        "s2": torch.randn(2, 10, 360, 360),
        "s1": torch.randn(2, 3, 360, 360),
    }
    print(backbone(x)[0].shape)
