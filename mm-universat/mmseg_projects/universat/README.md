# UniverSat for MMSegmentation

This project provides an MMSegmentation-compatible wrapper for the **UniverSat**
multimodal Earth-observation backbone, targeted at semantic segmentation tasks.

## Directory layout

```
mmseg_projects/
├── configs/universat/
│   └── base_universat_seg.py      # Example MMSeg training config
└── universat/
    ├── README.md
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   ├── backbones/
    │   │   ├── __init__.py
    │   │   ├── universat.py       # MMSeg backbone wrapper
    │   │   └── universat_modules/ # Original encoder code
    │   └── decode_heads/
    │       ├── __init__.py
    │       └── universat_seg_head.py
    ├── datasets/
    │   ├── __init__.py
    │   ├── universat_dataset.py   # Multimodal segmentation dataset
    │   └── pipelines/
    │       └── loading.py
    └── tools/
        ├── train.sh               # Single-node training
        ├── test.sh                # Inference / evaluation
        └── dist_train.sh          # Multi-GPU distributed training
```

## How to use

1. Copy the whole `mmseg_projects/` folder into your MMSegmentation repository:

   ```bash
   cp -r universat_run/mmseg_projects/* /path/to/mmsegmentation/projects/
   ```

2. Download the released UniverSat weights:

   ```bash
   huggingface-cli download g-astruc/UniverSat --local-dir checkpoints/UniverSat
   ```

3. Update the checkpoint path in `configs/universat/base_universat_seg.py`.

4. Launch training:

   ```bash
   cd /path/to/mmsegmentation
   bash projects/universat/tools/train.sh projects/universat/configs/base_universat_seg.py
   ```

5. Multi-GPU distributed training:

   ```bash
   bash projects/universat/tools/dist_train.sh \
        projects/universat/configs/base_universat_seg.py 4
   ```

6. Test / inference:

   ```bash
   bash projects/universat/tools/test.sh \
        projects/universat/configs/base_universat_seg.py \
        work_dirs/base_universat_seg/latest.pth \
        --eval mIoU --show-dir vis/universat/
   ```

## Notes

- The segmentation head (`UniverSatSegHead`) takes the single-scale feature map
  from UniverSat and upsamples it to the target resolution.
- For multi-scale feature maps, enable `multi_grids` in the backbone config and
  adapt the head accordingly.
