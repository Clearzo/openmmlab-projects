# UniverSat for MMDetection

This project provides an MMDetection-compatible wrapper for the **UniverSat**
multimodal Earth-observation backbone, targeted at object detection tasks.

## Directory layout

```
mmdet_projects/
├── configs/universat/
│   └── base_universat.py          # Example MMDet training config
└── universat/
    ├── README.md
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   └── backbones/
    │       ├── __init__.py
    │       ├── universat.py       # MMDet backbone wrapper
    │       └── universat_modules/ # Original encoder code
    ├── datasets/
    │   ├── __init__.py
    │   ├── universat_dataset.py   # Multimodal detection dataset
    │   └── pipelines/
    │       └── loading.py
    └── tools/
        ├── train.sh               # Single-node training
        ├── test.sh                # Inference / evaluation
        ├── dist_train.sh          # Multi-GPU distributed training
        └── train.py               # Python training entrypoint
```

## How to use

1. Copy the whole `mmdet_projects/` folder into your MMDetection repository:

   ```bash
   cp -r universat_run/mmdet_projects/* /path/to/mmdetection/projects/
   ```

2. Download the released UniverSat weights:

   ```bash
   huggingface-cli download g-astruc/UniverSat --local-dir checkpoints/UniverSat
   ```

3. Update the checkpoint path in `configs/universat/base_universat.py`.

4. Launch training:

   ```bash
   cd /path/to/mmdetection
   bash projects/universat/tools/train.sh projects/universat/configs/base_universat.py
   ```

5. Multi-GPU distributed training:

   ```bash
   bash projects/universat/tools/dist_train.sh \
        projects/universat/configs/base_universat.py 4
   ```

6. Test / inference:

   ```bash
   bash projects/universat/tools/test.sh \
        projects/universat/configs/base_universat.py \
        work_dirs/base_universat/latest.pth \
        --eval bbox --show-dir vis/universat/
   ```

## Notes

- The example config uses a standard Faster R-CNN with FPN on top of UniverSat.
- Since UniverSat returns a single-scale feature map, the FPN only sees one
  input level. Enable `multi_grids` in the backbone config for multi-scale
  features, or use a custom neck.
