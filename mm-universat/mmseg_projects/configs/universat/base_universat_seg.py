"""Base config for UniverSat + FCN on multimodal EO segmentation.

Place this file inside ``mmsegmentation/projects/universat/configs/`` and run::

    python tools/train.py projects/universat/configs/base_universat_seg.py
"""

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
norm_cfg = dict(type="BN", requires_grad=True)

model = dict(
    type="EncoderDecoder",
    backbone=dict(
        type="UniverSatBackbone",
        modalities=["s2", "s1"],
        embed_dim=768,
        num_heads=12,
        patch_size=40,
        output_grid=36,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="path/to/model.safetensors",
        ),
    ),
    decode_head=dict(
        type="UniverSatSegHead",
        in_channels=768,
        in_index=0,
        channels=256,
        num_convs=2,
        output_size=(360, 360),  # upsample to input resolution
        num_classes=10,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0,
        ),
    ),
    auxiliary_head=None,
    # training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
dataset_type = "UniverSatSegDataset"
data_root = "data/your_eo_seg_dataset/"

modalities = ["s2", "s1"]

img_norm_cfg = dict(
    mean={"s2": [0.0] * 10, "s1": [0.0] * 3},
    std={"s2": [1.0] * 10, "s1": [1.0] * 3},
    to_rgb=False,
)

crop_size = (360, 360)
train_pipeline = [
    dict(type="LoadMultimodalFromFile", modalities=modalities),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(360, 360), ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]

test_pipeline = [
    dict(type="LoadMultimodalFromFile", modalities=modalities),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(360, 360),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        modalities=modalities,
        data_root=data_root,
        img_dir="images",
        ann_dir="masks",
        split=data_root + "splits/train.json",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        modalities=modalities,
        data_root=data_root,
        img_dir="images",
        ann_dir="masks",
        split=data_root + "splits/val.json",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        modalities=modalities,
        data_root=data_root,
        img_dir="images",
        ann_dir="masks",
        split=data_root + "splits/test.json",
        pipeline=test_pipeline,
    ),
)

# ---------------------------------------------------------------------------
# Schedule & runtime
# ---------------------------------------------------------------------------
optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05)
optimizer_config = dict()
lr_config = dict(
    policy="poly",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)
runner = dict(type="IterBasedRunner", max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric="mIoU", pre_eval=True)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ],
)

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
cudnn_benchmark = True
