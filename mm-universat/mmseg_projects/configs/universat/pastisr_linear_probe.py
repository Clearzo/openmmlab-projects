"""Linear-probe semantic segmentation on PASTIS-R using standard MMSeg training.

This config uses the migrated UniverSat backbone with ``frozen_stages=0``
so that only the lightweight linear-probe head is trained. Place it under
``mmsegmentation/projects/universat/configs/`` and run::

    python tools/train.py projects/universat/configs/pastisr_linear_probe.py

Data expected at ``data/PASTIS-R/`` with the original PASTIS-R layout:
    - metadata.geojson
    - DATA_S2/
    - DATA_S1A/
    - ANNOTATIONS/
    - NORM_S2_mean.npy / NORM_S2_std.npy
    - NORM_S1A_mean.npy / NORM_S1A_std.npy
"""

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = dict(
    type="EncoderDecoder",
    backbone=dict(
        type="UniverSatBackbone",
        modalities=["s2", "s1"],
        embed_dim=768,
        num_heads=12,
        patch_size=40,
        output_grid=128,  # 128x128 tokens covering the PASTIS-R 1280m tile
        gating=True,
        frozen_stages=0,  # 0 = freeze whole backbone (linear probe)
        init_cfg=dict(
            type="Pretrained",
            checkpoint="checkpoints/UniverSat/model.safetensors",
        ),
    ),
    decode_head=dict(
        type="UniverSatLinearProbeHead",
        in_channels=768,
        in_index=0,
        channels=768,
        num_classes=20,
        output_size=(128, 128),
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0,
        ),
    ),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
dataset_type = "PASTISRDataset"
data_root = "data/PASTIS-R/"

modalities = ["s2", "s1"]

train_pipeline = [
    dict(type="LoadPASTISMultimodal"),
    dict(type="LoadPASTISAnnotations"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="MultimodalDefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"], meta_keys=["img_metas"]),
]

test_pipeline = [
    dict(type="LoadPASTISMultimodal"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(128, 128),
        flip=False,
        transforms=[
            dict(type="MultimodalDefaultFormatBundle"),
            dict(type="Collect", keys=["img"], meta_keys=["img_metas"]),
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
        img_dir=".",
        ann_dir=".",
        split=data_root + "splits/train.json",
        pipeline=train_pipeline,
        temporal_dropout=200,
        normalize=True,
    ),
    val=dict(
        type=dataset_type,
        modalities=modalities,
        data_root=data_root,
        img_dir=".",
        ann_dir=".",
        split=data_root + "splits/val.json",
        pipeline=test_pipeline,
        temporal_dropout=0,
        normalize=True,
    ),
    test=dict(
        type=dataset_type,
        modalities=modalities,
        data_root=data_root,
        img_dir=".",
        ann_dir=".",
        split=data_root + "splits/test.json",
        pipeline=test_pipeline,
        temporal_dropout=0,
        normalize=True,
    ),
)

# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
optimizer = dict(
    type="AdamW",
    lr=2e-3,
    weight_decay=0.0,
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.0, decay_mult=0.0),
        }
    ),
)
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

runner = dict(type="IterBasedRunner", max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric="mIoU", pre_eval=True)
log_config = dict(
    interval=50,
    hooks=[dict(type="TextLoggerHook", by_epoch=False)],
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
