"""Base config for UniverSat + Faster R-CNN on multimodal EO data.

Place this file inside ``mmdetection/projects/universat/configs/`` and run::

    python tools/train.py projects/universat/configs/base_universat.py
"""

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = dict(
    type="FasterRCNN",
    backbone=dict(
        type="UniverSatBackbone",
        modalities=["s2", "s1"],
        embed_dim=768,
        num_heads=12,
        patch_size=40,
        output_grid=36,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="/Users/clearzpz/Documents/Clearz/Projects/TXQH2-work/Codes/EO_test_working/checkpoints/pretrained_chpts/universat/model.safetensors",  # HF released weights
            # /Users/clearzpz/Documents/Clearz/Projects/TXQH2-work/Codes/EO_test_working/checkpoints/pretrained_chpts/universat
        ),
    ),
    neck=dict(
        type="FPN",
        in_channels=[768],
        out_channels=256,
        num_outs=5,
    ),
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
    ),
    roi_head=dict(
        type="StandardRoIHead",
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type="Shared2FCBBoxHead",
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        ),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type="nms", iou_threshold=0.5),
            max_per_img=100,
        ),
    ),
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
dataset_type = "UniverSatDataset"
data_root = "data/your_eo_dataset/"

modalities = ["s2", "s1"]

img_norm_cfg = dict(
    mean={"s2": [0.0] * 10, "s1": [0.0] * 3},
    std={"s2": [1.0] * 10, "s1": [1.0] * 3},
    to_rgb=False,
)

train_pipeline = [
    dict(type="LoadMultimodalFromFile", modalities=modalities),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=False),
    dict(type="Resize", img_scale=(360, 360), keep_ratio=False),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=4),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadMultimodalFromFile", modalities=modalities),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(360, 360),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=False),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=4),
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
        ann_file=data_root + "annotations/train.json",
        img_prefix=data_root + "images/",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        modalities=modalities,
        ann_file=data_root + "annotations/val.json",
        img_prefix=data_root + "images/",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        modalities=modalities,
        ann_file=data_root + "annotations/test.json",
        img_prefix=data_root + "images/",
        pipeline=test_pipeline,
    ),
)

# ---------------------------------------------------------------------------
# Schedule & runtime
# ---------------------------------------------------------------------------
optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11],
)
runner = dict(type="EpochBasedRunner", max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
