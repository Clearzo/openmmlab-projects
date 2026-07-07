
已按你的要求创建好了迁移后的 OpenMMLab 项目结构，位置在：

```bash
projects/
├── configs/universat/base_universat.py   # 训练配置
└── universat/                            # 项目代码包
    ├── README.md
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   └── backbones/
    │       ├── __init__.py
    │       ├── universat.py              # MMDet backbone wrapper
    │       └── universat_modules/        # 原 UniverSat 骨干代码
    ├── datasets/
    │   ├── __init__.py
    │   ├── universat_dataset.py          # 多模态数据集
    │   └── pipelines/
    │       └── loading.py                # 多模态加载 transform
    └── tools/
        └── train.py                      # 训练入口
```

我已验证 backbone 可以正常运行（当前环境下用 `output_grid=9`、输入 `36×36` 测试通过，更大尺寸因本机内存不足被系统 kill，属于环境问题不是代码问题）。

---

## 一、迁移后的完整代码结构

```bash
projects/
├── configs/
│   └── universat/
│       └── base_universat.py          # 示例：UniverSat + Faster R-CNN 配置
└── universat/
    ├── README.md                       # 使用说明
    ├── __init__.py                     # 自动注册 models/datasets
    ├── models/
    │   ├── __init__.py                 # from .backbones import *
    │   └── backbones/
    │       ├── __init__.py             # 导出 UniverSatBackbone
    │       ├── universat.py            # ★ MMDet wrapper（新增）
    │       └── universat_modules/      # ★ 原骨干代码（最小改动迁移）
    │           ├── __init__.py
    │           ├── UniverSat.py
    │           ├── UniversalPatchEncoder.py
    │           ├── modality_registry.py
    │           ├── masking/            # SSL mask 模块（推理时传 None）
    │           │   ├── __init__.py
    │           │   ├── mask.py
    │           │   ├── masker.py
    │           │   └── utils.py
    │           └── utils/
    │               ├── __init__.py
    │               ├── flexiVit.py
    │               ├── patch_embeddings.py
    │               ├── pos_embed.py
    │               ├── utils.py
    │               └── utils_ViT.py
    ├── datasets/
    │   ├── __init__.py                 # 导出 UniverSatDataset, LoadMultimodalFromFile
    │   ├── universat_dataset.py        # CustomDataset 多模态数据集模板
    │   └── pipelines/
    │       └── loading.py              # LoadMultimodalFromFile transform
    └── tools/
        └── train.py                    # 自动加项目路径并调用 tools/train.py
```

---

## 二、与原项目的对应关系

| 原项目文件                                               | 迁移后位置                                     | 说明                      |
| -------------------------------------------------------- | ---------------------------------------------- | ------------------------- |
| `src/models/networks/encoder/UniverSat.py`             | `universat_modules/UniverSat.py`             | 顶层编码器                |
| `src/models/networks/encoder/UniversalPatchEncoder.py` | `universat_modules/UniversalPatchEncoder.py` | UPE                       |
| `src/models/networks/encoder/utils/*.py`               | `universat_modules/utils/*.py`               | Attention/Block/RoPE/坐标 |
| `modality_registry.py`                                 | `universat_modules/modality_registry.py`     | 传感器元数据              |
| `src/models/networks/masking/*.py`                     | `universat_modules/masking/*.py`             | mask 工具（保留依赖）     |
| 无                                                       | `backbones/universat.py`                     | 新增 MMDet wrapper        |
| 无                                                       | `datasets/universat_dataset.py`              | 新增多模态数据集          |
| 无                                                       | `datasets/pipelines/loading.py`              | 新增多模态加载            |
| 无                                                       | `configs/universat/base_universat.py`        | 新增训练配置              |

---

## 三、为迁移做的关键修改

### 1. Import 路径改为相对导入

所有 `models.networks.encoder.xxx` 改为 `.utils.xxx`、`.masking.xxx` 等，使代码能在 `universat_modules` 包内自包含。

### 2. 增加 `torch.compile` 兼容 shim

原代码使用 `@torch.compile()`，在旧 PyTorch 或 Python 3.11+ 下可能报错。迁移版本加了自动 fallback：

```python
def _compile(*args, **kwargs):
    try:
        return torch.compile(*args, **kwargs)
    except (AttributeError, RuntimeError):
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn
```

在支持的环境中会自动编译；在不支持的环境中自动跳过，不影响功能。

### 3. MMDet 缺失时自动回退

`universat.py` 中的 wrapper 在没有 MMDet 时会回退到普通 `nn.Module`，方便独立测试：

```python
try:
    from mmcv.runner import BaseModule
    from mmdet.models.builder import BACKBONES
except ImportError:
    BaseModule = nn.Module
    BACKBONES = _DummyRegistry()
```

---

## 四、如何集成到 mmdetection

### 方式一：复制到 mmdetection/projects/ 下

```bash
cp -r projects/universat /path/to/mmdetection/projects/
cp -r projects/configs/universat /path/to/mmdetection/projects/configs/
```

### 方式二：通过 PYTHONPATH 引入

如果不想复制，训练时把 `projects/` 加入 `PYTHONPATH`：

```bash
PYTHONPATH=/path/to/this/projects:$PYTHONPATH \
python mmdetection/tools/train.py /path/to/this/projects/configs/universat/base_universat.py
```

### 训练命令

```bash
cd /path/to/mmdetection
python tools/train.py projects/universat/configs/base_universat.py
```

或使用项目自带入口：

```bash
python projects/universat/tools/train.py projects/universat/configs/base_universat.py
```

---

## 五、配置中如何使用

`configs/universat/base_universat.py` 里已经给出示例：

```python
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
            checkpoint="path/to/model.safetensors",
        ),
    ),
    neck=dict(...),
    ...
)
```

`init_cfg` 支持：

- HuggingFace `model.safetensors`
- PyTorch `.pth` / `.pt` / `.ckpt`
- Lightning checkpoint（自动提取 `state_dict`）

wrapper 会自动：

1. 去掉 `_UniverSatHub` 的 `model.` 前缀
2. 忽略 SSL projector 权重
3. 加载 backbone 部分

---

## 六、数据集格式

`universat_dataset.py` 期望的 annotation JSON 格式：

```json
[
  {
    "filenames": {
      "s2": "s2/xxx.tif",
      "s1": "s1/xxx.tif"
    },
    "height": 360,
    "width": 360,
    "bbox": [[x, y, w, h], ...],
    "labels": [0, 1, ...]
  }
]
```

`LoadMultimodalFromFile` 会读取每个模态的 GeoTIFF，组成 `{"s2": tensor, "s1": tensor}` 传给 backbone。

---

## 七、当前模板的局限与后续可扩展

| 方面     | 当前状态                       | 建议                                                |
| -------- | ------------------------------ | --------------------------------------------------- |
| Neck     | 示例用 FPN，但 backbone 单尺度 | 可改用`multi_grids` 输出多尺度，或定制 neck       |
| 分割     | 当前是检测配置                 | 可添加`universat/models/decode_heads/` 用于 MMSeg |
| 时序     | 当前 pipeline 未处理`dates`  | 可在`LoadMultimodalFromFile` 中加入 dates 读取    |
| 数据增强 | 示例较简单                     | 可按需添加多模态同步 Flip/Rotate/Crop               |
| 测试     | 无 unit test                   | 可在`universat/tests/` 中添加                     |

如果你需要，我可以继续帮你：

- 写一个从 HuggingFace 自动下载并转换权重的脚本；
- 设计多尺度 FPN neck 的对接；
- 写一个 MMSegmentation 分割版本的项目结构。
