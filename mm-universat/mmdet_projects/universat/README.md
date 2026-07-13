# UniverSat for MMDetection（中文使用文档）

本项目将 **UniverSat** 多模态遥感 Transformer 骨干网络迁移到 MMDetection 框架，用于目标检测任务。

## 目录结构

```
universat/                                   # UniverSat 源码仓库根
└── universat_run/                           # OpenMMLab 自定义项目集合
    └── mmdet_projects/                      # 对应 mmdetection/projects/
        └── universat/                       # 通用 UniverSat project
            ├── __init__.py
            ├── configs/
            │   └── base_universat.py        # 目标检测配置示例
            ├── models/
            │   ├── __init__.py
            │   └── backbones/
            │       ├── universat.py         # MMDetection 骨干网络封装
            │       └── universat_modules/   # 原始 UniverSat 编码器代码
            ├── datasets/
            │   ├── __init__.py
            │   ├── universat_dataset.py     # 多模态目标检测数据集
            │   └── pipelines/
            │       └── loading.py           # LoadMultimodalFromFile
            └── tools/
                ├── train.sh                 # 单卡训练
                ├── test.sh                  # 推理/测试
                ├── dist_train.sh            # 多卡分布式训练
                └── train.py                 # Python 训练入口
```

## 环境要求

建议使用 `universat` conda 环境：

```bash
conda env create -f UniverSat/environment.yaml
conda activate universat

pip install -U openmim
mim install mmcv-full
mim install mmdet
```

此外需要安装 `safetensors`（用于加载官方权重）：

```bash
pip install safetensors
```

## 权重下载

从 HuggingFace 下载官方 Base 模型权重：

```bash
huggingface-cli download g-astruc/UniverSat --local-dir checkpoints/UniverSat
```

下载完成后，在配置文件中修改 `model.backbone.init_cfg.checkpoint` 路径。

## 一、数据格式

`UniverSatDetDataset` 假设数据按 COCO 格式组织，同时在 `info` 中保存每个样本的多模态文件路径：

```
data/your_dataset/
├── images/
│   ├── s2/
│   │   ├── 0001.tif
│   │   └── ...
│   └── s1/
│       └── ...
├── annotations/
│   ├── instances_train.json    # COCO 格式，每张图片 info 包含 modalities 字段
│   ├── instances_val.json
│   └── instances_test.json
```

`instances_train.json` 中每张图片的 `info` 示例：

```json
{
  "id": 1,
  "file_name": "0001.tif",
  "width": 512,
  "height": 512,
  "modalities": {
    "s2": "s2/0001.tif",
    "s1": "s1/0001.tif"
  }
}
```

## 二、修改配置

编辑 `universat/configs/base_universat.py`：

* `data_root`：数据集根目录
* `modalities`：使用的模态列表，例如 `["s2", "s1"]`
* `model.backbone.init_cfg.checkpoint`：下载的权重路径
* `model.backbone.output_grid`：骨干输出特征图边长
* `model.rpn_head` / `model.roi_head` 中的 `num_classes`：目标类别数（不包含背景）

示例配置使用 **Faster R-CNN with FPN**。由于 UniverSat 默认输出单尺度特征，FPN 只会看到一个输入层级；如需多尺度特征，可在 backbone 中启用 `multi_grids`。

## 三、启动训练

本项目默认放在 UniverSat 仓库外部运行，不修改 MMDetection 源码：

```bash
cd /path/to/universat
bash universat_run/mmdet_projects/universat/tools/train.sh \
    universat_run/mmdet_projects/universat/configs/base_universat.py
```

如果 MMDetection 不在默认位置，请设置：

```bash
export MMDET_ROOT=/path/to/mmdetection
```

多卡分布式训练（4 卡示例）：

```bash
bash universat_run/mmdet_projects/universat/tools/dist_train.sh \
    universat_run/mmdet_projects/universat/configs/base_universat.py 4
```

## 四、测试/推理

```bash
bash universat_run/mmdet_projects/universat/tools/test.sh \
    universat_run/mmdet_projects/universat/configs/base_universat.py \
    work_dirs/base_universat/latest.pth \
    --eval bbox --show-dir vis/universat/
```

## 五、Python 入口

如果习惯直接调用 Python 脚本，可使用 `tools/train.py`：

```bash
python universat_run/mmdet_projects/universat/tools/train.py \
    universat_run/mmdet_projects/universat/configs/base_universat.py \
    --work-dir work_dirs/base_universat
```

若已复制到 MMDetection 内部，也可以按传统方式运行：

```bash
cp -r universat_run/mmdet_projects/* /path/to/mmdetection/projects/
cd /path/to/mmdetection
bash projects/universat/tools/train.sh projects/universat/configs/base_universat.py
```

## 配置文件关键参数说明

| 参数 | 含义 | 建议值 |
|---|---|---|
| `modalities` | 输入模态列表 | `["s2", "s1"]` |
| `embed_dim` | 骨干嵌入维度 | `768`（Base） |
| `num_heads` | 注意力头数 | `12`（Base） |
| `patch_size` | 物理 patch 大小（米） | `40` |
| `output_grid` | 输出特征图边长 | 根据输入尺寸与模态分辨率计算 |
| `multi_grids` | 多尺度输出 | 如需多尺度特征，设为列表如 `[32, 64, 128]` |
| `frozen_stages` | 冻结阶段 | `-1` 不冻结；`0` 冻结整个骨干 |

## 注意事项

1. **单尺度特征与 FPN**：默认 `UniverSatBackbone` 返回单尺度特征 `(B, C, H, W)`。示例配置中的 FPN 只接收这一个层级。若检测任务需要多尺度特征，建议在 backbone 中设置 `multi_grids=[...]`，并相应调整 neck/head 配置。
2. **输入尺寸**：UniverSat 使用物理 patch size（默认 40 米）和模态分辨率推断 latent grid。请确保配置中的 `patch_size`、`input_res`（通常来自 `modality_registry.py`）与输入图像尺寸匹配。
3. **自定义模态**：若需使用 `modality_registry.py` 中未注册的传感器，可在 backbone 配置中通过 `wavelengths`、`input_res`、`subpatches` 参数显式指定。
4. **内存限制**：大输入图像（如 360×360）在 CPU/Mac 上容易 OOM，建议在 GPU 服务器上运行完整实验。
