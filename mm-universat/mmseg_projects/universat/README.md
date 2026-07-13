# UniverSat for MMSegmentation（中文使用文档）

本项目将 **UniverSat** 多模态遥感 Transformer 骨干网络迁移到 MMSegmentation 框架，用于语义分割任务。支持通用多模态遥感数据集，以及 PASTIS-R 线性探测（linear probe）评估。

## 目录结构

```
mmseg_projects/
└── universat/
    ├── __init__.py
    ├── configs/
    │   ├── base_universat_seg.py       # 通用多模态语义分割配置示例
    │   └── pastisr_linear_probe.py     # PASTIS-R 线性探测配置
    ├── models/
    │   ├── __init__.py
    │   ├── backbones/
    │   │   ├── universat.py            # MMSegmentation 骨干网络封装
    │   │   └── universat_modules/      # 原始 UniverSat 编码器代码
    │   └── decode_heads/
    │       ├── universat_seg_head.py   # 常规分割头（带卷积+BN）
    │       └── universat_lp_head.py    # 线性探测分割头（LayerNorm+1×1）
    ├── datasets/
    │   ├── __init__.py
    │   ├── universat_dataset.py        # 通用多模态分割数据集
    │   ├── pastisr_dataset.py          # PASTIS-R 数据集
    │   └── pipelines/
    │       ├── loading.py              # LoadMultimodalFromFile / LoadPASTIS*
    │       └── formatting.py           # MultimodalDefaultFormatBundle
    └── tools/
        ├── train.sh                    # 单卡训练
        ├── test.sh                     # 推理/测试
        └── dist_train.sh               # 多卡分布式训练
```

## 环境要求

建议使用 `universat` conda 环境：

```bash
conda env create -f UniverSat/environment.yaml
conda activate universat

pip install -U openmim
mim install mmcv-full
mim install mmsegmentation
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

## 一、通用多模态语义分割

### 1. 数据格式

`UniverSatSegDataset` 假设数据按以下方式组织：

```
data/your_dataset/
├── images/
│   ├── s2/
│   │   ├── 0001.tif        # 每个模态一个子目录
│   │   └── ...
│   └── s1/
│       └── ...
├── masks/
│   ├── 0001.png            # 与样本名对应的单通道标注图
│   └── ...
└── splits/
    ├── train.json          # 每个 split 一个 JSON，包含 filenames 字典
    ├── val.json
    └── test.json
```

`train.json` 示例：

```json
[
  {
    "filename": {
      "s2": "0001.tif",
      "s1": "0001.tif"
    },
    "ann": {
      "seg_map": "0001.png"
    }
  }
]
```

### 2. 修改配置

编辑 `universat/configs/base_universat_seg.py`：

* `data_root`：数据集根目录
* `modalities`：使用的模态列表，例如 `["s2", "s1"]`
* `model.backbone.init_cfg.checkpoint`：下载的权重路径
* `model.decode_head.num_classes`：类别数
* `model.backbone.output_grid`：骨干输出特征图边长

### 3. 启动训练

本项目默认放在 UniverSat 仓库的 `universat_run/mmseg_projects/universat/` 目录下，与 `PASTIS_R/` 并列，**不需要复制到 MMSegmentation 源码内部**。脚本会自动将 `universat_run/mmseg_projects/` 加入 `PYTHONPATH`，并通过 `MMSEG_ROOT` 定位 MMSegmentation 的 `tools/train.py`。

如果 MMSegmentation 没有安装在默认位置，请设置：

```bash
export MMSEG_ROOT=/path/to/mmsegmentation
```

单卡训练：

```bash
cd /path/to/universat
bash universat_run/mmseg_projects/universat/tools/train.sh \
    universat_run/mmseg_projects/universat/configs/base_universat_seg.py
```

多卡分布式训练（4 卡示例）：

```bash
bash universat_run/mmseg_projects/universat/tools/dist_train.sh \
    universat_run/mmseg_projects/universat/configs/base_universat_seg.py 4
```

### 4. 测试/推理

```bash
bash universat_run/mmseg_projects/universat/tools/test.sh \
    universat_run/mmseg_projects/universat/configs/base_universat_seg.py \
    work_dirs/base_universat_seg/latest.pth \
    --eval mIoU --show-dir vis/universat/
```

如果你更习惯把 projects 复制到 MMSegmentation 内部，也可以执行：

```bash
cp -r universat_run/mmseg_projects/* /path/to/mmsegmentation/projects/
cd /path/to/mmsegmentation
bash projects/universat/tools/train.sh projects/universat/configs/base_universat_seg.py
```

## 二、PASTIS-R 线性探测

### 1. 数据格式

使用原始 PASTIS-R 数据集布局：

```
data/PASTIS-R/
├── metadata.geojson
├── DATA_S2/
│   ├── S2_1.npy
│   └── ...
├── DATA_S1A/
│   ├── S1A_1.npy
│   └── ...
├── ANNOTATIONS/
│   ├── TARGET_1.npy
│   └── ...
├── NORM_s2_patch.json        # 原始归一化统计文件
├── NORM_s1_patch.json
└── splits/
    ├── train.json            # { "folds": [1, 2, 3] }
    ├── val.json              # { "folds": [4] }
    └── test.json             # { "folds": [5] }
```

### 2. 修改配置

编辑 `universat/configs/pastisr_linear_probe.py`：

* `data_root`：PASTIS-R 数据集路径
* `model.backbone.init_cfg.checkpoint`：UniverSat 权重路径
* `data.train.split` / `data.val.split` / `data.test.split`：split JSON 路径

该配置默认：

* `frozen_stages=0`：冻结整个骨干网络
* `UniverSatLinearProbeHead`：仅训练 LayerNorm + 1×1 分类器
* 输入模态：`s2` + `s1`
* 输出分辨率：`128×128`（对应 PASTIS-R 原图尺寸）

### 3. 启动训练

本项目默认放在 UniverSat 仓库外部运行：

```bash
cd /path/to/universat
bash universat_run/mmseg_projects/universat/tools/train.sh \
    universat_run/mmseg_projects/universat/configs/pastisr_linear_probe.py
```

若已复制到 MMSegmentation 内部：

```bash
bash projects/universat/tools/train.sh projects/universat/configs/pastisr_linear_probe.py
```

### 4. 关于超参扫描

该配置文件只对应一组 `(lr, weight_decay)`。原始 `src/LP_eval.py` 会对多组 `(lr, wd)` 做网格搜索并取最优 mIoU。若需完整复现该流程，请使用 `PASTIS_R` 子项目：

```bash
bash universat_run/mmseg_projects/PASTIS_R/tools/run_linear_probe.sh \
    universat_run/mmseg_projects/PASTIS_R/configs/linear_probe_universat_pastisr.py
```

详细说明见 `../PASTIS_R/README.md`。

## 配置文件关键参数说明

| 参数 | 含义 | 建议值 |
|---|---|---|
| `modalities` | 输入模态列表 | `["s2", "s1"]` |
| `embed_dim` | 骨干嵌入维度 | `768`（Base） |
| `num_heads` | 注意力头数 | `12`（Base） |
| `patch_size` | 物理 patch 大小（米） | `40` |
| `output_grid` | 输出特征图边长 | 根据输入尺寸计算，PASTIS-R 用 `128` |
| `frozen_stages` | 冻结阶段 | `-1` 不冻结；`0` 冻结整个骨干（线性探测） |
| `multi_grids` | 多尺度输出 | 如需多尺度特征，设为列表如 `[32, 64, 128]` |

## 注意事项

1. **输入尺寸与 `output_grid`**：UniverSat 骨干会根据输入尺寸、`patch_size` 和模态分辨率自动推断 latent token 数量。`output_grid` 是期望的输出特征图边长，应覆盖目标标注分辨率。例如 PASTIS-R 为 `128×128`，设 `output_grid=128`。
2. **冻结阶段**：`frozen_stages=0` 会冻结 UPE 和全部 trunk block，仅训练分割头，适合线性探测。
3. **多模态数据打包**：MMSegmentation 默认把 `img` 当作单个张量。本项目通过 `MultimodalDefaultFormatBundle` 将 `img` 处理为模态字典， backbone `forward` 接收 dict 输入。
4. **内存限制**：Mac 本地测试大输入（如 360×360）容易 OOM，建议在 GPU 服务器上运行完整实验。
