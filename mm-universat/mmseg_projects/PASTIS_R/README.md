# UniverSat PASTIS-R 下游任务迁移（MMSegmentation 自定义项目）

本目录将 UniverSat 源码中 PASTIS-R 语义分割下游任务迁移为独立的 MMSegmentation 自定义 project，与 `../universat/` 并列放置在 `universat_run/mmseg_projects/` 下。

这种布局的好处是：
- **不破坏 MMSegmentation 源码**：所有自定义代码都在 UniverSat 仓库内部。
- **便于修改调试**：可以直接在本仓库修改 backbone、head、dataset，MMSegmentation 通过 `PYTHONPATH` 加载即可。

包含：

- **Linear Probe（线性探测）**：复现 `UniverSat/src/LP_eval.py` 的 (lr, weight_decay) 超参扫掠流程。
- **Fine-tuning（端到端微调）**：复现 `UniverSat/src/train.py` 的语义分割微调流程，通过 MMSegmentation 的 `tools/train.py` 运行。

## 目录结构

```text
universat/                                # UniverSat 源码仓库根
└── universat_run/                        # OpenMMLab 自定义项目集合
    └── mmseg_projects/                   # 相当于 mmsegmentation/projects/
        ├── universat/                    # 通用组件 project
        │   ├── models/
        │   │   ├── backbones/universat.py
        │   │   └── decode_heads/
        │   ├── datasets/
        │   │   ├── pastisr_dataset.py       # MMSeg 版 PASTIS-R dataset
        │   │   └── pipelines/
        │   └── configs/
        └── PASTIS_R/                     # PASTIS-R 下游任务 project（本目录）
            ├── configs/
            │   ├── linear_probe_universat_pastisr.py
            │   ├── linear_probe_universat_pastisr_local.py
            │   └── ft_universat_pastisr.py
            ├── datasets/
            │   └── pastisr_dataset.py       # 独立版 Dataset（用于 linear_probe.py）
            ├── models/
            │   └── linear_probe_head.py
            └── tools/
                ├── linear_probe.py
                ├── run_linear_probe.sh
                ├── dist_run_linear_probe.sh
                ├── train.sh
                ├── dist_train.sh
                └── test.sh
```

## `PASTIS_R/` 与 `universat/` 的关系

- `universat/`：通用组件库，提供 `UniverSatBackbone`、`UniverSatSegHead`、`UniverSatLinearProbeHead`、MMSeg 版 `PASTISRDataset` 及 pipelines。
- `PASTIS_R/`：PASTIS-R 下游任务专用 project，通过 Python import / MMSegmentation registry 引用 `universat/` 的组件。

## 依赖

在 `universat` conda 环境中已验证：

```bash
conda activate universat
python -m pip install geopandas rasterio einops
```

`../universat/` 目录下的 backbone 代码已经做了 MMSeg 缺失时的 graceful fallback，因此即使没有安装 `mmcv/mmseg` 也能运行 linear probe。

Fine-tuning 需要安装 MMSegmentation（可与本仓库独立安装）：

```bash
pip install -U openmim
mim install mmcv-full
mim install mmsegmentation
```

## 数据与权重准备

PASTIS-R 数据集应包含以下文件：

```text
data/PASTIS-R/
├── metadata.geojson
├── DATA_S2/
├── DATA_S1A/
├── ANNOTATIONS/
├── NORM_S2_mean.npy / NORM_S2_std.npy
└── NORM_S1A_mean.npy / NORM_S1A_std.npy
```

预训练权重：

- 官方 `model.safetensors`（Base，768-dim，12 heads）可直接加载。
- 若使用 Lightning 检查点，需过滤掉 `projector__*` 键并转换为 `state_dict` 格式。

## Linear Probe

Linear probe 不依赖 MMSegmentation，直接运行即可。

### 单卡运行

```bash
cd /path/to/universat
bash universat_run/mmseg_projects/PASTIS_R/tools/run_linear_probe.sh \
    universat_run/mmseg_projects/PASTIS_R/configs/linear_probe_universat_pastisr.py
```

或直接通过 Python：

```bash
cd /path/to/universat
PYTHONPATH="/path/to/universat/universat_run/mmseg_projects:${PYTHONPATH}" \
python universat_run/mmseg_projects/PASTIS_R/tools/linear_probe.py \
    universat_run/mmseg_projects/PASTIS_R/configs/linear_probe_universat_pastisr.py
```

### 多卡运行

```bash
cd /path/to/universat
bash universat_run/mmseg_projects/PASTIS_R/tools/dist_run_linear_probe.sh \
    universat_run/mmseg_projects/PASTIS_R/configs/linear_probe_universat_pastisr.py 2
```

### 配置说明

编辑 `configs/linear_probe_universat_pastisr.py` 中的路径和超参：

| 配置项 | 说明 |
|--------|------|
| `data_root` / `norm_path` | PASTIS-R 数据集路径 |
| `modalities` | 本地数据仅支持 `["s2", "s1"]`，`spot` 不存在需保持注释 |
| `train_folds` / `val_folds` / `test_folds` | 默认 `[1,2,3]` / `[4]` / `[5]` |
| `checkpoint` | 预训练权重路径 |
| `embed_dim=768`, `num_heads=12` | Base 模型配置 |
| `patch_size=40`, `output_grid=128` | PASTIS-R 1280m tile → 128×128 tokens |
| `patch_size_px` | 每个输出 token 预测的标注像素边长，PASTIS-R 128×128 label 对应 `1` |
| `lr_list` / `weight_decay_list` | linear probe 超参扫掠网格 |
| `batch_size` / `max_epochs` | 探测头训练参数 |

### 与原始 `LP_eval.py` 的对应关系

| 本脚本 | 原始代码 |
|--------|----------|
| `tools/linear_probe.py` | `src/LP_eval.py` |
| `datasets/pastisr_dataset.py` | `src/data/Pastis.py` |
| `models/linear_probe_head.py` | `src/models/probes.py` 中的 `LayerNormLinearClassifier` |
| `configs/linear_probe_universat_pastisr.py` | Hydra 实验配置中的 lr/wd 列表 |

脚本会依次执行：

1. 加载并冻结 UniverSat backbone。
2. 分别对 train/val/test 提取特征。
3. 对所有 (lr, wd) 组合并行训练 LayerNorm + Linear 探测头。
4. 输出验证集最优 mIoU 及其对应的测试集 mIoU / micro-IoU。

## Fine-tuning（端到端微调）

### 使用 MMSegmentation 风格配置

`configs/ft_universat_pastisr.py` 使用 `../universat/` 提供的 backbone 与分割头，端到端微调整个模型。由于 projects 放在 MMSegmentation 外部，脚本会自动把 `universat_run/mmseg_projects/` 加入 `PYTHONPATH`。

#### 指定 MMSegmentation 路径

脚本会自动通过已安装的 `mmseg` 包定位 MMSegmentation 根目录。如果自动检测失败，请手动设置：

```bash
export MMSEG_ROOT=/path/to/mmsegmentation
```

#### 单卡训练

```bash
cd /path/to/universat
bash universat_run/mmseg_projects/PASTIS_R/tools/train.sh \
    universat_run/mmseg_projects/PASTIS_R/configs/ft_universat_pastisr.py
```

#### 多卡分布式训练（2 卡示例）

```bash
bash universat_run/mmseg_projects/PASTIS_R/tools/dist_train.sh \
    universat_run/mmseg_projects/PASTIS_R/configs/ft_universat_pastisr.py 2
```

#### 从 checkpoint 续训

```bash
bash universat_run/mmseg_projects/PASTIS_R/tools/train.sh \
    universat_run/mmseg_projects/PASTIS_R/configs/ft_universat_pastisr.py \
    /path/to/checkpoint/last.pth
```

#### 测试

```bash
bash universat_run/mmseg_projects/PASTIS_R/tools/test.sh \
    universat_run/mmseg_projects/PASTIS_R/configs/ft_universat_pastisr.py \
    /path/to/checkpoint/latest.pth \
    --eval mIoU --show-dir vis/pastisr_ft/
```

### 使用原始 UniverSat Hydra 实验

若当前环境没有安装 MMSegmentation，可直接使用原仓库实验：

```bash
cd UniverSat
python src/train.py \
    dataset=PastisLP \
    model/network/encoder=UniverSat_Base \
    model=SemSeg \
    model/network=UniverSat_SemSeg \
    model/network/mlp=mlpSemSeg \
    modalities=[s2,s1] \
    trainer.devices=2 \
    trainer.trainer.accelerator=gpu
```

对应的实验配置文件见 `UniverSat/configs/exp/UniverSat_Pastis_FT.yaml`。

## 常见问题

1. **mIoU 极低（~0.005）**
   - 检查 backbone 权重是否加载成功：`init_weights` 会打印 missing/unexpected keys 数量。
   - 确保使用 `UniverSat_Base`（768-dim，12 heads），Tiny 模型无法加载 Base 权重。
   - 从 `model.safetensors` 转换时，必须过滤 `projector__*` 键。

2. **`lp_eval_callback` 报错**
   - 该 callback 依赖 `logger.wandb.group`，直接训练 PASTIS-R 时应在配置中禁用。
   - 标准测试请在训练完成后使用 `src/eval.py` 或 `src/train.py train=False test=True ckpt_path=...`。

3. **`train.log` 写入项目根目录**
   - 默认 Hydra 日志以追加模式写入 `${hydra.runtime.cwd}/train.log`。
   - 可通过自定义 `configs/hydra/job_logging/colorlog.yaml` 重定向到 `${hydra:runtime.output_dir}/train.log` 并覆盖。

4. **终端 progress bar 不显示 val 指标**
   - 验证指标会正常写入 wandb / csv 日志；progress bar 默认不展示，不影响训练。
