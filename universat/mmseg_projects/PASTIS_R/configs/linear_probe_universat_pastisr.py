"""Linear-probe config for UniverSat on PASTIS-R semantic segmentation.

This is a plain Python config (not Hydra) consumed by
``tools/linear_probe.py``.
"""

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data_root = "data/PASTIS-R/"
norm_path = "data/PASTIS-R/"

modalities = ["s2", "s1"]
train_folds = [1, 2, 3]
val_folds = [4]
test_folds = [5]

reference_date = "2018-01-01"
nb_split = 1
num_classes = 20
normalize = True
temporal_dropout = 200

# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
checkpoint = "checkpoints/UniverSat/model.safetensors"  # HF released weights
embed_dim = 768
num_heads = 12
patch_size = 40       # physical patch size in meters
output_grid = 128     # backbone output side length (128x128 tokens for PASTIS-R 1280m)
gating = True

# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------
# Each output token predicts a patch_size_px x patch_size_px region.
# For output_grid=128 and PASTIS-R 128x128 pixel labels, set patch_size_px=1.
patch_size_px = 1

lr_list = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
weight_decay_list = [0, 1e-2, 1e-1]
batch_size = 1024
max_epochs = 50

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
global_batch_size = 1
num_workers = 4
