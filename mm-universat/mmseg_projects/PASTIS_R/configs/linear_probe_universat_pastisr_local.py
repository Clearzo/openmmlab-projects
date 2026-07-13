"""Local linear-probe config for UniverSat on PASTIS-R.

Adjusts paths to the NAS-mounted PASTIS-R copy and local UniverSat weights.
Copy or symlink this file to overwrite ``linear_probe_universat_pastisr.py``
when running in the current working environment.
"""

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data_root = "/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/PASTIS-R/"
norm_path = "/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/PASTIS-R/"

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
checkpoint = "/mnt/ht2-nas2/EO_test/openmmlab-archive/checkpoints/UniverSat/model.safetensors"
embed_dim = 768
num_heads = 12
patch_size = 40
output_grid = 128
gating = True

# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------
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
