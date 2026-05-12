"""
LesionIQ Hybrid Classifier — Shared Configuration
===================================================
Edit the placeholder paths and hyper-parameters below before running.
"""

from pathlib import Path

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[1]

# ── Data paths (EDIT THESE) ──────────────────────────────────
TRAIN_IMG_DIR = r"path/to/LesionIQ/Segregated"       # per-class subfolders
TEST_IMG_DIR  = r"path/to/LesionIQ/Test"              # per-class subfolders
METADATA_CSV  = r"path/to/LesionIQ/metadata.csv"      # isic_id, age_approx, sex, region, disease-class
OUTPUT_DIR    = str(BACKEND_ROOT / "output")  # checkpoints, logs, reports

# ── Metadata column names ────────────────────────────────────
META_ID_COL     = "isic_id"
META_AGE_COL    = "age_approx"
META_SEX_COL    = "sex"
META_REGION_COL = "region"
META_LABEL_COL  = "disease-class"

# ── Class configuration ──────────────────────────────────────
NUM_CLASSES = 8
CLASS_NAMES = None  # auto-detected from subfolder names when None

# ── Training hyper-parameters ────────────────────────────────
EPOCHS       = 50
BATCH_SIZE   = 16
GRAD_ACCUM_STEPS = 3
LR           = 1e-4
WEIGHT_DECAY = 0.01
IMG_SIZE     = 384
VAL_SPLIT    = 0.2
SEED         = 42
NUM_WORKERS  = 4

FOCAL_GAMMA = 2.0
FOCAL_ALPHA = [1.0] * 8  # Uniform alpha — class balance is handled by WeightedRandomSampler
                         # Using both sampler + alpha = double compensation that hurts common classes

# ── Early stopping ───────────────────────────────────────────
PATIENCE = 10

# ── Cosine-annealing ─────────────────────────────────────────
# Using plain CosineAnnealingLR (no restarts) to avoid timing conflict:
# WarmRestarts T_0=10 + patience=10 causes early stopping right after a
# LR restart before the model has time to recover in the new cycle.
COSINE_T_MAX = 50  # Decay LR smoothly over all epochs

# ── Swin layer-wise LR decay ────────────────────────────────
SWIN_LR_DECAY = 0.65

# ── Label smoothing (reduces overconfidence on rare classes) ─
LABEL_SMOOTHING = 0.1

# ── Metadata branch ─────────────────────────────────────────
META_EMBED_DIM   = 64   # dimensionality kept high enough to stay relevant
META_AUX_WEIGHT  = 0.3  # auxiliary-loss weight so the metadata branch never goes dormant
META_LR_SCALE    = 3.0  # metadata branch LR = LR * META_LR_SCALE (trains from scratch)

# ── Explainability ───────────────────────────────────────────
CONFIDENCE_THRESHOLD        = 0.7
NUM_EXPLAINABILITY_SAMPLES  = 20
SHAP_BACKGROUND_SAMPLES     = 100

# ── Mixed precision ──────────────────────────────────────────
USE_AMP = True

# ── Device ───────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
