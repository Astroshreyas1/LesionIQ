"""
LesionIQ — Fix SWA BN Update
==============================
Re-runs the SWA batch norm update with metadata for 'full' mode checkpoint.
No retraining needed — takes ~2 minutes.

Usage:
    python fix_swa.py
"""

import sys
import torch
import numpy as np
from pathlib import Path
from torch.optim.swa_utils import AveragedModel

sys.path.insert(0, r"C:\Users\Admin\Desktop\models")

from config import DEVICE, BATCH_SIZE, OUTPUT_DIR, FOCAL_GAMMA, FOCAL_ALPHA, LABEL_SMOOTHING
from models import LesionIQHybrid
from dataloader import get_dataloaders
from train import FocalLoss, _validate

# ── Config ────────────────────────────────────────────────────
CKPT_PATH = r"C:\Users\Admin\Desktop\models\output\checkpoints\best_full.pt"

def main():
    print("[FIX-SWA] Loading dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    print(f"[FIX-SWA] Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    mode = ckpt.get("mode", "full")
    
    model = LesionIQHybrid(num_classes=8, meta_dim=13, mode=mode)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    print(f"[FIX-SWA] Model loaded (mode={mode}, epoch={ckpt.get('epoch')}, val_f1={ckpt.get('val_f1', '?')})")

    # Create SWA wrapper
    swa_model = AveragedModel(model).to(DEVICE)

    # Custom BN update with metadata
    print("[FIX-SWA] Running BN statistics update with metadata...")
    swa_model.train()
    momenta = {}
    for module in swa_model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum
            module.momentum = None
            module.num_batches_tracked *= 0

    with torch.no_grad():
        for i, (images, meta, labels) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            meta = meta.to(DEVICE, non_blocking=True)
            swa_model(images, meta)
            if (i + 1) % 50 == 0:
                print(f"  BN update: batch {i+1}/{len(train_loader)}")

    for module, momentum in momenta.items():
        module.momentum = momentum

    # Evaluate SWA model
    alpha = torch.tensor(FOCAL_ALPHA, dtype=torch.float32).to(DEVICE)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha,
                          label_smoothing=LABEL_SMOOTHING).to(DEVICE)
    swa_loss, swa_acc, swa_f1, swa_auc = _validate(swa_model, val_loader, criterion, DEVICE)
    print(f"\n[FIX-SWA] SWA model: val_f1={swa_f1:.4f}  val_auc={swa_auc:.4f}  val_acc={swa_acc:.4f}")

    # Save
    swa_path = str(Path(OUTPUT_DIR) / "checkpoints" / "best_full_swa.pt")
    torch.save({
        "epoch": ckpt.get("epoch"),
        "model_state_dict": swa_model.module.state_dict(),
        "val_f1": swa_f1,
        "val_auc": swa_auc,
        "mode": mode,
        "swa": True,
    }, swa_path)
    print(f"[FIX-SWA] SWA checkpoint saved -> {swa_path}")

    # Compare
    orig_f1 = ckpt.get("val_f1", 0)
    if swa_f1 > orig_f1:
        print(f"[FIX-SWA] SWA improved F1: {orig_f1:.4f} -> {swa_f1:.4f} !")
    else:
        print(f"[FIX-SWA] SWA F1 ({swa_f1:.4f}) vs original ({orig_f1:.4f})")

    print("\nDone.")

if __name__ == "__main__":
    main()
