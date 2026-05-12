"""
LesionIQ Hybrid Classifier — Main Entry Point
================================================
Usage:
    python train_classifier.py                          # train hybrid (default)
    python train_classifier.py --model efficientnet     # EfficientNet-B4 only
    python train_classifier.py --model swin             # Swin-Base only
    python train_classifier.py --eval-only --checkpoint path/to/best_model.pt
    python train_classifier.py --fix-swa --checkpoint path/to/best_model.pt
"""

import argparse
import os
import sys
import json
import random
from pathlib import Path

import numpy as np
import torch

from backend.classifier.config import (
    DEVICE, SEED, OUTPUT_DIR, META_AGE_COL, META_SEX_COL, META_REGION_COL,
)
from backend.classifier.dataset import build_dataloaders
from backend.classifier.models import LesionIQHybrid
from backend.classifier.train import train
from backend.classifier.evaluate import evaluate
from backend.classifier.explainability import run_explainability
from torch.optim.swa_utils import AveragedModel


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _meta_feature_names(encoder) -> list:
    """Build human-readable feature names for SHAP from the encoder."""
    names = [META_AGE_COL]
    for cat in encoder.sex_cats:
        names.append(f"{META_SEX_COL}={cat}")
    for cat in encoder.region_cats:
        names.append(f"{META_REGION_COL}={cat}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="LesionIQ Hybrid Classifier")
    parser.add_argument("--mode", choices=["effnet_only", "swin_only", "image_only", "full"],
                        default="full", help="Architecture mode (default: full)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; run evaluation + explainability only")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a saved checkpoint (.pt)")
    parser.add_argument("--skip-explain", action="store_true",
                        help="Skip the explainability suite after evaluation")
    parser.add_argument("--fix-swa", action="store_true",
                        help="Re-run SWA BN update on existing checkpoint (no retraining)")
    args = parser.parse_args()

    _set_seed()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────
    (train_loader, val_loader, test_loader,
     meta_encoder, class_names, class_weights) = build_dataloaders()

    # ── Model ─────────────────────────────────────────────────
    model = LesionIQHybrid(meta_dim=meta_encoder.dim, mode=args.mode)

    # ── Load checkpoint (if provided) ─────────────────────────
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[CKPT] Loaded {args.checkpoint}  "
              f"(epoch={ckpt.get('epoch')}, val_f1={ckpt.get('val_f1', '?')})")

    # ── Fix SWA (re-run BN update only) ───────────────────────
    if args.fix_swa:
        if not args.checkpoint:
            print("ERROR: --fix-swa requires --checkpoint path/to/best_model.pt")
            sys.exit(1)
        print("\n  Re-running SWA batch norm update with metadata...")
        swa_model = AveragedModel(model).to(DEVICE)
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
            for images, meta, labels in train_loader:
                images = images.to(DEVICE, non_blocking=True)
                meta = meta.to(DEVICE, non_blocking=True)
                swa_model(images, meta)

        for module, momentum in momenta.items():
            module.momentum = momentum

        swa_path = str(Path(OUTPUT_DIR) / "checkpoints" / "best_model_swa.pt")
        torch.save({
            "model_state_dict": swa_model.module.state_dict(),
            "mode": args.mode,
            "swa": True,
        }, swa_path)
        print(f"  SWA checkpoint saved -> {swa_path}")

        # Evaluate SWA model
        from backend.classifier.train import FocalLoss
        from backend.classifier.config import FOCAL_GAMMA, FOCAL_ALPHA, LABEL_SMOOTHING
        alpha = torch.tensor(FOCAL_ALPHA, dtype=torch.float32).to(DEVICE)
        criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha,
                              label_smoothing=LABEL_SMOOTHING).to(DEVICE)
        from backend.classifier.train import _validate
        swa_loss, swa_acc, swa_f1, swa_auc = _validate(swa_model, val_loader, criterion, DEVICE)
        print(f"  SWA model: val_f1={swa_f1:.4f}  val_auc={swa_auc:.4f}  val_acc={swa_acc:.4f}")
        print("\nDone.")
        return

    # ── Train ─────────────────────────────────────────────────
    if not args.eval_only:
        best_path = train(model, train_loader, val_loader, class_weights)
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[CKPT] Reloaded best checkpoint for evaluation")

    model.to(DEVICE)

    # ── Evaluate ──────────────────────────────────────────────
    report = evaluate(model, test_loader, class_names)

    # ── Explainability ────────────────────────────────────────
    if not args.skip_explain:
        feat_names = _meta_feature_names(meta_encoder)
        run_explainability(model, test_loader, class_names, feat_names)

    # ── Save run config ───────────────────────────────────────
    run_info = {
        "mode": args.mode,
        "seed": SEED,
        "device": DEVICE,
        "class_names": class_names,
        "meta_dim": meta_encoder.dim,
        "meta_features": _meta_feature_names(meta_encoder),
    }
    info_path = Path(OUTPUT_DIR) / "run_info.json"
    with open(info_path, "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"\nDone. All outputs → {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
