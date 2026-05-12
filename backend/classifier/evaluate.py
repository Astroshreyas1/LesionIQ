"""
LesionIQ Hybrid Classifier — Evaluation / Clinical Test Suite
==============================================================
Runs on the held-out test set and produces:
  • Accuracy, Macro-F1, Per-class F1, AUC-ROC, Precision, Recall
  • Sensitivity & Specificity per class
  • Brier score (calibration)
  • Confusion matrix (PNG)
  • Full classification report (JSON)
  • Console summary table
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
    brier_score_loss,
)

from backend.classifier.config import DEVICE, USE_AMP, OUTPUT_DIR, NUM_CLASSES, CONFIDENCE_THRESHOLD
from backend.classifier.models import HybridClassifier

# ── Collect predictions ──────────────────────────────────────

@torch.no_grad()
def _gather_predictions(
    model: HybridClassifier,
    loader: DataLoader,
    device: str = DEVICE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (all_probs  [N, C], all_preds  [N], all_labels  [N])."""
    model.eval()
    probs_list, labels_list = [], []

    for images, meta, labels in loader:
        images = images.to(device, non_blocking=True)
        meta   = meta.to(device, non_blocking=True)

        with autocast(enabled=USE_AMP):
            logits, _ = model(images, meta)

        probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
        labels_list.append(labels.numpy())

    all_probs  = np.concatenate(probs_list, axis=0)
    all_preds  = all_probs.argmax(axis=1)
    all_labels = np.concatenate(labels_list, axis=0)
    return all_probs, all_preds, all_labels

# ── Per-class sensitivity & specificity ──────────────────────

def _sensitivity_specificity(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    sens, spec = {}, {}
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens[c] = float(tp / (tp + fn)) if (tp + fn) else 0.0
        spec[c] = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return sens, spec

# ── Brier score (multi-class) ────────────────────────────────

def _brier_score_multiclass(y_true: np.ndarray, probs: np.ndarray) -> float:
    one_hot = np.eye(probs.shape[1])[y_true]
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())

# ── Confusion matrix plot ────────────────────────────────────

def _plot_confusion_matrix(
    cm: np.ndarray, class_names: List[str], save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# ── Main evaluation entry point ──────────────────────────────

def evaluate(
    model: HybridClassifier,
    test_loader: DataLoader,
    class_names: List[str],
    device: str = DEVICE,
) -> Dict:
    """Run full clinical evaluation suite on the test set.

    Returns a metrics dictionary and saves artefacts to OUTPUT_DIR/reports/.
    """
    report_dir = Path(OUTPUT_DIR) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    probs, preds, labels = _gather_predictions(model, test_loader, device)
    num_cls = probs.shape[1]

    # ── Core metrics ──────────────────────────────────────────
    acc       = accuracy_score(labels, preds)
    macro_f1  = f1_score(labels, preds, average="macro")
    per_f1    = f1_score(labels, preds, average=None).tolist()
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall    = recall_score(labels, preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    sens, spec = _sensitivity_specificity(labels, preds, num_cls)
    brier = _brier_score_multiclass(labels, probs)

    # ── Confidence analysis ───────────────────────────────────
    max_conf = probs.max(axis=1)
    uncertain_mask = max_conf < CONFIDENCE_THRESHOLD
    n_uncertain = int(uncertain_mask.sum())
    pct_uncertain = n_uncertain / len(labels) * 100

    # ── Confusion matrix ──────────────────────────────────────
    cm = confusion_matrix(labels, preds, labels=list(range(num_cls)))
    cm_path = str(report_dir / "confusion_matrix.png")
    _plot_confusion_matrix(cm, class_names, cm_path)

    # ── Build report dict ─────────────────────────────────────
    report = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(precision, 4),
        "macro_recall": round(recall, 4),
        "auc_roc_ovr": round(auc, 4) if not np.isnan(auc) else None,
        "brier_score": round(brier, 5),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "uncertain_predictions": n_uncertain,
        "uncertain_pct": round(pct_uncertain, 2),
        "per_class": {},
    }
    for i, name in enumerate(class_names):
        report["per_class"][name] = {
            "f1": round(per_f1[i], 4),
            "sensitivity": round(sens[i], 4),
            "specificity": round(spec[i], 4),
        }

    report["sklearn_classification_report"] = classification_report(
        labels, preds, target_names=class_names, output_dict=True, zero_division=0,
    )

    # ── Save JSON ─────────────────────────────────────────────
    json_path = str(report_dir / "classification_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Console summary ───────────────────────────────────────
    print(f"\n{'='*62}")
    print(f" TEST RESULTS")
    print(f"{'='*62}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Macro F1      : {macro_f1:.4f}")
    print(f"  Macro Prec.   : {precision:.4f}")
    print(f"  Macro Recall  : {recall:.4f}")
    print(f"  AUC-ROC (OVR) : {auc:.4f}")
    print(f"  Brier Score   : {brier:.5f}")
    print(f"  Uncertain     : {n_uncertain}/{len(labels)} ({pct_uncertain:.1f}%)"
          f"  [threshold={CONFIDENCE_THRESHOLD}]")
    print(f"{'─'*62}")
    print(f"  {'Class':<22} {'F1':>6}  {'Sens':>6}  {'Spec':>6}")
    print(f"  {'─'*46}")
    for i, name in enumerate(class_names):
        print(f"  {name:<22} {per_f1[i]:6.4f}  {sens[i]:6.4f}  {spec[i]:6.4f}")
    print(f"{'='*62}")
    print(f"  Confusion matrix → {cm_path}")
    print(f"  Full report      → {json_path}\n")

    return report
