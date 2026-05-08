"""
LesionIQ -- Inference + Explainability Pipeline
==================================================
5-stage pipeline: Input → Preprocess → Classify → Explain → SLM Output

Each image produces a diagnostic bundle:
  output/<image_name>/
    ├── original.png        # Preprocessed input (384×384)
    ├── gradcam.png         # Grad-CAM++ heatmap overlay
    ├── attention.png       # Swin attention rollout overlay
    └── diagnosis.json      # Full diagnostic data for SLM

Usage:
  python classifier/inference.py --image lesion.png
  python classifier/inference.py --image img.png --age 65 --sex male --site "head/neck"
  python classifier/inference.py --image img.png --age NA --sex NA --site NA
  python classifier/inference.py --dir path/to/images/ --output-dir ./results
"""

import os, sys, json, argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

# ---------------------------------------------------------------------------
#  Resolve paths — add REPO_ROOT so `from preprocessing import ...` works
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
CKPT_DIR   = REPO_ROOT / "checkpoints"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CLASS_FULL  = {
    "MEL": "Melanoma", "NV": "Melanocytic Nevus", "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis", "BKL": "Benign Keratosis", "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion", "SCC": "Squamous Cell Carcinoma",
}
MALIGNANT   = {"MEL", "BCC", "AK", "SCC"}
IMG_SIZE    = 384
NUM_CLASSES = 8
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

SEX_COLS  = ["sex_female", "sex_male", "sex_unknown"]
SITE_COLS = [
    "site_anterior torso", "site_head/neck", "site_lateral torso",
    "site_lower extremity", "site_oral/genital", "site_palms/soles",
    "site_posterior torso", "site_unknown", "site_upper extremity",
]
ALL_SITES     = [s.replace("site_", "") for s in SITE_COLS]
META_FEATURES = ["age_approx"] + SEX_COLS + SITE_COLS  # 13 features


# ===================================================================
#  Stage 1 — Interactive Metadata Input
# ===================================================================

def _prompt_metadata_interactive():
    """Prompt user for each metadata field with NA option."""
    print("\n╔══════════════════════════════════════╗")
    print("║        Patient Metadata Input        ║")
    print("╚══════════════════════════════════════╝\n")

    # --- Age ---
    while True:
        raw = input("  Age (years, or NA): ").strip()
        if raw.upper() == "NA" or raw == "":
            age = None; break
        try:
            age = float(raw)
            if 0 <= age <= 120: break
            print("    → Please enter a value between 0-120.")
        except ValueError:
            print("    → Enter a number or NA.")

    # --- Sex ---
    while True:
        raw = input("  Sex (male / female / NA): ").strip().lower()
        if raw in ("na", ""):
            sex = None; break
        if raw in ("male", "female"):
            sex = raw; break
        print("    → Enter male, female, or NA.")

    # --- Site ---
    site_options = [
        "anterior torso", "head/neck", "lateral torso", "lower extremity",
        "oral/genital", "palms/soles", "posterior torso", "upper extremity",
    ]
    print("\n  Anatomical site:")
    for i, s in enumerate(site_options, 1):
        print(f"    {i}. {s}")
    print(f"    9. NA (unknown)")

    while True:
        raw = input("  Select [1-9]: ").strip()
        if raw.upper() == "NA" or raw == "9" or raw == "":
            site = None; break
        try:
            idx = int(raw)
            if 1 <= idx <= 8:
                site = site_options[idx - 1]; break
            print("    → Enter 1-9.")
        except ValueError:
            print("    → Enter a number 1-9 or NA.")

    return age, sex, site


def encode_metadata(age=None, sex=None, site=None):
    """Encode patient metadata into a 13-d tensor matching training format."""
    meta = np.zeros(13, dtype=np.float32)
    meta[0] = min(age, 100) / 100.0 if age is not None else 0.5
    sex = (sex or "unknown").lower()
    if sex == "female":   meta[1] = 1.0
    elif sex == "male":   meta[2] = 1.0
    else:                 meta[3] = 1.0
    site = (site or "unknown").lower()
    matched = False
    for i, s in enumerate(ALL_SITES):
        if s == site:
            meta[4 + i] = 1.0
            matched = True
            break
    if not matched:
        meta[4 + ALL_SITES.index("unknown")] = 1.0
    return torch.tensor(meta).unsqueeze(0)


# ===================================================================
#  Stage 2 — Preprocessing (delegates to preprocessing package)
# ===================================================================

from preprocessing import run_pipeline as _run_preprocess_pipeline


def preprocess_image(image_path):
    """Full Layer 0 pipeline + model normalization → tensor [1,3,384,384]."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    img_bgr = _run_preprocess_pipeline(str(image_path), target_size=IMG_SIZE)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transform(image=img_rgb)["image"].unsqueeze(0)


def get_display_image(image_path):
    """Preprocessed image as [0,1] RGB numpy for visualization overlays."""
    img_bgr = _run_preprocess_pipeline(str(image_path), target_size=IMG_SIZE)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32) / 255.0


# ===================================================================
#  Stage 3a — Model (self-contained, no config.py dependency)
# ===================================================================

def build_model(mode="full", checkpoint_path=None):
    import timm

    class LesionIQHybrid(nn.Module):
        def __init__(self, num_classes=NUM_CLASSES, meta_dim=13, mode='full'):
            super().__init__()
            self.mode = mode
            if mode in ('effnet_only', 'image_only', 'full'):
                self.effnet = timm.create_model(
                    'efficientnet_b4', pretrained=False, num_classes=0)
            if mode in ('swin_only', 'image_only', 'full'):
                self.swin = timm.create_model(
                    'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k',
                    pretrained=False, num_classes=0)
            if mode == 'full':
                self.meta_mlp = nn.Sequential(
                    nn.Linear(meta_dim, 64), nn.BatchNorm1d(64),
                    nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(64, 32), nn.ReLU(),
                )
            fusion_dim = {'effnet_only': 1792, 'swin_only': 1024,
                          'image_only': 2816, 'full': 2848}[mode]
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 512), nn.BatchNorm1d(512),
                nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )

        def forward(self, img, meta=None):
            features = []
            if self.mode in ('effnet_only', 'image_only', 'full'):
                features.append(self.effnet(img))
            if self.mode in ('swin_only', 'image_only', 'full'):
                features.append(self.swin(img))
            if self.mode == 'full' and meta is not None:
                features.append(self.meta_mlp(meta))
            return self.classifier(torch.cat(features, dim=1))

    model = LesionIQHybrid(mode=mode).to(DEVICE)
    if checkpoint_path is None:
        checkpoint_path = str(CKPT_DIR / f"best_{mode}.pt")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[OK] Model loaded: {mode} from {checkpoint_path}")
    return model


# ===================================================================
#  Stage 3b — 4-way TTA prediction with temperature scaling
# ===================================================================

@torch.no_grad()
def predict(model, image_tensor, meta_tensor=None, temperature=1.0, scales=None):
    """4-way TTA prediction with optional temperature + DiffEvo scaling."""
    image_tensor = image_tensor.to(DEVICE)
    if meta_tensor is not None:
        meta_tensor = meta_tensor.to(DEVICE)

    def _fwd(x):
        out = model(x, meta_tensor)
        return out[0] if isinstance(out, tuple) else out

    logits = (_fwd(image_tensor)
              + _fwd(torch.flip(image_tensor, dims=[3]))
              + _fwd(torch.flip(image_tensor, dims=[2]))
              + _fwd(torch.flip(image_tensor, dims=[2, 3]))) / 4.0

    # Temperature scaling (applied before softmax)
    logits = logits / temperature

    probs = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]

    # DiffEvo threshold scaling (applied after softmax)
    if scales is not None:
        probs = probs * scales
        probs = probs / probs.sum()

    return probs


# ===================================================================
#  Stage 4a — Grad-CAM++ (EfficientNet-B4 branch)
# ===================================================================

class GradCAMPP:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, image, meta, class_idx=None):
        self.model.eval()
        image = image.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(image, meta)
            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()
            score = logits[0, class_idx]
            self.model.zero_grad()
            score.backward()

        grads = self.gradients[0]
        acts  = self.activations[0]
        alpha_num   = grads.pow(2)
        alpha_denom = 2.0 * grads.pow(2) + (acts * grads.pow(3)).sum(
            dim=(1, 2), keepdim=True)
        alpha_denom = torch.where(
            alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
        alpha   = alpha_num / alpha_denom
        weights = (alpha * F.relu(grads)).sum(dim=(1, 2))
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


def _get_effnet_target_layer(model):
    if hasattr(model, 'effnet'):
        return model.effnet.blocks[-1]
    return None


def _make_heatmap_overlay(image_np, cam, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """Overlay heatmap on [H,W,3] image in [0,1]. Returns [0,1] RGB."""
    h, w = image_np.shape[:2]
    cam_resized = cv2.resize(cam.astype(np.float32), (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = heatmap[:, :, ::-1].astype(np.float32) / 255.0
    return np.clip(alpha * heatmap + (1 - alpha) * image_np, 0, 1)


# ===================================================================
#  Stage 4b — Swin Attention Rollout
# ===================================================================

def _extract_swin_attention(model, image):
    """Gradient-based spatial attribution for the Swin branch.

    SwinV2 uses windowed attention, which makes global attention rollout
    infeasible.  Instead we hook the last feature map before pooling and
    compute gradient × activation (Grad-AM) with respect to the predicted
    class.  This produces a clean spatial heatmap that highlights the
    regions the Swin branch relies on most.
    """
    if not hasattr(model, 'swin'):
        return None

    feature_map = {}

    def _fwd_hook(module, inp, out):
        out.retain_grad()
        feature_map['feat'] = out

    # Hook the norm layer right before the global average pool
    hook = model.swin.norm.register_forward_hook(_fwd_hook)

    # Need gradients for this pass
    img = image.clone().requires_grad_(True)
    with torch.enable_grad():
        logits = model.swin(img)
        pred_idx = logits.argmax(dim=1).item()
        score = logits[0, pred_idx]
        model.swin.zero_grad()
        score.backward()

    hook.remove()

    feat = feature_map.get('feat')
    if feat is None:
        return None

    # feat shape: [1, H, W, C] for SwinV2 (spatial, channels last)
    grad = feat.grad[0]      # [H, W, C] or [N, C]
    act  = feat.detach()[0]

    # Grad-AM: element-wise gradient × activation, summed over channels
    spatial = (grad * act).sum(dim=-1)    # [H, W] or [N]
    spatial = F.relu(spatial)             # keep positive attributions
    spatial = spatial.cpu().numpy()

    # Handle both [H, W] (4D norm output) and [N] (3D flattened) cases
    if spatial.ndim == 1:
        side = int(np.sqrt(spatial.shape[0]))
        spatial = spatial[:side * side].reshape(side, side)

    spatial = (spatial - spatial.min()) / (spatial.max() - spatial.min() + 1e-8)
    return spatial


# ===================================================================
#  Stage 4c — SHAP (perturbation-based metadata attribution)
# ===================================================================

def _compute_shap_values(model, image_tensor, meta_tensor):
    """Perturbation-based feature attribution for metadata features.

    For each metadata feature, perturb it to its baseline (zero/unknown)
    and measure the change in predicted probability.  This gives a
    lightweight SHAP-like attribution without needing the shap library.
    """
    if meta_tensor is None:
        return None

    model.eval()
    image_t = image_tensor.to(DEVICE)
    meta_t  = meta_tensor.to(DEVICE)

    with torch.no_grad():
        base_logits = model(image_t, meta_t)
        base_probs  = torch.softmax(base_logits.float(), dim=1).cpu().numpy()[0]
        pred_class  = base_probs.argmax()

    baseline_meta = torch.zeros_like(meta_t)
    shap_values   = {}

    for i, feat_name in enumerate(META_FEATURES):
        perturbed = meta_t.clone()
        perturbed[0, i] = baseline_meta[0, i]
        with torch.no_grad():
            pert_logits = model(image_t, perturbed)
            pert_probs  = torch.softmax(pert_logits.float(), dim=1).cpu().numpy()[0]
        delta = float(base_probs[pred_class] - pert_probs[pred_class])
        shap_values[feat_name] = round(delta, 6)

    return shap_values


# ===================================================================
#  Stage 5 — Full diagnostic pipeline (per image)
# ===================================================================



def diagnose_image(model, image_path, meta_tensor, scales, temperature,
                   output_dir, raw_meta, mel_threshold=None):
    """Run the full 5-stage pipeline on a single image.

    Produces: original.png, gradcam.png, attention.png, diagnosis.json
    """
    stem       = Path(image_path).stem
    img_out_dir = Path(output_dir) / stem
    img_out_dir.mkdir(parents=True, exist_ok=True)

    # -- Preprocess --
    img_tensor  = preprocess_image(image_path)
    img_display = get_display_image(image_path)

    # Save preprocessed original
    cv2.imwrite(str(img_out_dir / "original.png"),
                cv2.cvtColor((img_display * 255).astype(np.uint8),
                             cv2.COLOR_RGB2BGR))

    # -- Classify (TTA + temperature + DiffEvo) --
    probs   = predict(model, img_tensor, meta_tensor, temperature, scales)

    # MEL safety override: if raw MEL prob exceeds threshold, force MEL
    mel_safety_applied = False
    if mel_threshold is not None:
        raw_mel_prob = float(probs[CLASS_NAMES.index('MEL')])
        if raw_mel_prob >= mel_threshold:
            mel_safety_applied = True

    ranked  = np.argsort(probs)[::-1]
    if mel_safety_applied:
        pred_cls = 'MEL'
        pred_conf = float(probs[CLASS_NAMES.index('MEL')])
    else:
        pred_cls  = CLASS_NAMES[ranked[0]]
        pred_conf = float(probs[ranked[0]])
    is_mal    = pred_cls in MALIGNANT

    # -- Grad-CAM++ --
    gradcam_file = None
    if hasattr(model, 'effnet'):
        target_layer = _get_effnet_target_layer(model)
        if target_layer is not None:
            gcpp = GradCAMPP(model, target_layer)
            cam  = gcpp.generate(
                img_tensor.to(DEVICE),
                meta_tensor.to(DEVICE) if meta_tensor is not None else None,
                class_idx=int(ranked[0]))
            overlay = _make_heatmap_overlay(img_display, cam, cv2.COLORMAP_JET)
            gradcam_file = "gradcam.png"
            cv2.imwrite(str(img_out_dir / gradcam_file),
                        cv2.cvtColor((overlay * 255).astype(np.uint8),
                                     cv2.COLOR_RGB2BGR))

    # -- Swin Attention --
    attention_file = None
    if hasattr(model, 'swin'):
        attn_map = _extract_swin_attention(model, img_tensor.to(DEVICE))
        if attn_map is not None:
            overlay = _make_heatmap_overlay(img_display, attn_map,
                                           cv2.COLORMAP_INFERNO)
            attention_file = "attention.png"
            cv2.imwrite(str(img_out_dir / attention_file),
                        cv2.cvtColor((overlay * 255).astype(np.uint8),
                                     cv2.COLOR_RGB2BGR))

    # -- SHAP (metadata) --
    shap_vals = None
    if meta_tensor is not None:
        shap_vals = _compute_shap_values(model, img_tensor, meta_tensor)

    # -- Build diagnosis.json (SLM input) --
    diagnosis = {
        "version": "1.0",
        "image": Path(image_path).name,
        "model": {
            "mode": model.mode,
            "temperature": round(temperature, 4),
            "thresholds_applied": scales is not None,
            "mel_safety_threshold": round(mel_threshold, 3) if mel_threshold else None,
            "mel_safety_triggered": mel_safety_applied,
        },
        "prediction": {
            "class": pred_cls,
            "class_full": CLASS_FULL[pred_cls],
            "confidence": round(pred_conf, 4),
            "is_malignant": is_mal,
        },
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]), 4) for i in ranked
        },
        "top3": [
            {"class": CLASS_NAMES[ranked[i]],
             "full_name": CLASS_FULL[CLASS_NAMES[ranked[i]]],
             "probability": round(float(probs[ranked[i]]), 4),
             "is_malignant": CLASS_NAMES[ranked[i]] in MALIGNANT}
            for i in range(3)
        ],
        "clinical_flags": {
            "malignant_total_prob": round(float(
                sum(probs[CLASS_NAMES.index(c)] for c in MALIGNANT)), 4),
            "requires_biopsy": is_mal and pred_conf >= 0.5,
            "low_confidence": pred_conf < 0.5,
            "differential_diagnosis": pred_conf < 0.7,
        },
        "explainability": {
            "gradcam": gradcam_file,
            "attention": attention_file,
            "shap_metadata": shap_vals,
        },
        "metadata_input": {
            "age": raw_meta.get("age"),
            "sex": raw_meta.get("sex"),
            "site": raw_meta.get("site"),
        },
    }

    with open(str(img_out_dir / "diagnosis.json"), "w") as f:
        json.dump(diagnosis, f, indent=2)

    # Console summary
    tag = "** MALIGNANT **" if is_mal else "benign"
    print(f"\n  {stem}: {CLASS_FULL[pred_cls]} ({pred_cls}) "
          f"[{pred_conf:.1%}] {tag}")
    print(f"    -> {img_out_dir}")

    return diagnosis


# ===================================================================
#  CLI entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LesionIQ — Inference + Explainability Pipeline")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image")
    group.add_argument("--dir",   type=str, help="Path to directory of images")

    parser.add_argument("--mode", type=str, default="full",
                        choices=["effnet_only", "swin_only",
                                 "image_only", "full"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./output/inference")

    # Metadata — pass "NA" or omit for unknown/population defaults
    parser.add_argument("--age",  type=str, default=None,
                        help="Patient age (number or NA)")
    parser.add_argument("--sex",  type=str, default=None,
                        choices=["male", "female", "unknown", "NA"])
    parser.add_argument("--site", type=str, default=None,
                        help="Anatomical site (e.g. head/neck) or NA")

    parser.add_argument("--no-scales",      action="store_true",
                        help="Disable DiffEvo threshold scaling")
    parser.add_argument("--no-temperature", action="store_true",
                        help="Disable temperature scaling")
    parser.add_argument("--no-mel-safety",  action="store_true",
                        help="Disable MEL recall safety threshold")
    parser.add_argument("--interactive",    action="store_true",
                        help="Force interactive metadata prompts")

    args = parser.parse_args()

    # ── Load model ──
    model = build_model(args.mode, args.checkpoint)

    # ── Load DiffEvo scales ──
    scales = None
    if not args.no_scales:
        scales_path = CKPT_DIR / "optimal_scales.npy"
        if scales_path.exists():
            scales = np.load(str(scales_path))
            print("[OK] DiffEvo threshold scales loaded")

    # ── Load temperature ──
    temperature = 1.0
    if not args.no_temperature:
        temp_path = CKPT_DIR / "optimal_temperature.npy"
        if temp_path.exists():
            temperature = float(np.load(str(temp_path)))
            print(f"[OK] Temperature scaling: T={temperature:.4f}")

    # ── Load MEL safety threshold ──
    mel_threshold = None
    if not args.no_mel_safety:
        mel_path = CKPT_DIR / "mel_safety_threshold.npy"
        if mel_path.exists():
            mel_threshold = float(np.load(str(mel_path)))
            print(f"[OK] MEL safety threshold: {mel_threshold:.3f}")

    # ── Metadata ──
    # Determine if we need interactive prompts:
    #   - Single image mode + no metadata CLI args + not piped stdin
    has_cli_meta = any([args.age, args.sex, args.site])
    want_interactive = (args.interactive
                        or (args.image and not has_cli_meta
                            and sys.stdin.isatty()))

    if args.mode == "full":
        if want_interactive and not args.dir:
            age, sex, site = _prompt_metadata_interactive()
        else:
            # Parse CLI args — treat "NA" as None
            age  = None
            if args.age and args.age.upper() != "NA":
                try:    age = float(args.age)
                except: age = None
            sex  = args.sex if args.sex and args.sex.upper() != "NA" else None
            site = args.site if args.site and args.site.upper() != "NA" else None

        meta_tensor = encode_metadata(age, sex, site)
        raw_meta = {
            "age":  int(age) if age is not None else None,
            "sex":  sex,
            "site": site,
        }

        parts = []
        if age is not None: parts.append(f"age={int(age)}")
        if sex:             parts.append(f"sex={sex}")
        if site:            parts.append(f"site={site}")
        print(f"[OK] Metadata: {', '.join(parts) if parts else 'NA (population defaults)'}")
    else:
        meta_tensor = None
        raw_meta = {"age": None, "sex": None, "site": None}

    # ── Collect images ──
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    if args.image:
        image_paths = [args.image]
    else:
        image_paths = sorted([
            str(f) for f in Path(args.dir).iterdir()
            if f.suffix.lower() in extensions
        ])
        print(f"[OK] Found {len(image_paths)} images in {args.dir}")

    # ── Run pipeline ──
    print(f"\n{'='*55}")
    print(f"  LesionIQ Diagnostic Pipeline")
    print(f"  Mode: {args.mode} | Temp: {temperature:.4f} | "
          f"Scales: {'on' if scales is not None else 'off'}")
    print(f"{'='*55}")

    all_results = []
    for img_path in image_paths:
        try:
            result = diagnose_image(
                model, img_path, meta_tensor, scales, temperature,
                args.output_dir, raw_meta, mel_threshold)
            all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] {Path(img_path).name}: {e}")
            import traceback; traceback.print_exc()

    # ── Batch summary ──
    if len(all_results) > 1:
        print(f"\n{'='*55}")
        print(f"  BATCH SUMMARY ({len(all_results)} images)")
        print(f"{'='*55}")
        from collections import Counter
        counts = Counter(r["prediction"]["class"] for r in all_results)
        for cls in CLASS_NAMES:
            if counts[cls] > 0:
                mal = " [!]" if cls in MALIGNANT else ""
                print(f"    {cls:>5s}: {counts[cls]:3d}{mal}")
        mal_count = sum(1 for r in all_results
                        if r["prediction"]["is_malignant"])
        print(f"\n  Malignant: {mal_count}/{len(all_results)}")

    print(f"\n  Output: {args.output_dir}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
