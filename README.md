# LesionIQ — Hybrid Dermatoscopy Classifier with Synthetic Data Augmentation

A research-grade skin lesion classification pipeline combining dual-backbone deep learning (EfficientNet-B4 + Swin Transformer V2) with clinical metadata fusion and synthetic data augmentation via StyleGAN2-ADA. Built for the [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/) 8-class dermoscopy classification task.

---

## Architecture

```
                    ┌──────────────────────────────┐
    384×384 Image → │ EfficientNet-B4  (17.5M)     │ → 1792-d
                    └──────────────────────────────┘
                                                     ╲
                                                      → Concat → FC(512) → BN → ReLU → Dropout(0.5) → FC(8)
                                                     ╱
                    ┌──────────────────────────────┐╱
    384×384 Image → │ SwinV2-Base      (86.9M)     │ → 1024-d
                    └──────────────────────────────┘
                                                     ╲
    Patient Meta* → MLP(13→64→32)  (0.005M) ──────────→ +32-d  (full mode only)
```

*Note: The 13 metadata features are derived from one-hot encoding the anatomical site, sex, and normalized age.*

**Total parameters:** 104.4M (backbones) + 1.45M (classifier head) = **105.85M**

### Ablation Modes
| Mode | Backbones | Metadata | Purpose |
|------|-----------|----------|---------|
| `effnet_only` | EfficientNet-B4 | ✗ | Baseline A |
| `swin_only` | SwinV2-Base | ✗ | Baseline B |
| `image_only` | Both | ✗ | Fusion ablation |
| `full` | Both | ✓ | Complete hybrid |

---

## Project Structure

```
LesionIQ/
├── classifier/              # Core classification pipeline
│   ├── config.py            # Hyperparameters and paths
│   ├── models.py            # LesionIQHybrid architecture
│   ├── dataloader.py        # Dataset, augmentations, weighted sampling
│   ├── train.py             # Training loop (CutMix, SWA, 4-way TTA, progressive unfreezing)
│   ├── evaluate.py          # Test-set evaluation suite
│   ├── explainability.py    # Grad-CAM++, SHAP, attention maps, calibration
│   ├── post_training.py     # Threshold tuning, ensembling, temperature scaling
│   ├── run_ablation.py      # Automated 4-experiment ablation runner
│   └── train_classifier.py  # CLI entry point
│
├── preprocessing/           # Image preprocessing & cleanup
│   ├── dull_razor.py                # Hair removal using morphological operations
│   ├── shades_of_grey.py            # Color normalization
│   ├── apply_clahe.py               # Contrast enhancement (LAB color space)
│   ├── remove_circular_border.py    # Vignette/border removal
│   ├── postprocess.py               # Histogram matching, blur correction
│   ├── finalize_dataset.py          # Dataset assembly and validation
│   ├── SSIM_Final.py                # SSIM evaluation logic
│   └── SSIM for ISIC 2019/          # SSIM structural similarity tracking reports
│
├── synthetic/               # Synthetic data generation (StyleGAN2-ADA)
│   ├── quality_check.py             # FID, SSIM, perceptual quality metrics
│   └── stylegan2/
│       └── train_stylegan2ada.py    # StyleGAN2-ADA training wrapper
│
├── data/                    # Data preparation utilities
│   ├── training.py          # LesionIQ dataset class (standalone)
│   ├── jitter_metadata.py   # Gaussian jitter for metadata augmentation
│   └── update_lesioniq_metadata.py  # Metadata CSV processing
│
├── checkpoints/             # Trained model weights (.pt files)
│
├── docs/
│   └── fine_tuning_log.txt  # Complete hyperparameter changelog
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Key Technical Features

### Training Optimizations
- **CutMix** (30% probability) — preserves lesion texture better than MixUp
- **4-Way Test-Time Augmentation** — original + H-flip + V-flip + H+V flip
- **Stochastic Weight Averaging (SWA)** — starts epoch 20, finds wider loss basins
- **Progressive Backbone Unfreezing** — epoch 15: last backbone stages unlock at 0.1× head LR
- **Dynamic Patience** — increases from 10 → 15 after unfreezing to protect SWA
- **Focal Loss** with label smoothing (γ=2.0, ε=0.1)
- **WeightedRandomSampler** for class-balanced training

### Post-Training Pipeline
- **Per-Class Threshold Tuning** via Nelder-Mead optimization with 80/20 overfit protection
- **4-Model Ensemble** — simple average + optimized weighted average
- **Temperature Scaling** — LBFGS-optimized calibration for clinical confidence scores

### Clinical Explainability (Work in Progress)
- **EfficientNet Branch**: Grad-CAM++ heatmaps for CNN feature visualization.
- **SwinV2 Branch**: Grad-based attention weights for transformer patch attribution.
- **Metadata Branch (MLP)**: SHAP DeepExplainer for tabular feature importance.
- **Final Output Generation**: Outputs from all explainability methods are fed into a Small Language Model (SLM) to generate a cohesive, human-readable clinical explanation.

### Frontend UI (Work in Progress)
- A web-based clinical dashboard is currently under development to serve the model predictions and SLM explanations to end-users.

---

## 📈 Results

*Full results pending — model training in progress.*

| Mode | Val AUC | Val F1 (raw) | Val F1 (tuned) | Val Acc |
|------|---------|--------------|----------------|---------|
| `effnet_only` | — | — | — | — |
| `swin_only` | — | — | — | — |
| `image_only` | — | — | — | — |
| `full` | — | — | — | — |
| **Ensemble** | — | — | — | — |

> Evaluated on held-out ISIC 2019 test set. "Tuned" F1 uses per-class Nelder-Mead threshold optimization strictly learned on the validation split (with 80/20 overfit protection) and then applied blindly to the test set to prevent data leakage.

---

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Data Setup
1. Download [ISIC 2019 Training Data](https://challenge.isic-archive.com/data/#2019)
2. Run Layer 0 preprocessing to create `layer0_train.csv`, `layer0_val.csv`, `layer0_test.csv`
3. Update paths in `classifier/config.py` and `classifier/dataloader.py`

### Training
```bash
# Run full 4-experiment ablation (effnet_only → swin_only → image_only → full)
python classifier/run_ablation.py

# Or train a single mode
python classifier/train_classifier.py --mode full
```

### Post-Training Optimization
```bash
# After all 4 checkpoints are saved:
python classifier/post_training.py
```

### Evaluation
```bash
python classifier/train_classifier.py --eval-only --checkpoint output/checkpoints/best_full.pt
```

---

## Preprocessing Pipeline

To clean, normalize, and standardize the dataset before model training, all images go through a rigid 4-step preprocessing pipeline. Detailed structural similarity (SSIM) reports tracking image quality through these stages are available in `preprocessing/SSIM for ISIC 2019/`.

1. **DullRazor Hair Removal**: Uses a multi-directional morphological approach with four directional line-shaped kernels (0°, 45°, 90°, 135°) to detect hairs regardless of orientation. It applies Black-Hat filtering followed by OpenCV's `INPAINT_TELEA`. 
    - `kernel_length=17`: Chosen to detect thicker hairs. If larger, it may pick up lesion edges; if smaller, it misses thick hairs.
    - `Threshold=10`: Highly selective. A higher threshold may miss faint hairs, while a lower threshold detects too much noise.
    - `Dilation kernel=(3,3), iter=1`: Removes shadow without destroying real skin texture (larger kernels blur texture, smaller kernels leave ghost shadows).
    - `inpaintRadius=5`: Provides a smooth fill. A larger radius blurs lesion borders, while a smaller radius creates streaky fills for thick hairs.
2. **Shades of Gray Color Normalization**: Corrects for varying dermoscope lighting and color temperatures.
    - `power=4`: Studies on ISIC datasets (e.g., Celebi et al., 2015) suggest `power=6` is optimal for overall classification. However, aggressive color normalization actively reduces melanoma detection sensitivity because it flattens the internal color variation that distinguishes melanoma from benign nevi. `power=4` is a deliberate, conservative choice to correct device-level bias while preserving critical lesion-level color variation.
3. **CLAHE Contrast Enhancement**: 
    - **LAB Color Space**: CLAHE is never applied directly to BGR/RGB channels. Enhancing R, G, and B independently creates severe color shifts—stretching the flat, textureless inpainted hair blobs into unnatural neon colors. By converting to LAB (Lightness, A-color, B-color) and applying CLAHE *only* to the Lightness (L) channel, brightness and contrast are enhanced while natural skin and lesion colors are perfectly preserved.
    - `clipLimit=2.0`: The default OpenCV limit of 4.0 is too aggressive post-DullRazor. Inpainting creates smooth, flat regions where hairs used to be. A high clipLimit stretches these flat areas aggressively, creating visible rectangular tile artifacts outlining the removed hairs. A limit of 2.0 allows meaningful contrast enhancement on real lesion structures while leaving flat inpainted areas unaffected.
4. **Vignette / Border Removal**: Removes the dark circular dermoscope lens artifacts.
    - **Crop Percentage**: 6% per edge.
    - **Image Dimensions**: Reduces image sizes slightly (e.g., from 1022×767 to 900×675 pixels).
    - **Vignette Border**: The visible blue-grey edge is completely removed.
    - **Lesion Content**: Kept full (no lesion is cropped out). This prevents the model from associating hardware-induced vignettes with specific diagnoses.

---

## Dataset

**ISIC 2019** — 8 dermoscopy disease classes:

| Class | Real Samples | Synthetic Samples | Total Train | Description |
|-------|--------------|-------------------|-------------|-------------|
| NV | 10,304 | 0 | 10,304 | Melanocytic nevus |
| MEL | 3,640 | 0 | 3,640 | Melanoma |
| BCC | 2,668 | 0 | 2,668 | Basal cell carcinoma |
| BKL | 2,126 | 0 | 2,126 | Benign keratosis |
| AK | 694 | 806 | 1,500 | Actinic keratosis |
| SCC | 505 | 800 | 1,305 | Squamous cell carcinoma |
| VASC | 205 | 410 | 615 | Vascular lesion |
| DF | 196 | 392 | 588 | Dermatofibroma |

> **Class imbalance:** NV has 17.5× more samples than DF. This is addressed through WeightedRandomSampler, CutMix, and post-training threshold tuning.

---

## Synthetic Data Pipeline

For rare classes (DF, VASC, AK, SCC), we generate high-fidelity synthetic dermoscopy images using:

1. **StyleGAN2-ADA** — class-conditional GAN training on per-class ISIC subsets (512×512, ψ=0.85)
2. **Post-processing** — histogram matching + Gaussian blur to match real ISIC distribution
3. **Quality gate** — FID, SSIM, and perceptual metrics before inclusion

### Generation Quality (FID / SSIM)

| Class | Real Samples | Synthetic | FID ↓ | SSIM (mean) | Pass Rate |
|-------|-------------|-----------|-------|-------------|-----------|
| AK | 694 | 806 | **36.30** | 0.31 | 94.5% |
| SCC | 505 | 800 | **47.51** | 0.30 | 93.9% |
| DF | 196 | 392 | 89.17 | 0.37 | 88.3% |
| VASC | 205 | 410 | 95.32 | 0.36 | 91.2% |

> FID computed against the real training distribution. Lower FID = closer to real. DF and VASC have higher FID due to extremely small real reference sets (196 and 205 images respectively).
> SSIM computed as mean pairwise similarity between 100 randomly sampled synthetic and real image pairs per class. Low values reflect structural diversity within classes rather than generation failure.

---

## Hardware Requirements

- **GPU:** NVIDIA RTX A6000 (48GB) recommended
- **VRAM (training):** ~20 GB minimum (AMP enabled, batch size 16, 384×384)
- **VRAM (inference):** ~5 GB (suitable for RTX 3080/4070 and above)
- **Storage:** ~50 GB for dataset + checkpoints
- **Training time:** ~20 min/epoch for `image_only` mode, ~11 min/epoch for single-backbone modes

---

## Citation

If you use this code, please cite:

```bibtex
@misc{lesioniq2026,
  title={LesionIQ: Hybrid Deep Learning for Dermatoscopy Classification with Synthetic Data Augmentation},
  year={2026},
  url={https://github.com/Astroshreyas1/LesionIQ}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Limitations

- **Class imbalance ceiling:** Macro-F1 is structurally constrained by ISIC 2019 — DF and VASC have <600 training samples each, making per-class F1 fragile
- **External validation:** Not yet validated on external dermoscopy datasets (ISIC 2020, HAM10000, PH²)
- **Synthetic data:** StyleGAN2-ADA augmentation has not been independently validated for clinical utility
- **Decision support only:** This is a clinical decision-support tool — **not a diagnostic replacement.** All predictions require review by a qualified dermatologist
- **SLM Hallucination Risk:** Using a Small Language Model to translate feature attributions (Grad-CAM, SHAP) into clinical text carries inherent risk. If a heatmap highlights an artifact, the SLM may hallucinate a plausible but clinically false rationale. Explainability outputs must always be audited alongside the source images.
- **Privacy-preserving design:** Processes de-identified images only. No PII stored or transmitted. Suitable for HIPAA-aligned research workflows — formal compliance requires institutional review.

---

## Acknowledgments

- [ISIC Archive](https://www.isic-archive.com/) for the dermoscopy dataset
- [timm](https://github.com/huggingface/pytorch-image-models) for pretrained backbone models
- [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) for synthetic generation
