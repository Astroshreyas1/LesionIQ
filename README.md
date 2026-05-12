# LesionIQ — Hybrid Dermatoscopy Classifier with Synthetic Data Augmentation

[![CI](https://github.com/Astroshreyas1/LesionIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/Astroshreyas1/LesionIQ/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

`deep-learning` · `medical-imaging` · `dermoscopy` · `skin-cancer` · `skin-lesion-classification` · `pytorch` · `efficientnet` · `swin-transformer` · `explainability` · `grad-cam` · `stylegan2` · `isic` · `clinical-ai` · `dermatology` · `synthetic-data`

A research-grade skin lesion classification pipeline combining dual-backbone deep learning (EfficientNet-B4 + Swin Transformer V2) with clinical metadata fusion and synthetic data augmentation via StyleGAN2-ADA. Built for the [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/) 8-class dermoscopy classification task.

---

## Repository Layout

LesionIQ is organized as a local backend/frontend monorepo:

```text
LesionIQ/
├── backend/
│   ├── __init__.py             # Backend Python package marker
│   ├── api.py                  # FastAPI bridge used by the frontend
│   ├── inference.py            # Convenience wrapper for classifier/inference.py
│   ├── Dockerfile              # Backend container image (Python 3.11-slim + uvicorn)
│   ├── classifier/             # Model, training, inference, explainability
│   ├── preprocessing/          # DullRazor, color normalization, CLAHE, border removal
│   ├── synthetic/              # StyleGAN2-ADA utilities and quality checks
│   ├── data/                   # Dataset preparation utilities
│   ├── checkpoints/            # Calibration files; large .pt/.pth weights are git-ignored
│   └── requirements.txt
├── frontend/
│   ├── src/                    # React + TypeScript clinical review UI
│   ├── public/
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.ts
│   ├── tsconfig*.json
│   ├── tailwind.config.ts
│   ├── postcss.config.js
│   ├── eslint.config.js
│   └── .env.example
├── docs/
│   ├── fine_tuning_log.txt     # Complete hyperparameter changelog
│   └── CONTEXT_HANDOFF.md      # Frontend dev handoff context
├── scripts/                    # Utility scripts (version-controlled)
├── output/                     # Generated inference bundles (git-ignored)
├── docker-compose.yml
├── .dockerignore
├── README.md
├── LICENSE
└── .gitignore
```

Generated folders such as `node_modules/`, `frontend/dist/`, `output/`, `backend/output/`, caches, and local `.env` files are ignored. Large checkpoint weights (`backend/checkpoints/*.pt`, `*.pth`) should be stored outside git or supplied separately.

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

## Backend Package Map

```
LesionIQ/backend/
├── __init__.py              # Backend Python package marker
├── api.py                   # FastAPI bridge: POST /cases/analyze → CaseRecord
├── inference.py             # Convenience wrapper re-exporting classifier/inference.py
├── Dockerfile               # Python 3.11-slim container for uvicorn serving
│
├── classifier/              # Core classification pipeline
│   ├── config.py            # Hyperparameters and paths
│   ├── models.py            # LesionIQHybrid architecture
│   ├── dataloader.py        # Dataset, augmentations, weighted sampling
│   ├── train.py             # Training loop (CutMix, SWA, 4-way TTA, progressive unfreezing)
│   ├── evaluate.py          # Test-set evaluation suite
│   ├── explainability.py    # Grad-CAM++, SHAP, attention maps, calibration
│   ├── post_training.py     # Threshold tuning, ensembling, temperature scaling
│   ├── inference.py         # Standalone inference CLI + importable run_inference_pipeline()
│   ├── boost_f1.py          # Differential Evolution threshold optimizer
│   ├── boost_f1_v2.py       # Clinical-aware DiffEvo + MEL safety + confusion analysis
│   ├── boost_f1_v3.py       # Confusion matrix reframe: BCC suppress + SCC boost + MEL safety
│   ├── fix_swa.py           # SWA BN recovery for metadata-mode models (one-shot utility)
│   ├── run_ablation.py      # Automated 4-experiment ablation runner
│   └── train_classifier.py  # CLI entry point
│
├── preprocessing/           # Image preprocessing & cleanup
│   ├── __init__.py                  # Public API: run_pipeline() for inference
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
│   ├── layer0_train.csv     # Training manifest (22,746 images + metadata)
│   ├── layer0_val.csv       # Validation manifest (4,993 images)
│   ├── layer0_test.csv      # Test manifest (8,238 images)
│   ├── training.py          # LesionIQ dataset class (standalone)
│   ├── jitter_metadata.py   # Gaussian jitter for metadata augmentation
│   └── update_lesioniq_metadata.py  # Metadata CSV processing
│
├── checkpoints/             # Trained model weights (.pt files)
│   ├── best_full.pt         # Full hybrid model (F1=0.5924)
│   ├── best_image_only.pt   # Dual-backbone, no metadata (F1=0.5646)
│   ├── best_effnet_only.pt  # EfficientNet-B4 only (F1=0.5392)
│   ├── best_full_swa.pt     # SWA-averaged full model
│   ├── optimal_scales.npy   # Clinical DiffEvo threshold scales
│   ├── optimal_temperature.npy  # Calibrated temperature
│   └── mel_safety_threshold.npy # MEL recall safety threshold (0.265)
│
├── requirements.txt
└── .gitignore
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

### Clinical Explainability
- **EfficientNet Branch**: Grad-CAM++ heatmaps for CNN feature visualization.
- **SwinV2 Branch**: Attention rollout visualisations for transformer patch attribution.
- **Metadata Branch (MLP)**: Perturbation-based feature attribution for tabular feature importance.
- **Temperature Scaling**: Logits are divided by calibrated temperature (T=0.75) before softmax for improved confidence calibration.
- **Final Output Generation**: Visual explainability artifacts (Grad-CAM++ overlays, Swin attention maps) and structured feature data (SHAP values, confidence scores, metadata) are packaged into a diagnostic bundle (`diagnosis.json` + images) and fed into Gemma 3 4B-IT (served locally via Ollama) to generate image-aware, clinically grounded explanations. Deterministic post-validation catches hallucinated claims against source evidence.
- **SLM Validation & Fallback**: `validate_and_repair_slm_output()` enforces a strict schema check (required headings, forbidden bracket tokens, unsafe directive phrases). If the SLM response fails validation, a deterministic `_fallback_slm_report()` is generated from the evidence packet, guaranteeing a clinically grounded report is always available.
- **Feature Hints Pipeline**: `build_feature_hints()` converts raw Grad-CAM++ and attention summaries into conservative clinical hints (primary feature, evidence alignment, uncertainty flags, clinical context) which are embedded in the SLM evidence packet.

### Frontend UI
- A React + TypeScript + Vite clinical review frontend is included in `frontend/src/`. It connects to the live FastAPI inference bridge through a single upload endpoint and consumes the typed `CaseRecord` response across all tabs.
- The app is a decision-support workstation for 8-class dermoscopy classification review, calibrated confidence, threshold-aware differential diagnosis, preprocessing provenance, explainability audit, SLM explanations, and dermatologist feedback.
- It is explicitly **not** a diagnostic replacement. Every screen keeps dermatologist review and clinical verification visible.
- **Demo cases** are behind an environment flag (`VITE_LESIONIQ_ENABLE_DEMO_CASES=true`). When disabled (the default), only live-uploaded cases appear in the case selector.

---

## 📈 Results

### Ablation Study

| Mode | Val F1 (macro) | Val AUC | Val Acc | Notes |
|------|----------------|---------|---------|-------|
| `effnet_only` | 0.5392 | — | — | EfficientNet-B4 backbone only |
| `image_only` | 0.5646 | — | — | EfficientNet-B4 + Swin-Base (no metadata) |
| `full` | 0.5924 | 0.9330 | 0.7503 | Both backbones + metadata MLP |
| `full` + DiffEvo | 0.6066 | 0.9330 | 0.7613 | + Differential Evolution thresholds |
| **3-Model Ensemble + Clinical DiffEvo** | **0.6165** | **0.9404** | **0.7571** | Best overall configuration |

### Per-Class Breakdown (Ensemble + Clinical DiffEvo)

| Class | Precision | Recall | F1 | Support | Clinical Notes |
|-------|-----------|--------|------|---------|----------------|
| MEL | 0.6385 | 0.6349 | 0.6367 | 882 | 21% misclassified as NV |
| NV | 0.8973 | 0.8670 | 0.8819 | 2571 | Dominant class, well-separated |
| BCC | 0.6597 | 0.8198 | 0.7311 | 655 | High recall, absorbs SCC misses |
| AK | 0.4459 | 0.3815 | 0.4112 | 173 | 20% confused with BKL |
| BKL | 0.6206 | 0.5683 | 0.5933 | 498 | — |
| DF | 0.4667 | 0.4884 | 0.4773 | 43 | Small sample, volatile |
| VASC | 0.9444 | 0.7083 | 0.8095 | 48 | Highest precision |
| SCC | 0.3759 | 0.4065 | 0.3906 | 123 | **31% confused with BCC** |

### Confusion Matrix Insights

The dominant misclassification patterns are **not** AK↔SCC as initially hypothesized, but rather:
- **SCC → BCC (31%)**: The largest SCC failure mode. Both are keratinocyte-origin lesions with overlapping morphology.
- **MEL → NV (22%)**: Melanocytic lesion confusion — the classic dermoscopy challenge.
- **AK → BKL (20%)**: AK misclassified as benign keratosis.
- **AK ↔ SCC (10-14%)**: Bidirectional confusion, but secondary to BCC/BKL confusion.

### Post-Training Optimization

- **Threshold tuning**: Clinical-aware Differential Evolution (asymmetric bounds for AK/SCC [1.5, 6.0]) boosts Macro-F1 from 0.5938 → **0.6165** (+0.0227).
- **MEL recall safety**: Forcing MEL recall ≥ 85% costs -0.04 macro-F1 (precision drops to 0.40). Discarded in favor of maintaining overall F1 > 0.60.
- **Temperature scaling**: Optimal T=0.75 (NLL 0.7761 → 0.7432). Improves confidence calibration.
- **Ensemble**: 3-model average (effnet_only + image_only + full) boosts AUC from 0.9330 → 0.9404.

> Evaluated on held-out ISIC 2019 validation set (4,993 images). The clinical-aware optimizer uses weighted per-class F1 (MEL=2.0, SCC=2.5, NV=0.5) to prioritize malignancy detection.

---

## Quick Start

### Backend Quick Start
```bash
pip install -r backend/requirements.txt
```

### Frontend Quick Start
```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL printed in the terminal, usually `http://127.0.0.1:5173/`.

The frontend expects the live backend by default:
```bash
VITE_LESIONIQ_API_BASE_URL=http://localhost:8000
```

Run the API and local SLM stack:
```bash
docker compose up --build
```

The compose stack exposes:
```text
backend API: http://localhost:8000
Ollama:      http://localhost:11434
model:       gemma3:4b-it-qat
```

Build the production bundle:
```bash
cd frontend
npm run build
```

#### Frontend Project Structure
```text
frontend/src/
  app/
    App.tsx            # Shared workflow state, case routing, screen composition
    navigation.ts      # Screen IDs and sidebar nav items
  components/
    layout/            # AppShell, SidebarNav, TopBar, CaseSelector, ThemeToggle
    primitives/        # Card, StatusBadge, SectionTabs, MetricCard, PageHeader, DisclaimerBanner, EmptyState
    domain/            # UploadInferenceCard, ImageViewerCard, PredictionList, AttributionBars,
                       # AuditChecklist, AuditNoteList, MetadataCard, DermoscopyMock, OverlayToggle,
                       # PipelineStepCard, ActionPanel, HistoryList, ComparePanel
  screens/             # CaseReview, Explainability, Preprocessing, History, Compare, Settings
  data/
    cases.ts           # Demo seed CaseRecords (behind VITE_LESIONIQ_ENABLE_DEMO_CASES flag)
    classes.ts         # 8-class label map and canonical class order
    system.ts          # System status and review action definitions
  types/
    lesioniq.ts        # CaseRecord, PredictionScore, ExplainabilityBundle, InferenceBundle,
                       # UploadMetadataInput, and all supporting type definitions
  lib/
    format.ts          # Formatting utilities
    lesioniqApi.ts     # Backend API adapter (runLesionIQAnalysis, resolveLesionIQArtifactUrl)
  hooks/
    useTheme.ts        # Dark/light theme hook
  styles/
    index.css          # Tailwind entrypoint and CSS custom properties
  main.tsx             # React DOM entry point
```

#### Case Data Organization
Seed records live in `frontend/src/data/cases.ts` and implement the same typed contract returned by the backend in `frontend/src/types/lesioniq.ts`.
Each `CaseRecord` contains case identifiers, masked patient metadata, calibrated prediction scores for all 8 ISIC classes, per-class threshold margins, Grad-CAM / attention / metadata attribution summaries, SLM explanation text, audit checks, preprocessing steps, history entries, and compare entries.

The exact class set is preserved:
`MEL`, `NV`, `BCC`, `BKL`, `AK`, `SCC`, `VASC`, and `DF`.

#### Frontend Behavior
- The case selector in the top bar switches between multiple realistic de-identified mock cases.
- Selected case context is carried across Case Review, Explainability, Preprocessing, History, Compare, and Settings.
- Case Review includes a dermoscopy viewer with Raw, Grad-CAM, Attention, and Metadata overlay modes, threshold-aware prediction summary, all 8 ranked classes, metadata fusion context, SHAP-like attribution, and clinician actions.
- Case Review includes a metadata-gated upload workflow. Uploaded images do not run inference until age, sex, and anatomical site are complete. Those values mirror the hybrid metadata branch described above.
- Uploaded images plus complete metadata call the live backend API adapter in `frontend/src/lib/lesioniqApi.ts`. The response must be a `CaseRecord` with preprocessing artifacts, calibrated classifier output, Grad-CAM, attention, metadata attribution, `diagnosis.json`, and SLM explanation status.
- Explainability includes tabs for Grad-CAM, Attention, Metadata Attribution, and SLM Summary, plus audit checks and feedback affordances.
- Preprocessing exposes the exact 4-step LesionIQ pipeline: DullRazor hair removal, Shades-of-Gray color normalization, CLAHE in LAB color space, and vignette border removal.
- History supports This Patient Only, All Cases, and High Concern filters.
- Compare supports Current vs Previous, Raw vs Grad-CAM, and Raw vs Preprocessed review modes.
- Settings shows calibration, threshold tuning, explainability status, model mode, theme controls, backend integration notes, and legal caveats.

#### Frontend Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_LESIONIQ_API_BASE_URL` | `http://localhost:8000` | Backend API base URL |
| `VITE_LESIONIQ_ENABLE_DEMO_CASES` | unset (false) | Set to `true` to show demo seed cases in the case selector |

#### Backend Integration Contract
The frontend uses a single live analysis adapter:
`frontend/src/lib/lesioniqApi.ts`.

It exports two functions:
- `runLesionIQAnalysis({ image, previewUrl, metadata })` — sends the multipart analysis request and returns a typed `CaseRecord`.
- `resolveLesionIQArtifactUrl(url, outputDirectory)` — resolves relative artifact paths from the backend bundle into absolute URLs for image rendering.

Configure the backend URL with:
```bash
VITE_LESIONIQ_API_BASE_URL=http://localhost:8000
```

When configured, the frontend sends one multipart request:
```http
POST /cases/analyze
Content-Type: multipart/form-data
```

Request fields:
```text
image           Dermoscopy image file
metadata        JSON string: { ageYears, sex, anatomicalSite, modelMode }
slm_container   lesioniq_ollama
slm_model       gemma3:4b-it-qat
```

The backend should run the complete pipeline:
```text
image + metadata
  -> preprocessing layers
  -> final preprocessed image
  -> classifier inference
  -> Grad-CAM++ and graded attention weights
  -> diagnosis.json
  -> local SLM call through Docker/Ollama
  -> CaseRecord response
```

The local SLM is served by Docker Compose:
```text
container: lesioniq_ollama
model:     gemma3:4b-it-qat
```

The frontend expects the backend response to match `CaseRecord` from `frontend/src/types/lesioniq.ts`. This keeps the UI modular: Case Review, Preprocessing, Explainability, History, Compare, and Settings all continue reading the same typed record from the shared live analysis response.

Important response fields:
- `predictionScores`: all 8 class probabilities, thresholds, and margins.
- `metadata`: age, sex, anatomical site, Fitzpatrick type when available, and completeness.
- `preprocessingSteps`: raw, DullRazor, color normalization, CLAHE, and border removal provenance.
- `explainability`: Grad-CAM summary, attention summary, metadata attribution, SLM summary, audit checks, and notes.
- `inferenceBundle`: artifact names or URLs for raw image, preprocessing layers, final preprocessed image, Grad-CAM, attention weights, and `diagnosis.json`.

The backend serves artifact URLs from `/artifacts/{case_id}/`:
```text
backend/output/inference/<case_id>/
  raw.png
  01_dullrazor.png
  02_shades_of_grey.png
  03_clahe.png
  04_border_removed.png
  final_preprocessed.png
  gradcam.png
  attention.png
  diagnosis.json
```

#### Prototype Assumptions and Caveats
- Current dermoscopy images are stylized local UI previews, not patient images.
- Demo seed data is de-identified, contains no PII, and is gated behind `VITE_LESIONIQ_ENABLE_DEMO_CASES=true`.
- Uploaded image analysis uses the live FastAPI backend. No database or clinical report export service is connected yet.
- The UI is suitable for hackathon/product demonstration and backend contract iteration, not clinical deployment.
- `frontend/src/lib/lesioniqApi.ts` includes `resolveLesionIQArtifactUrl()` to handle both absolute and relative artifact URLs from the backend bundle.

### Data Setup
1. Download [ISIC 2019 Training Data](https://challenge.isic-archive.com/data/#2019)
2. Run Layer 0 preprocessing to create `layer0_train.csv`, `layer0_val.csv`, `layer0_test.csv`
3. Update paths in `backend/classifier/config.py` and `backend/classifier/dataloader.py`

### Training
```bash
# Run full 4-experiment ablation (effnet_only → swin_only → image_only → full)
python backend/classifier/run_ablation.py

# Or train a single mode
python backend/classifier/train_classifier.py --mode full
```

### SWA Recovery
If SWA batch norm update fails (e.g., metadata shape mismatch), re-run without retraining:
```bash
python backend/classifier/train_classifier.py --fix-swa --checkpoint backend/checkpoints/best_full.pt
```

> **Note**: PyTorch's built-in `update_bn` discards metadata inputs, causing a shape mismatch in `full` mode (2816 vs 2848 features). The custom BN update loop in `train.py` passes both images and metadata correctly.

### Post-Training Optimization
```bash
# Works with 2+ checkpoints (skips missing ones gracefully):
python backend/classifier/post_training.py
```

### Evaluation
```bash
python backend/classifier/train_classifier.py --eval-only --checkpoint backend/output/checkpoints/best_full.pt
```

---

## Preprocessing Pipeline

To clean, normalize, and standardize the dataset before model training, all images go through a rigid 4-step preprocessing pipeline. Detailed structural similarity (SSIM) reports tracking image quality through these stages are available in `backend/preprocessing/SSIM for ISIC 2019/`.

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
    - **Vignette Border**: The visible lens edge is completely removed.
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

| | Training | Inference / Post-Training |
|---|---|---|
| **GPU** | RTX A6000 (48 GB) or RTX 5070 Ti (16 GB) | Any GPU with ≥ 8 GB VRAM |
| **VRAM** | ~20 GB (AMP, batch 16, 384×384) | ~5 GB |
| **Batch size** | 16 (A6000) / 8 (5070 Ti, reduce in config.py) | 16 |
| **Storage** | ~50 GB (dataset + checkpoints) | ~2.5 GB (checkpoints only) |
| **Epoch time** | ~20 min (`image_only`), ~11 min (single backbone) | — |

> **5070 Ti note:** 16 GB VRAM is sufficient for training with `BATCH_SIZE=8` and `GRAD_ACCUM_STEPS=6` (effective batch = 48). For inference and post-training optimization, batch size 16 works fine.

---

## Machine Transfer

To move the pipeline to a new machine:

### 1. Clone and install
```bash
git clone https://github.com/Astroshreyas1/LesionIQ.git
cd LesionIQ

python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r backend/requirements.txt
```

### 2. Download checkpoints
```bash
python scripts/download_checkpoints.py
```

Or download manually from the [GitHub Releases](https://github.com/Astroshreyas1/LesionIQ/releases/tag/v1.0-checkpoints) page and place files in `backend/checkpoints/`.

### 3. Update data paths
Edit `backend/classifier/dataloader.py` lines 20–22:
```python
TRAIN_CSV = "path/to/data/layer0_train.csv"
VAL_CSV   = "path/to/data/layer0_val.csv"
TEST_CSV  = "path/to/data/layer0_test.csv"
```

Then fix `image_path` column in each CSV to point to the actual image directory on the new machine:
```python
import pandas as pd
for csv_file in ['layer0_train.csv', 'layer0_val.csv', 'layer0_test.csv']:
    df = pd.read_csv(csv_file)
    df['image_path'] = df['image_path'].str.replace(
        '<old_image_dir>',
        '<your_image_dir>', regex=False
    )
    df.to_csv(csv_file, index=False)
```

### 4. For 5070 Ti (16 GB VRAM)
In `backend/classifier/config.py`:
```python
BATCH_SIZE = 8           # halved from 16
GRAD_ACCUM_STEPS = 6     # doubled from 3 (effective batch stays 48)
NUM_WORKERS = 4          # adjust to CPU core count
```

---

## Inference

A standalone 5-stage pipeline that classifies dermoscopy images and produces explainability artifacts for SLM-based clinical report generation. No training dependencies required.

```
Input Image + Metadata → Preprocessing → Classifier (4-way TTA) → Explainability → SLM Output Bundle
```

### Importable API

The inference pipeline is also available as an importable Python function for backend integration:

```python
from classifier.inference import run_inference_pipeline

result = run_inference_pipeline(
    image_path="lesion.png",
    age=65,
    sex="male",
    site="head/neck",
    mode="full",
    output_dir="./output/inference",
)
# result["diagnosis"]       → diagnosis.json-compatible dict
# result["artifact_paths"]  → absolute paths to all generated artifacts
# result["output_dir"]      → case bundle directory
```

The FastAPI bridge (`backend/api.py`) calls `run_inference_pipeline()` internally.

Each image produces a diagnostic bundle:
```
backend/output/inference/<case_id>/
├── raw.png
├── 01_dullrazor.png
├── 02_shades_of_grey.png
├── 03_clahe.png
├── 04_border_removed.png
├── final_preprocessed.png  # Model input (384×384)
├── gradcam.png             # Grad-CAM++ heatmap overlay
├── attention.png           # Swin Transformer attention rollout
└── diagnosis.json          # CaseRecord-compatible inference trace
```

### Usage
```bash
# Interactive mode — prompts for age, sex, site (with NA option)
python backend/inference.py --image lesion.png

# Explicit metadata
python backend/inference.py --image lesion.png --age 65 --sex male --site "head/neck"

# NA metadata (uses population defaults: age=50, sex=unknown, site=unknown)
python backend/inference.py --image lesion.png --age NA --sex NA --site NA

# Batch (entire directory, no interactive prompts)
python backend/inference.py --dir path/to/images/ --output-dir ./results

# Lighter model (no Swin, no attention map output)
python backend/inference.py --image lesion.png --mode effnet_only

# Disable post-training optimizations
python backend/inference.py --image lesion.png --no-temperature --no-scales
```

### Interactive Metadata Input

When running on a single image without `--age`/`--sex`/`--site` flags, the pipeline prompts for each field interactively:

```
╔══════════════════════════════════════╗
║        Patient Metadata Input        ║
╚══════════════════════════════════════╝

  Age (years, or NA): 65
  Sex (male / female / NA): male

  Anatomical site:
    1. anterior torso    2. head/neck
    3. lateral torso     4. lower extremity
    5. oral/genital      6. palms/soles
    7. posterior torso   8. upper extremity
    9. NA (unknown)
  Select [1-9]: 2
```

Entering `NA` for any field uses population defaults (age=0.5 normalized, sex=unknown, site=unknown).

### diagnosis.json (SLM input)
```json
{
  "version": "1.0",
  "image": "ISIC_0024306.png",
  "model": {
    "mode": "full",
    "temperature": 0.75,
    "thresholds_applied": true,
    "mel_safety_threshold": 0.265,
    "mel_safety_triggered": true
  },
  "prediction": {
    "class": "MEL",
    "class_full": "Melanoma",
    "confidence": 0.7321,
    "is_malignant": true
  },
  "probabilities": {
    "MEL": 0.7321, "NV": 0.1215, "BCC": 0.0534,
    "AK": 0.0312, "BKL": 0.0841, "DF": 0.0021,
    "VASC": 0.0010, "SCC": 0.0246
  },
  "top3": [
    {"class": "MEL", "full_name": "Melanoma", "probability": 0.7321, "is_malignant": true},
    {"class": "NV",  "full_name": "Melanocytic Nevus", "probability": 0.1215, "is_malignant": false},
    {"class": "BKL", "full_name": "Benign Keratosis", "probability": 0.0841, "is_malignant": false}
  ],
  "clinical_flags": {
    "malignant_total_prob": 0.7892,
    "requires_biopsy": true,
    "low_confidence": false,
    "differential_diagnosis": false
  },
  "explainability": {
    "gradcam": "gradcam.png",
    "attention": "attention.png",
    "shap_metadata": {
      "age_approx": 0.0234,
      "sex_male": 0.0012,
      "site_head/neck": 0.0089
    }
  },
  "metadata_input": {
    "age": 65,
    "sex": "male",
    "site": "head/neck"
  }
}
```

The SLM receives `final_preprocessed.png` + `gradcam.png` + `attention.png` + `diagnosis.json` plus metadata, then generates the clinical narrative.

The evidence packet sent to the SLM is built by `build_slm_payload()`, which includes:
- Prediction with confidence level, threshold, and margin
- Full probability distribution and top-3 classes
- Grad-CAM++ and attention spatial summaries (region, area coverage, peak pixel)
- `feature_hints` — conservative clinical feature hints derived from heatmap analysis (primary feature, evidence alignment, uncertainty flags, clinical context)
- Clinical flags and metadata

The SLM response is validated by `validate_and_repair_slm_output()`, which checks for required report headings, rejects bracket tokens and unsafe clinical directives, and falls back to a deterministic template (`_fallback_slm_report()`) if the response fails validation.

### CLI options
| Flag | Default | Description |
|------|---------|-------------|
| `--image` | — | Path to a single image |
| `--dir` | — | Path to directory of images |
| `--mode` | `full` | `full`, `image_only`, `swin_only`, or `effnet_only` |
| `--checkpoint` | auto | Custom checkpoint path |
| `--output-dir` | `./backend/output/inference` | Where to save diagnostic bundles |
| `--age` | interactive | Patient age (number or `NA`) |
| `--sex` | interactive | `male`, `female`, `unknown`, or `NA` |
| `--site` | interactive | Anatomical site (e.g. `head/neck`) or `NA` |
| `--no-scales` | off | Disable DiffEvo threshold scaling |
| `--no-temperature` | off | Disable temperature scaling |
| `--no-mel-safety` | off | Disable MEL recall safety override |
| `--interactive` | auto | Force interactive metadata prompts |

> **Requirements:** `torch`, `torchvision`, `timm`, `albumentations`, `pillow`, `numpy`, `opencv-python`. VRAM: ~5 GB.

---

## Post-Training Optimization Updates

Three rounds of post-training threshold optimization were performed on the 3-model ensemble (effnet_only + image_only + full) using 4,993 validation images. No model retraining was performed — all improvements come from DiffEvo logit scaling and post-hoc MEL recall constraints.

### v1 — Baseline DiffEvo (`boost_f1.py`)

First pass: symmetric bounds `[0.3, 4.0]` for all 8 classes. Three methods tested:
- **Differential Evolution** (global optimizer with local polish)
- **Greedy per-class grid search** (sequential, 3 passes)
- **5-fold cross-validated thresholds** (unbiased estimate)

| Configuration | Macro-F1 |
|---------------|----------|
| Single model (full) baseline | 0.5924 |
| Single model + DiffEvo | 0.6108 |
| Ensemble baseline (3 models) | 0.5938 |
| **Ensemble + DiffEvo** | **0.6165** |

**Outcome:** DiffEvo dominated. Ensemble + DiffEvo selected as the production baseline.

---

### v2 — Clinical-Aware Optimization (`boost_f1_v2.py`)

Targeted AK/SCC confusion and MEL recall. Key changes from v1:
- **Asymmetric bounds:** AK and SCC boosted to `[1.5, 6.0]` (vs. `[0.3, 4.0]`)
- **Clinical-weighted DiffEvo:** MEL weight=2.0, SCC weight=2.5 (penalizes malignant misses)
- **MEL recall safety:** post-hoc threshold to force MEL recall >= 85%

| Configuration | Macro-F1 | MEL Recall |
|---------------|----------|------------|
| Ensemble baseline | 0.5938 | 0.633 |
| Asymmetric DiffEvo | 0.6165 | 0.635 |
| Clinical-Weighted DiffEvo | 0.6165 | 0.635 |
| + MEL safety (target 85%) | 0.5724 | 0.850 |

**Outcome:** 85% MEL recall target was too aggressive — dropped macro-F1 by 0.044. Asymmetric DiffEvo without MEL safety selected.

---

### v3 — Confusion Matrix Reframe (`boost_f1_v3.py`) *(current)*

Reframed priorities based on actual confusion matrix failure modes:

| Priority | Failure | Miss Rate | Clinical Risk |
|----------|---------|-----------|---------------|
| P1 | MEL to NV | 21.7% | Missed melanoma |
| P2 | SCC to BCC | 31.7% | Missed SCC mismanaged as BCC |
| P3 | AK to BKL | 19.7% | Pre-malignant missed as benign |

Key changes from v2:
- **BCC suppressed** to `[0.3, 3.0]` (was `[0.3, 4.0]`) — BCC was absorbing SCC predictions
- **SCC boosted** to `[2.0, 7.0]` (was `[1.5, 6.0]`) — forces SCC to compete with BCC
- **NV capped** to `[0.3, 2.0]` — prevents NV from absorbing MEL
- **MEL recall target reduced** to 80% (85% was too aggressive)
- **Clinical weights updated:** MEL=2.5, SCC=3.0

#### v3 Results

| Metric | Baseline | v3 DiffEvo | + MEL Safety |
|--------|----------|-----------|--------------|
| **Macro-F1** | 0.5938 | **0.6155** | 0.5868 |
| **SCC recall** | 0.252 | **0.407** (+62%) | 0.350 |
| **MEL recall** | 0.633 | 0.635 | **0.806** |
| MEL to NV miss | 21.7% | 20.9% | **7.7%** |
| SCC to BCC miss | 31.7% | 30.9% | 28.5% |
| AK to BKL miss | 19.7% | **15.6%** | 14.5% |

#### Deployed Configuration

The production pipeline uses **both** v3 DiffEvo scales and MEL safety:

```
optimal_scales.npy          -- v3 Clinical-Weighted DiffEvo scales
mel_safety_threshold.npy    -- MEL raw probability threshold (0.265)
optimal_temperature.npy     -- LBFGS-calibrated temperature (0.75)
```

All three calibration files are stored in `backend/checkpoints/` and are loaded automatically by `_load_runtime()` (cached via `lru_cache`) when the inference pipeline or API is invoked.

At inference, if the raw (pre-scaling) MEL probability >= 0.265, the prediction is overridden to MEL regardless of the scaled argmax. This forces MEL recall to ~80% at the cost of precision (0.64 to 0.44). The tradeoff is clinically appropriate — **flagging a false positive is preferable to missing a melanoma.**

Disable with `--no-mel-safety` when precision matters more (e.g., batch screening).

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
- **SLM Hallucination Risk:** Using a multimodal Small Language Model to translate visual explainability artifacts (Grad-CAM++ overlays, attention maps) and structured feature attributions (SHAP values) into clinical text carries inherent risk. If a heatmap highlights an artifact, the SLM may hallucinate a plausible but clinically false rationale. Constrained decoding, structured I/O schemas, and deterministic post-validation are used to mitigate this, but explainability outputs must always be audited by a qualified dermatologist.
- **Privacy-preserving design:** Processes de-identified images only. No PII stored or transmitted. Suitable for HIPAA-aligned research workflows — formal compliance requires institutional review.

---

## Acknowledgments

- [ISIC Archive](https://www.isic-archive.com/) for the dermoscopy dataset
- [timm](https://github.com/huggingface/pytorch-image-models) for pretrained backbone models
- [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) for synthetic generation
