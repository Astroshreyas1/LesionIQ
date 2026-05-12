# LesionIQ Frontend Handoff Context

## Current Goal
Continue polishing the existing LesionIQ React/Vite frontend without redesigning the approved layout.

## Product Semantics
- LesionIQ is a dermoscopy clinical decision-support UI, not a diagnostic replacement.
- It supports 8-class skin lesion classification: MEL, NV, BCC, BKL, AK, SCC, VASC, DF.
- It uses image + metadata fusion.
- It exposes calibrated confidence, per-class threshold margins, Grad-CAM, Swin attention, SHAP-like metadata attribution, preprocessing provenance, and SLM explanation output.
- Dermatologist review is required for all predictions.

## Current State Architecture
Shared upload/workflow state is lifted into `src/app/App.tsx`.

Important fields:
- `uploadedImage`
- `uploadedPreviewUrl`
- `selectedCase`
- `analysisReady`
- `activeTab`
- `hasUploadedImage`
- `caseStatus`
- `resultsBundle`
- `workflowStage`

The shared state is passed to:
- `CaseReview`
- `Explainability`
- `Preprocessing`

This prevents uploaded image state from being trapped in only one tab.

## Upload State Rules
There are three states:
1. `empty`: no image, no analysis
2. `uploaded`: image selected, analysis not run
3. `analyzed`: result bundle/case ready

Case Review gates result sections with this state:
- Empty: upload intake + placeholder only
- Uploaded pending: image evidence preview + analysis pending card only
- Analyzed: image evidence, ranked differential, prediction summary, threshold support, attribution, actions

## Header
The header was cleaned up.
It no longer owns the uploaded-case strip.
It now shows:
- LesionIQ title
- compact status
- case selector
- theme toggle

Uploaded image/case context lives in the left workflow area after upload.

## Visual System
Canonical dark palette:
- Background: `#0B0B0B`
- Panels: `#1A1A1A`
- Primary text: `#E0E0E0`
- Secondary text: `#A0A0A0`
- Success accent: `#5AC18E`
- Bronze/key data accent: `#96896C`

Do not introduce blue, cyan, purple, bright gold, beige UI, or glossy gradients.

## Key Files
- `src/app/App.tsx`: shared workflow state and routing
- `src/screens/CaseReview.tsx`: empty/uploaded/analyzed gating
- `src/screens/Explainability.tsx`: shared image + analysis gating
- `src/screens/Preprocessing.tsx`: shared image + analysis gating
- `src/components/domain/UploadInferenceCard.tsx`: upload controls and workflow summary
- `src/lib/uploadInference.ts`: mock inference case generation
- `src/styles/index.css`: theme tokens

## Verification
Run:
```bash
npm.cmd run build
```

Current build passes.
