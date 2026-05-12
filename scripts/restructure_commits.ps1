# ============================================================
# LesionIQ — Create structured git history (PowerShell)
# ============================================================
#
# This script stages all pending changes from the monorepo
# restructure and creates meaningful, well-labeled commits.
#
# Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts\restructure_commits.ps1
#
# After running, push with:
#   git push origin main
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "══════════════════════════════════════════════════════════"
Write-Host "  LesionIQ — Structuring git history"
Write-Host "══════════════════════════════════════════════════════════"
Write-Host ""

# --- Commit 1: Backend/frontend monorepo restructure ---
Write-Host "[1/5] Staging monorepo restructure..."
git add backend/ frontend/ docker-compose.yml .dockerignore
git add -u  # stages all deleted old flat-layout files
git commit -m @"
refactor: restructure into backend/frontend monorepo

- Move classifier/, preprocessing/, synthetic/, data/, checkpoints/
  into backend/ with __init__.py package marker
- Add backend/api.py (FastAPI bridge) and backend/inference.py (wrapper)
- Add backend/Dockerfile for containerized serving
- Add frontend/ (React + TypeScript + Vite clinical review UI)
- Add docker-compose.yml for backend + Ollama SLM stack
- Add .dockerignore

The flat-layout files (classifier/, preprocessing/, etc.) are removed
from the repo root and now live under backend/.

BREAKING: All import paths and CLI commands now reference the backend/
prefix (e.g. python backend/inference.py instead of python inference.py).
"@

Write-Host ""

# --- Commit 2: CI/CD ---
Write-Host "[2/5] Staging CI workflow..."
git add .github/
git commit -m @"
ci: add GitHub Actions workflow for backend install + frontend build

- Backend job: Python 3.10 + 3.11 matrix, CPU PyTorch, pip install,
  import smoke test (models, preprocessing, inference)
- Frontend job: Node 20, npm ci, tsc + vite build
- Lint job: eslint on frontend source
"@

Write-Host ""

# --- Commit 3: Scripts ---
Write-Host "[3/5] Staging utility scripts..."
git add scripts/
git commit -m @"
feat: add checkpoint download script and GitHub topics setup

- scripts/download_checkpoints.py: Downloads model weights from
  GitHub Releases with SHA-256 verification
- scripts/set_github_topics.sh: Sets repository topics via GitHub API
  for search discoverability (deep-learning, medical-imaging, etc.)
- scripts/README.md: Documents available scripts
"@

Write-Host ""

# --- Commit 4: Docs ---
Write-Host "[4/5] Staging documentation..."
git add docs/
git commit -m @"
docs: add frontend handoff context document

- docs/CONTEXT_HANDOFF.md: Architecture notes, upload state rules,
  visual system palette, key file references for frontend development
"@

Write-Host ""

# --- Commit 5: README + gitignore ---
Write-Host "[5/5] Staging README and config updates..."
git add README.md .gitignore
git commit -m @"
docs: comprehensive README update for monorepo structure

- Add CI badge, Python/PyTorch/License shields, and topic tags
- Update repository layout tree with all new files
- Expand backend package map (Dockerfile, __init__.py, mel_safety_threshold.npy)
- Document importable run_inference_pipeline() API
- Document SLM validation/fallback pipeline and feature hints
- Add frontend environment variables table (VITE_LESIONIQ_ENABLE_DEMO_CASES)
- Expand frontend project structure with all component files
- Replace Windows-only paths with cross-platform instructions
- Add checkpoint download instructions via scripts/download_checkpoints.py
- Add GitHub Releases link for manual checkpoint download
"@

Write-Host ""
Write-Host "══════════════════════════════════════════════════════════"
Write-Host "  Done! New commits created."
Write-Host ""
Write-Host "  Review with:  git log --oneline -10"
Write-Host "  Push with:    git push origin main"
Write-Host "══════════════════════════════════════════════════════════"
Write-Host ""
