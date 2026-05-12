#!/usr/bin/env bash
# ============================================================
# LesionIQ — Create structured git history
# ============================================================
#
# This script stages all pending changes from the monorepo
# restructure and creates meaningful, well-labeled commits.
#
# Run from the repository root:
#   bash scripts/restructure_commits.sh
#
# After running, push with:
#   git push origin main
# ============================================================

set -euo pipefail

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  LesionIQ — Structuring git history"
echo "══════════════════════════════════════════════════════════"
echo ""

# --- Commit 1: Backend/frontend monorepo restructure ---
echo "[1/5] Staging monorepo restructure..."
git add backend/ frontend/ docker-compose.yml .dockerignore
git add -u  # stages all deleted old flat-layout files
git commit -m "refactor: restructure into backend/frontend monorepo

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
prefix (e.g. python backend/inference.py instead of python inference.py)."

echo ""

# --- Commit 2: CI/CD ---
echo "[2/5] Staging CI workflow..."
git add .github/
git commit -m "ci: add GitHub Actions workflow for backend install + frontend build

- Backend job: Python 3.10 + 3.11 matrix, CPU PyTorch, pip install,
  import smoke test (models, preprocessing, inference)
- Frontend job: Node 20, npm ci, tsc + vite build
- Lint job: eslint on frontend source"

echo ""

# --- Commit 3: Scripts ---
echo "[3/5] Staging utility scripts..."
git add scripts/
git commit -m "feat: add checkpoint download script and GitHub topics setup

- scripts/download_checkpoints.py: Downloads model weights from
  GitHub Releases with SHA-256 verification
- scripts/set_github_topics.sh: Sets repository topics via GitHub API
  for search discoverability (deep-learning, medical-imaging, etc.)
- scripts/README.md: Documents available scripts"

echo ""

# --- Commit 4: Docs ---
echo "[4/5] Staging documentation..."
git add docs/
git commit -m "docs: add frontend handoff context document

- docs/CONTEXT_HANDOFF.md: Architecture notes, upload state rules,
  visual system palette, key file references for frontend development"

echo ""

# --- Commit 5: README + gitignore ---
echo "[5/5] Staging README and config updates..."
git add README.md .gitignore
git commit -m "docs: comprehensive README update for monorepo structure

- Add CI badge, Python/PyTorch/License shields, and topic tags
- Update repository layout tree with all new files
- Expand backend package map (Dockerfile, __init__.py, mel_safety_threshold.npy)
- Document importable run_inference_pipeline() API
- Document SLM validation/fallback pipeline and feature hints
- Add frontend environment variables table (VITE_LESIONIQ_ENABLE_DEMO_CASES)
- Expand frontend project structure with all component files
- Replace Windows-only paths with cross-platform instructions
- Add checkpoint download instructions via scripts/download_checkpoints.py
- Add GitHub Releases link for manual checkpoint download"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Done! New commits created."
echo ""
echo "  Review with:  git log --oneline -10"
echo "  Push with:    git push origin main"
echo "══════════════════════════════════════════════════════════"
echo ""
