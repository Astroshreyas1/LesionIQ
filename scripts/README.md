# Scripts

Utility scripts that are safe to keep under version control.

| Script | Description |
|--------|-------------|
| `start_backend.ps1` | **Quick start** — launches the FastAPI backend + ngrok tunnel in one command |
| `download_checkpoints.py` | Download model checkpoints from GitHub Releases into `backend/checkpoints/` |
| `set_github_topics.sh` | Set repository topics via the GitHub API for search discoverability |

## Quick Start

To start the backend and expose it to the Vercel-hosted frontend:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_backend.ps1
```

This will:
1. Activate your virtual environment
2. Start the FastAPI backend on port 8000
3. Start the ngrok tunnel (your public URL appears in the terminal)

Press `Ctrl+C` to stop everything.

Generated outputs, caches, local environment files, and large model artifacts should stay out of this directory and are ignored by the repository-level `.gitignore`.
