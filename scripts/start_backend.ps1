# ============================================================
# LesionIQ — Quick Start (Backend + ngrok)
# ============================================================
#
# Double-click this file or run from PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\start_backend.ps1
#
# This starts the FastAPI backend AND the ngrok tunnel in one go.
# The frontend is hosted on Vercel and doesn't need to be started.
# ============================================================

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

Write-Host ""
Write-Host "══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  LesionIQ — Starting backend + tunnel"                    -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Activate venv if it exists
$venvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "[1/3] Activating virtual environment..." -ForegroundColor Yellow
    & $venvActivate
} else {
    Write-Host "[1/3] No venv found — using system Python" -ForegroundColor Yellow
}

# Start uvicorn in the background
Write-Host "[2/3] Starting FastAPI backend on port 8000..." -ForegroundColor Yellow
$backendJob = Start-Process -PassThru -NoNewWindow powershell -ArgumentList "-Command", "python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000"

# Give the server a moment to boot
Start-Sleep -Seconds 2

# Start ngrok
Write-Host "[3/3] Starting ngrok tunnel..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  Your ngrok URL will appear below." -ForegroundColor Green
Write-Host "  Frontend (Vercel) is already live — no action needed." -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop everything." -ForegroundColor Gray
Write-Host ""

try {
    ngrok http 8000
} finally {
    # When ngrok is stopped (Ctrl+C), also kill the backend
    if ($backendJob -and !$backendJob.HasExited) {
        Stop-Process -Id $backendJob.Id -Force -ErrorAction SilentlyContinue
        Write-Host ""
        Write-Host "  Backend stopped." -ForegroundColor Yellow
    }
    Write-Host "  Done. Goodbye!" -ForegroundColor Cyan
}
