# LesionIQ — Dev Session Chat Export
**Date:** 2026-06-01  
**Session:** Backend optimization + remote SLM setup  
**Project:** LesionIQ (Hackathon)  

---

## Conversation

---

### User
> 1. update my local repo to the repo's version

### Claude
Fetched origin. Local branch was 6 commits behind `origin/main` with local modifications across ~100 files.

Attempted `git reset --hard origin/main` — failed with `Operation not permitted` from the sandbox (files were read-only from sandbox perspective).

**Instruction given:**
```powershell
cd C:\LesionIQ
git fetch origin
git reset --hard origin/main
```

---

### User
> how do i give you permission?

### Claude
Run in PowerShell **as Administrator**:
```powershell
icacls "C:\LesionIQ" /grant Everyone:F /T
```

---

### User
> done, now do it

### Claude
Attempted reset again — hit a `.git/index.lock` file blocking the operation (leftover from a previous crashed git process).

Tried to remove the lock from the sandbox — still `Operation not permitted`.

**Instruction given:**
```powershell
Remove-Item "C:\LesionIQ\.git\index.lock" -Force
git -C "C:\LesionIQ" reset --hard origin/main
```

---

### User
*(ran the commands — lock file was already gone, reset succeeded)*
```
HEAD is now at 04141d3 fix: move clahe_comparison.png to correct location, remove internal handoff doc
```

### Claude
Confirmed — local repo synced to `origin/main` at `04141d3`.

---

### User
> now i have a few changes i want you to make
> 1. i want to host the backend on this laptop and the SLM on another laptop and connect it via the web (explain this too)
> 2. can you try and make the backend a little faster, initially i was very scared for the consumption of 6 gb vram, check if you can make it go a little faster

### Claude
Read `backend/api.py`, `backend/classifier/inference.py`, and `backend/classifier/models.py` to understand the full pipeline.

**Architecture summary:**
- `LesionIQHybrid` = EfficientNet-B4 + SwinV2-Base + metadata MLP fused at classifier head
- 4-way TTA (original + 3 flips) per inference call
- Grad-CAM++ on EfficientNet branch
- SwinV2 Grad-AM for attention
- 13 sequential SHAP perturbation passes for metadata attribution
- Ollama (`gemma3:4b-it-qat`) called via `httpx` at `OLLAMA_BASE_URL`

**Identified changes:**

| Problem | Root cause | Fix |
|---|---|---|
| 6 GB VRAM | fp32 weights + large SwinV2 activations | Load model in fp16 via `.half()` |
| Slow inference | 4 forward passes per image (TTA) | Reduce to 2 (original + horizontal flip) |
| Slow SHAP | 13 sequential forward passes | Batch all 13 into one pass |
| Minor overhead | Albumentations imported inside hot path | Move to module-level import |
| SLM on same machine | Ollama shares GPU budget | Point `OLLAMA_BASE_URL` to another laptop |

**Made two initial changes to `inference.py`:**
- `build_model()`: added `model.half()` on CUDA (gated by `LESIONIQ_FP16` env var, default on)
- `predict()`: reduced 4-way TTA to 2-way, added `torch.autocast` context

---

### User
> i hit 6 gb with SLM + Inference... whats the difference between fp16 vs fp32

### Claude
**fp32** — 4 bytes per weight. Default PyTorch.  
**fp16** — 2 bytes per weight. Modern GPUs (RTX 2000+) run it faster via Tensor Cores.

The real VRAM hit isn't just weights — it's **activations** (intermediate tensors at each layer). SwinV2-Base at 384px resolution produces enormous attention maps. That's where most of the 6 GB goes.

fp16 halves both weight storage and activation memory (via autocast).

| Component | fp32 VRAM |
|---|---|
| SwinV2-Base (activations at 384px) | ~3.5–4 GB |
| EfficientNet-B4 | ~1–1.5 GB |
| Ollama gemma3:4b-it-qat | ~3–4 GB |

Running both on the same GPU = 7–9 GB total in fp32. After fp16, inference drops to ~2–2.5 GB, leaving headroom for Ollama — but separating them onto two laptops fully eliminates the competition.

---

### User
> set up the backend for now and then try making it slightly faster too after that help us install the SLM on that other laptop

### Claude
Discovered `api.py` was **truncated** (606 lines locally, 642 in git). Restored from `git show HEAD:backend/api.py`.

Then discovered `inference.py` was also **truncated** (1192 lines locally, 1356 in git) — the same truncation bug from before the git reset.

Applied all patches via a Python script on the full git-sourced file to avoid re-truncation:

---

**Changes made to `backend/classifier/inference.py`:**

```python
# 1. Module-level imports (was inside hot path — re-imported every call)
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 2. fp16 model loading in build_model()
if DEVICE == "cuda" and os.getenv("LESIONIQ_FP16", "1") != "0":
    model.half()

# 3. 2-way TTA + autocast in predict()
autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
with autocast_ctx:
    logits = (
        _fwd(image_tensor)
        + _fwd(torch.flip(image_tensor, dims=[3]))  # horizontal flip only
    ) / 2.0

# 4. Batched SHAP — 13 sequential passes → 1 batched pass
batched_meta  = torch.cat(perturbed_metas, dim=0)           # [14, 13]
batched_image = image_t.expand(len(perturbed_metas), -1, -1, -1)  # [14, 3, H, W]
with torch.no_grad():
    logits = model(batched_image, batched_meta)
    probs  = torch.softmax(logits.float(), dim=1).cpu().numpy()
```

---

**Changes made to `docker-compose.yml`:**

```yaml
# OLLAMA_BASE_URL is now read from .env
OLLAMA_BASE_URL: ${OLLAMA_BASE_URL:-http://ollama:11434}
LESIONIQ_FP16: ${LESIONIQ_FP16:-1}

# Ollama service is profile-gated — skipped when using remote SLM
profiles: ["local-slm", ""]
```

---

**New file: `.env`**
```env
# Local SLM (default)
OLLAMA_BASE_URL=http://ollama:11434

# Remote SLM — uncomment and fill in other laptop's IP
# OLLAMA_BASE_URL=http://192.168.x.x:11434

OLLAMA_MODEL=gemma3:4b-it-qat
LESIONIQ_FP16=1
```

---

**New file: `scripts/setup_remote_slm.md`**  
Full step-by-step instructions for installing Ollama on the other laptop, enabling external connections, pulling the model, and switching the backend to point at it.

---

## Summary of all files changed

| File | Change |
|---|---|
| `backend/classifier/inference.py` | fp16 loading, 2-way TTA, autocast, batched SHAP, module-level imports |
| `docker-compose.yml` | Env var passthrough, profile-gating Ollama for remote-SLM mode |
| `.env` | New file — runtime config (gitignored) |
| `scripts/setup_remote_slm.md` | New file — remote SLM setup guide |

## Net performance impact (estimated)

| Stage | Before | After |
|---|---|---|
| Model VRAM | ~5–6 GB | ~2.5–3 GB |
| TTA forward passes | 4× | 2× |
| SHAP forward passes | 13× sequential | 1× batched |
| Albumentations import | Every call | Once at startup |
| SLM VRAM competition | Shared GPU | Separate machine (when remote) |

---

*Exported from active Cowork session — 2026-06-01*
