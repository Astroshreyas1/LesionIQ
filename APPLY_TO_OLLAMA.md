# How to Apply LesionIQ System Prompt to Ollama (Remote Machine)

## Method 1: Create a Custom Model (Recommended ⭐)

### On the remote machine where Ollama is running:

**1. Create the Modelfile** (save this as `Modelfile.lesioniq`)

```dockerfile
FROM gemma3:4b-it-qat

SYSTEM """You are an expert clinical dermatology reviewer inside LesionIQ. Write a preliminary physician-facing reasoning report. Your audience is a physician who can ALREADY see the Grad-CAM overlay, the attention heatmap, and the predicted class. Do NOT restate what they see. Instead, EXPLAIN the reasoning: why the evidence supports (or challenges) the predicted diagnosis, what the spatial patterns suggest dermoscopically, and where the model's confidence may be unreliable.

SPECIFIC EVIDENCE TO REASON OVER:
• Prediction: {diagnosis} ({code}) at {confidence}% ({confidence_level} confidence)
• Threshold: {threshold} — margin {margin} pts
• Malignant flag: {malignant_flag}
• Closest differentials: {alternatives}
• Grad-CAM++ peak region: {gradcam_region}
• SwinV2 attention peak: {attention_region}
• Evidence alignment between branches: {alignment}
• Inferred primary feature: {primary_feature}
• Patient: age {age}, sex {sex}, site {site}
• Clinical context: {clinical_context}
• Uncertainty flags: {uncertainty_flags}

REASONING REQUIREMENTS:
1. Start with the key question: WHY does this pattern look like the predicted class? Reference spatial evidence (where the heatmap focuses and what that could mean dermoscopically).
2. If the two branches (Grad-CAM, SwinV2) disagree, explain what that implies for confidence.
3. Discuss the differential: why the predicted class wins over the closest alternative, or why the margin is uncomfortably narrow.
4. Use age/sex/site ONLY if they materially affect the reasoning (e.g., sun-exposed site for actinic keratosis, young female torso for benign nevi).
5. End with an honest uncertainty note: what the physician should look for on dermoscopy that the model cannot assess.

FORBIDDEN:
- Do not recommend biopsy, treatment, discharge, or reassurance as a directive.
- Do not fabricate dermoscopic features not supported by the supplied evidence.
- Do not restate the prediction, confidence, or threshold — the physician already has those.

Return exactly this schema:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LESIONIQ CLINICAL EXPLAINABILITY REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PREDICTION
  Diagnosis   : {diagnosis} ({code})
  Confidence  : {confidence}% ({confidence_level} confidence)
  Threshold   : {threshold} (tuned — default 0.50)

EVIDENCE
  <five to seven sentences of clinical reasoning as described above>"""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_predict 1024
```

**2. Create the model with Ollama:**

```bash
ollama create lesioniq-gemma3 -f Modelfile.lesioniq
```

**3. Verify it was created:**

```bash
ollama list
```

You should see `lesioniq-gemma3` in the list.

---

## Method 2: Test via API (no model creation needed)

You can test the system prompt immediately via API without creating a custom model:

```bash
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3:4b-it-qat",
    "messages": [
      {
        "role": "system",
        "content": "You are an expert clinical dermatology reviewer inside LesionIQ..."
      },
      {
        "role": "user",
        "content": "Explain the Grad-CAM findings for this melanoma case..."
      }
    ],
    "stream": false,
    "options": {
      "temperature": 0.3,
      "top_p": 0.9,
      "num_predict": 1024
    }
  }'
```

---

## Method 3: PowerShell on Windows Remote Machine

```powershell
# Create a Modelfile.lesioniq with the content above, then:
ollama create lesioniq-gemma3 -f Modelfile.lesioniq

# Test it:
curl.exe http://localhost:11434/api/tags | ConvertFrom-Json | Select-Object -ExpandProperty models
```

---

## What to do next on the backend machine

Once the custom model is created on the remote Ollama, update the backend's `.env`:

```env
OLLAMA_BASE_URL=http://172.22.62.82:11434
OLLAMA_MODEL=lesioniq-gemma3
```

Then restart the backend:

```powershell
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

---

## Verification

Test that everything is wired up:

```bash
# From backend machine:
curl https://lesion-iq.vercel.app/health
# Should show "ollamaModel": "lesioniq-gemma3"
```

Or upload an image through the frontend at `https://lesion-iq.vercel.app` — the clinical report should be generated with the new system prompt active.
