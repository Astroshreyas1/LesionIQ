# Setting up LesionIQ System Prompt in Ollama

The SLM (Small Language Model) on the remote machine needs to know how to generate clinical reports. This guide sets up a custom Ollama model with the LesionIQ system prompt baked in.

## On the remote laptop where Ollama is running

### Option A: Using the Modelfile (Recommended)

1. **Copy the Modelfile** from this repo to the remote machine:
   ```bash
   # On Windows
   Copy-Item ".\Modelfile.lesioniq" -Destination "C:\path\to\ollama\models"
   
   # On Linux
   scp Modelfile.lesioniq user@<remote-ip>:~/
   ```

2. **Create the custom model** by running Ollama on the remote machine:
   ```bash
   ollama create lesioniq-gemma3 -f Modelfile.lesioniq
   ```
   
   This creates a new model called `lesioniq-gemma3` that wraps `gemma3:4b-it-qat` with the system prompt.

3. **Verify it was created:**
   ```bash
   ollama list
   # Should show: lesioniq-gemma3    latest
   ```

### Option B: Inline system message (if not using Modelfile)

If you prefer to skip the Modelfile, the backend can send the system prompt as a separate message. The backend already does this, so no action needed on the Ollama side.

## Updating the backend to use the custom model

Edit the `.env` file in the LesionIQ backend:

```env
OLLAMA_BASE_URL=http://172.22.62.82:11434
OLLAMA_MODEL=lesioniq-gemma3  # <-- Changed from gemma3:4b-it-qat
```

Then restart the backend:
```powershell
# Kill the current backend and tunnel
# Then restart:
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

## Verifying it works

Test the model directly:
```bash
curl http://localhost:11434/api/chat \
  -d '{"model": "lesioniq-gemma3", "messages": [{"role": "user", "content": "You are clinical expert. Explain dermoscopy."}], "stream": false}'
```

Or test through the backend:
```bash
curl https://lesion-iq.vercel.app/health
# Should see the new model name
```

## Rolling back

If you want to go back to the base `gemma3:4b-it-qat`:
1. Delete the custom model: `ollama rm lesioniq-gemma3`
2. Update `.env`: `OLLAMA_MODEL=gemma3:4b-it-qat`
3. Restart the backend

## Notes

- The Modelfile includes the exact system prompt used by LesionIQ
- Parameters are tuned for clinical reasoning (temperature 0.3, top_p 0.9, max 1024 tokens)
- The custom model inherits all capabilities of `gemma3:4b-it-qat`
- The system prompt is now always active—no need to send it with every request
