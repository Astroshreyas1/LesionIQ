# Setting up the SLM on the other laptop

The backend (inference model) stays on this laptop.
Ollama (the SLM — `gemma3:4b-it-qat`) runs on the other laptop.
They communicate over your LAN.

---

## On the OTHER laptop (SLM machine)

### 1. Install Ollama

**Windows:**
Download and run the installer from https://ollama.com/download

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Allow external connections

By default Ollama only listens on localhost. You need to make it listen on all interfaces.

**Windows** — set a system environment variable before starting Ollama:
```powershell
[System.Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0", "User")
# Then restart Ollama (close and reopen the app)
```

**Linux:**
```bash
# Edit the systemd service
sudo systemctl edit ollama
# Add these lines inside the [Service] block:
# [Service]
# Environment="OLLAMA_HOST=0.0.0.0"

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### 3. Pull the model

```bash
ollama pull gemma3:4b-it-qat
```

### 4. Find this laptop's LAN IP

**Windows:**
```powershell
ipconfig
# Look for "IPv4 Address" under your active adapter — e.g. 192.168.1.42
```

**Linux:**
```bash
ip a | grep "inet " | grep -v 127
```

### 5. Test it's reachable from the backend laptop

From the backend laptop's terminal:
```powershell
curl http://<other-laptop-ip>:11434/
# Should return: {"status":"Ollama is running"}
```

---

## On THIS laptop (backend machine)

Open `C:\LesionIQ\.env` and update `OLLAMA_BASE_URL`:

```
# Comment out the local line:
# OLLAMA_BASE_URL=http://ollama:11434

# Uncomment and fill in the other laptop's IP:
OLLAMA_BASE_URL=http://192.168.x.x:11434
```

Then start only the backend (no local Ollama):

```powershell
docker compose --profile remote-slm up --build
```

---

## Switching back to local SLM

Revert `.env` to:
```
OLLAMA_BASE_URL=http://ollama:11434
```

Then start normally:
```powershell
docker compose up --build
```
