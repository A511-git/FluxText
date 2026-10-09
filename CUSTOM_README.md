# Quickstart Runbook for FluxText
> Blackwell GPU Docker Container & Host Remote Bridge Guide

This repository is configured to run inside a high-performance **NVIDIA Blackwell GPU Docker container**, while keeping the **Cloudflare Remote Bridge strictly on the host system** so web links and file uploads never disconnect during container restarts.

---

## Architecture Overview

```
Host System (Outside Docker)
 ├── Remote Bridge Daemon (Cloudflare Tunnels)
 │    ├── [CONFIRMED LINK 1] Upload Portal UI  --> Saves to host ./uploaded_stuff
 │    ├── [CONFIRMED LINK 2] Repo HTTP Viewer  --> Browses host ./ (results, logs, code)
 │    └── [OPTIONAL  LINK 3] Inbuilt Gradio    --> Proxies container port 6681
 └── Host Storages (Mounted into Container)
      ├── ./ (entire repo)       --> Mounted to /workspace/FluxText
      └── ~/.cache/huggingface   --> Mounted to /root/.cache/huggingface
```

---

## Step 1: Start the Remote Bridge on the Host (Background Mode)

Run the Remote Bridge in the background on your host machine to activate your public Cloudflare URLs:

### Linux / macOS / Remote SSH Host:
```bash
mkdir -p .remote_bridge && nohup python3 scripts/remote_bridge_launcher.py --repo . > .remote_bridge/bridge.log 2>&1 &
```

### Windows (PowerShell):
```powershell
Start-Process python -ArgumentList "scripts/remote_bridge_launcher.py --repo ." -WindowStyle Hidden
```

### View Your Active Access Links:
```bash
cat .remote_bridge/bridge_links.md
```

You will get 2 confirmed active links (and 1 optional frontend link):
- **[CONFIRMED LINK 1] Upload Portal**: Drag-and-drop test images, weights, or entire folders from your browser directly into `./uploaded_stuff/`.
- **[CONFIRMED LINK 2] Repo HTTP Viewer**: Browse and download code, logs, and generated images in `./results/` directly in your browser.
- **[OPTIONAL  LINK 3] Inbuilt Frontend**: Online public link for the Gradio interface (mapped from container port `6681`).

---

## Step 2: Start & Enter the GPU Docker Container

On the host machine, build and launch the container in the background:

```bash
docker compose up --build -d
```

Attach a bash shell into the container:
```bash
docker compose exec fluxtext bash
```

> **Note**: Entering the container automatically places you inside `/workspace/FluxText`. All relative paths (`./`) refer directly to this repository!

---

## Step 3: Run Inference Inside the Container

Once inside the container (`/workspace/FluxText`):

### Option A: Universal Batch Inference (Pointed to Uploaded Files)
Use the batch inference runner to process images uploaded via Link 1:
```bash
python scripts/batch_inference_runner.py -i ./uploaded_stuff -o ./results
```
Outputs are written to `./results/` and can immediately be inspected in your browser via **Link 2 (Repo HTTP Viewer)**!

### Option B: Launch Inbuilt Gradio Frontend
```bash
python app.py
# Or for low VRAM mode:
python app_low_VRAM.py
```
The app binds to port `6681` inside the container, which is forwarded to the host and accessible over the public internet via **Link 3**!

---

## Step 4: Clean Teardown

When finished:

1. **Stop the Docker Container (on Host):**
   ```bash
   docker compose down
   ```

2. **Stop the Background Remote Bridge (on Host):**
   - **Linux / macOS:**
     ```bash
     pkill -f remote_bridge_launcher.py
     ```
   - **Windows (PowerShell):**
     ```powershell
     Get-Process -Name python, cloudflared -ErrorAction SilentlyContinue | Stop-Process
     ```
