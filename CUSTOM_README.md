# Quickstart Runbook for FluxText
> Blackwell GPU Docker Container & 2-Tier Storage Remote Bridge Guide

This repository is configured to run inside a high-performance **NVIDIA Blackwell GPU Docker container**, while enforcing a **strict 2-tier storage architecture**:
1. **Persistent Storage (72 GB)** at `/home/ubuntu/working/FluxText` (Source code, git history, configs).
2. **Ephemeral NVMe Workspace (410 GB)** at `/opt/dlami/nvme/workspace` (Hugging Face cache, checkpoints, uploaded files, and batch inference results).

The **Cloudflare Remote Bridge runs strictly on the host system** so web links and file uploads never disconnect during container restarts.

---

## Architecture Overview (2-Tier Storage Layout)

```
Persistent Storage (/home/ubuntu/working/ - 72 GB)
 └── FluxText/ (Code, Git history, .env, configs)
      ├── ./uploaded_stuff ➔ Symlinked to NVMe workspace/uploaded_stuff
      ├── ./results        ➔ Symlinked to NVMe workspace/results
      ├── ./models         ➔ Symlinked to NVMe workspace/models
      └── ./nvme_workspace ➔ Symlinked to NVMe /opt/dlami/nvme/workspace

NVMe Ephemeral Storage (/opt/dlami/nvme/workspace/ - 410 GB)
 ├── cache/huggingface/ (Base models & checkpoints)
 ├── uploaded_stuff/    (User uploaded assets & folders via Link 1)
 ├── results/           (Inference outputs & generated images via Link 2)
 └── models/            (FLUX-Text checkpoints & custom LoRAs)

Host System Remote Bridge Daemon (Cloudflare Tunnels)
 ├── [CONFIRMED LINK 1] Upload Portal UI  --> Saves directly to NVMe uploaded_stuff
 ├── [CONFIRMED LINK 2] Repo HTTP Viewer  --> Browses code, results, nvme_workspace
 └── [OPTIONAL  LINK 3] Inbuilt Frontend  --> Proxies container port 6681
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
- **[CONFIRMED LINK 1] Upload Portal**: Drag-and-drop test images, weights, or entire folders from your browser directly into NVMe `./uploaded_stuff/`.
- **[CONFIRMED LINK 2] Repo HTTP Viewer**: Browse and download code, logs, and generated images in `./results/` and `./nvme_workspace/` directly in your browser.
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
