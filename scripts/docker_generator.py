#!/usr/bin/env python3
"""
Automated Docker & Docker Compose Generator for Blackwell AI Model Runner
Inspects repository architecture and dependencies, then generates:
1. Dockerfile configured for NVIDIA Blackwell GPUs (CUDA 12.8) with WORKDIR /workspace/[repo_name]
2. docker-compose.yml mapping ./:/workspace/[repo_name] with GPU pass-through
3. CUSTOM_README.md in repo root documenting background Remote Bridge, Docker commands, and CLI inference
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Import repo analyzer from the same scripts directory
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from repo_analyzer import RepoAnalyzer


def generate_dockerfile_content(repo_name: str) -> str:
    return f"""# ==============================================================================
# Blackwell GPU AI Model Runner Dockerfile
# Optimized for NVIDIA Blackwell (RTX PRO 4500, B100, B200 - sm_100/sm_120)
# ==============================================================================

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="12.0;10.0"

# Install essential system dependencies and modern Python
RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3.10 \\
    python3.10-dev \\
    python3-pip \\
    git \\
    wget \\
    curl \\
    ffmpeg \\
    libsm6 \\
    libxext6 \\
    libgl1 \\
    libglib2.0-0 \\
    build-essential \\
    ninja-build \\
    ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \\
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Install uv for fast, deterministic dependency resolution
RUN pip install --no-cache-dir uv ninja gdown wheel setuptools

# Install PyTorch with CUDA 12.8 (Native Blackwell sm_100/sm_120 Support)
RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Bypass NVCC version mismatch check in PyTorch cpp_extension
RUN python -c 'import torch.utils.cpp_extension as ce; p = ce.__file__; s = open(p).read(); t = "raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)"; sub = "print(f\\"[WARNING] CUDA mismatch: {cuda_str_version} vs {torch.version.cuda}\\")"; open(p, "w").write(s.replace(t, sub)) if t in s else None'

# Set working directory inside container
WORKDIR /workspace/{repo_name}

# Install repo dependencies if requirements.txt exists
COPY requirements.txt* /workspace/{repo_name}/
RUN if [ -f /workspace/{repo_name}/requirements.txt ]; then \\
        sed -i '/xformers/d' /workspace/{repo_name}/requirements.txt && \\
        sed -i '/^torch==/d' /workspace/{repo_name}/requirements.txt && \\
        sed -i '/^torchvision==/d' /workspace/{repo_name}/requirements.txt && \\
        sed -i '/^torchaudio==/d' /workspace/{repo_name}/requirements.txt && \\
        sed -i '/^triton==/d' /workspace/{repo_name}/requirements.txt && \\
        uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r /workspace/{repo_name}/requirements.txt || true ; \\
    fi

# Ensure modern transformers & accelerate using cu128 index
RUN uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    "transformers>=4.37.0" "accelerate>=0.28.0" diffusers peft \
    opencv-python-headless gradio matplotlib pyyaml einops ftfy sentencepiece lightning prodigyopt \
    ujson easydict scikit-image Levenshtein pandas pandarallel webcolors av lpips \
    mmengine modelscope safetensors datasets "numpy<2" tqdm requests "openai-clip>=1.0.1"

# Default entry command
CMD ["/bin/bash"]
"""


def generate_docker_compose_yaml(service_name: str, repo_name: str, frontend_port: int) -> str:
    ports_section = ""
    if frontend_port:
        ports_section = f"""    ports:
      - "{frontend_port}:{frontend_port}"
"""

    compose_yaml = f"""services:
  {service_name}:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: {service_name}_blackwell_container
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - FORCE_CUDA=1
      - TORCH_CUDA_ARCH_LIST=12.0;10.0
      - UV_SYSTEM_PYTHON=1
      - HF_HOME=/root/.cache/huggingface
{ports_section}    volumes:
      # Mount entire repository into /workspace/{repo_name}
      - ./:/workspace/{repo_name}
      # Shared Hugging Face cache on high-capacity NVMe drive (380+ GB)
      - ${{HF_CACHE_DIR:-/opt/dlami/nvme/huggingface_cache}}:/root/.cache/huggingface
    ipc: host
    stdin_open: true
    tty: true
    working_dir: /workspace/{repo_name}
    command: /bin/bash
"""
    return compose_yaml


def generate_custom_readme(repo_name: str, service_name: str, has_frontend: bool, frontend_type: str, frontend_port: int, entrypoint: str) -> str:
    fe_section = ""
    if has_frontend:
        fe_section = f"""### Option B: Launch Inbuilt Frontend ({frontend_type.upper()})
```bash
python {entrypoint}
```
The frontend binds to port `{frontend_port}` inside the container, which is forwarded to the host and accessible over the public internet via **Link 3**!
"""
    else:
        fe_section = """*Note: No inbuilt frontend detected in this repository. All execution is performed via CLI batch inference above.*
"""

    return f"""# Quickstart Runbook for {repo_name}
> Automated Blackwell GPU Container & Remote Bridge Guide

This repository is configured to run inside a high-performance **NVIDIA Blackwell GPU Docker container**, while keeping the **Cloudflare Remote Bridge strictly on the host system** so web links and file uploads never disconnect during container restarts.

---

## Architecture Overview

```
Host System (Outside Docker)
 ├── Remote Bridge Daemon (Cloudflare Tunnels)
 │    ├── Confirmed Link 1: Upload Portal UI  --> Saves to host ./uploaded_stuff
 │    ├── Confirmed Link 2: Repo HTTP Viewer  --> Browses host ./
 │    └── Optional  Link 3: Inbuilt Frontend  --> Proxies port {frontend_port}
 └── Host Storages (Mounted into Container)
      ├── ./ (entire repo)       --> Mounted to /workspace/{repo_name}
      └── ~/.cache/huggingface   --> Mounted to /root/.cache/huggingface
```

---

## Step 1: Start the Remote Bridge on the Host (Background Mode)

Run the Remote Bridge in the background on your host machine to get your public Cloudflare URLs:

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
- **Confirmed Link 1 (Upload Portal)**: Drag-and-drop test files or entire folders from your browser directly into `./uploaded_stuff/`.
- **Confirmed Link 2 (Repo HTTP Viewer)**: Browse and download code, logs, and generated images in `./results/`.
- **Optional Link 3 (Inbuilt Frontend)**: Live interactive UI if {frontend_type or 'a frontend'} is running.

---

## Step 2: Start & Enter the GPU Docker Container

On the host machine, build and launch the container in the background:

```bash
docker compose up --build -d
```

Attach a bash shell into the container:
```bash
docker compose exec {service_name} bash
```
> **Note**: Entering the container automatically places you inside `/workspace/{repo_name}`. All relative paths (`./`) refer directly to this repository!

---

## Step 3: Run Inference Inside the Container

Once inside the container (`/workspace/{repo_name}`):

### Option A: Universal Batch Inference (Pointed to Uploaded Files)
Use the batch inference runner to process images uploaded via Link 1:
```bash
python scripts/batch_inference_runner.py -i ./uploaded_stuff -o ./results
```
Outputs are written to `./results/` and can immediately be inspected in your browser via **Link 2 (Repo HTTP Viewer)**!

{fe_section}

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
"""


class DockerSetupGenerator:
    def __init__(self, repo_dir: str = "."):
        self.repo_dir = Path(repo_dir).resolve()
        self.analyzer = RepoAnalyzer(str(self.repo_dir))

    def generate(self, overwrite: bool = False):
        analysis = self.analyzer.analyze()
        repo_name = analysis.get("repo_name", "model_runner")
        service_name = repo_name.lower().replace(" ", "_").replace("-", "_")
        frontend_port = analysis.get("detected_port", 6681)
        entrypoint = analysis.get("frontend_entrypoint", "app.py")
        has_frontend = analysis.get("has_frontend", False)
        frontend_type = analysis.get("frontend_type", "")

        print("=" * 70)
        print("          DOCKER CONTAINER & RUNBOOK GENERATOR")
        print("=" * 70)
        print(f"[*] Target Repository : {self.repo_dir}")
        print(f"[*] Repository Name   : {repo_name}")
        print(f"[*] Inbuilt Frontend  : {has_frontend} ({frontend_type})")
        print(f"[*] Container Workdir : /workspace/{repo_name}")
        print("=" * 70)

        # 1. Ensure required host storage mount folders exist
        uploaded_dir = self.repo_dir / "uploaded_stuff"
        results_dir = self.repo_dir / "results"
        uploaded_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        # 2. Write Dockerfile
        dockerfile_path = self.repo_dir / "Dockerfile"
        if dockerfile_path.exists() and not overwrite:
            print(f"[*] Dockerfile already exists at {dockerfile_path}")
        else:
            dockerfile_path.write_text(generate_dockerfile_content(repo_name), encoding="utf-8")
            print(f"[+] Created Blackwell Dockerfile: {dockerfile_path}")

        # 3. Write docker-compose.yml
        compose_path = self.repo_dir / "docker-compose.yml"
        if compose_path.exists() and not overwrite:
            print(f"[*] docker-compose.yml already exists at {compose_path}")
        else:
            compose_content = generate_docker_compose_yaml(
                service_name=service_name,
                repo_name=repo_name,
                frontend_port=frontend_port
            )
            compose_path.write_text(compose_content, encoding="utf-8")
            print(f"[+] Created docker-compose.yml: {compose_path}")

        # 4. Write CUSTOM_README.md in root
        readme_path = self.repo_dir / "CUSTOM_README.md"
        if readme_path.exists() and not overwrite:
            print(f"[*] CUSTOM_README.md already exists at {readme_path}")
        else:
            readme_content = generate_custom_readme(
                repo_name=repo_name,
                service_name=service_name,
                has_frontend=has_frontend,
                frontend_type=frontend_type,
                frontend_port=frontend_port,
                entrypoint=entrypoint
            )
            readme_path.write_text(readme_content, encoding="utf-8")
            print(f"[+] Created root CUSTOM_README.md: {readme_path}")

        print("\n[+] All Docker & Runbook assets generated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Docker & Docker Compose Generator for Blackwell AI Repos")
    parser.add_argument("--repo", "-r", type=str, default=".", help="Target repository directory path")
    parser.add_argument("--overwrite", "-w", action="store_true", help="Overwrite existing Docker files")
    args = parser.parse_args()

    generator = DockerSetupGenerator(args.repo)
    generator.generate(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
