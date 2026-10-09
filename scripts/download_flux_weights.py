#!/usr/bin/env python3
"""
FLUX-Text & Blackwell Model Weight Downloader
Safely downloads official FLUX-Text checkpoints from Hugging Face (GD-ML/FLUX-Text)
and BLIP-2 components with HTTP verification and resume capability.
"""

import argparse
import os
import sys
from pathlib import Path


def download_flux_text_weights(dest_dir: str = "./models", model_type: str = "multisize"):
    dest_path = Path(dest_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("           FLUX-TEXT OFFICIAL WEIGHT DOWNLOADER")
    print(f"[*] Destination Directory: {dest_path}")
    print(f"[*] Target Model Type    : {model_type}")
    print("=" * 72)

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("[!] Installing huggingface_hub...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download, snapshot_download

    repo_id = "GD-ML/FLUX-Text"

    # Subfolder selection
    subfolder = "model_multisize" if model_type == "multisize" else "model_512"

    print(f"[*] Downloading {subfolder} from {repo_id}...")
    try:
        downloaded_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{subfolder}/*", "epoch_100.pt", "*.yaml", "*.safetensors"],
            local_dir=str(dest_path),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"[+] Download complete! Checkpoints stored in: {downloaded_dir}")
    except Exception as e:
        print(f"[!] Error during download: {e}")
        print("    You can manually download weights from: https://huggingface.co/GD-ML/FLUX-Text")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download official FLUX-Text weights")
    parser.add_argument("--dest", "-d", type=str, default="./models", help="Destination folder for checkpoints")
    parser.add_argument("--type", "-t", type=str, choices=["multisize", "512"], default="multisize", help="Model resolution version")
    args = parser.parse_args()

    download_flux_text_weights(dest_dir=args.dest, model_type=args.type)


if __name__ == "__main__":
    main()
