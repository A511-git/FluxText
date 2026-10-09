#!/usr/bin/env python3
"""
FLUX-Text & Blackwell Model Weight Downloader
Safely downloads official FLUX-Text checkpoints from Hugging Face (GD-ML/FLUX-Text),
the base model (black-forest-labs/FLUX.1-Fill-dev), and BLIP-2 components with
authenticated gated repo access, token support, and resume capability.
"""

import argparse
import os
import sys
from pathlib import Path


def get_hf_token(cli_token: str = None) -> str:
    """Resolves HF token from CLI flag, environment variables, or local .env file."""
    if cli_token:
        return cli_token

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token

    # Check local .env file
    env_file = Path(".env")
    if env_file.exists():
        try:
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN=") or line.startswith("HUGGING_FACE_HUB_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if token:
                        return token
        except Exception:
            pass

    return None


def download_flux_text_weights(dest_dir: str = "./models", model_type: str = "multisize", token: str = None, download_base: bool = False):
    dest_path = Path(dest_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)
    resolved_token = get_hf_token(token)

    print("=" * 72)
    print("           FLUX-TEXT OFFICIAL WEIGHT DOWNLOADER")
    print(f"[*] Destination Directory : {dest_path}")
    print(f"[*] Target Model Type     : {model_type}")
    print(f"[*] HF Authentication     : {'TOKEN CONFIGURED' if resolved_token else 'ANONYMOUS (Set HF_TOKEN for gated models)'}")
    print("=" * 72)

    try:
        from huggingface_hub import hf_hub_download, snapshot_download, login
    except ImportError:
        print("[!] Installing huggingface_hub...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download, snapshot_download, login

    if resolved_token:
        try:
            login(token=resolved_token, add_to_git_credential=False)
            print("[+] Logged into Hugging Face successfully.")
        except Exception as e:
            print(f"[!] Warning: HF login verification notice: {e}")

    # 1. Download FLUX-Text adapter checkpoints
    repo_id = "GD-ML/FLUX-Text"
    subfolder = "model_multisize" if model_type == "multisize" else "model_512"

    print(f"\n[*] Downloading {subfolder} from {repo_id}...")
    try:
        downloaded_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{subfolder}/*", "epoch_100.pt", "*.yaml", "*.safetensors"],
            local_dir=str(dest_path),
            local_dir_use_symlinks=False,
            resume_download=True,
            token=resolved_token,
        )
        print(f"[+] FLUX-Text checkpoints stored in: {downloaded_dir}")
    except Exception as e:
        print(f"[!] Error downloading {repo_id}: {e}")

    # 2. Optionally pre-download the gated base model (FLUX.1-Fill-dev)
    if download_base:
        base_repo = "black-forest-labs/FLUX.1-Fill-dev"
        print(f"\n[*] Pre-downloading base gated model: {base_repo}...")
        if not resolved_token:
            print(f"[!] WARNING: {base_repo} is a gated model. You MUST provide an HF_TOKEN with accepted terms.")
        try:
            snapshot_download(
                repo_id=base_repo,
                resume_download=True,
                token=resolved_token,
            )
            print(f"[+] Base model '{base_repo}' cached successfully in HF cache.")
        except Exception as e:
            print(f"[!] Error downloading {base_repo}: {e}")
            print(f"    Ensure your token has accepted access at: https://huggingface.co/{base_repo}")


def main():
    parser = argparse.ArgumentParser(description="Download official FLUX-Text and base models")
    parser.add_argument("--dest", "-d", type=str, default="./models", help="Destination folder for checkpoints")
    parser.add_argument("--type", "-t", type=str, choices=["multisize", "512"], default="multisize", help="Model resolution version")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token (or set HF_TOKEN env var)")
    parser.add_argument("--download-base", action="store_true", help="Also pre-download the gated black-forest-labs/FLUX.1-Fill-dev base model")
    args = parser.parse_args()

    download_flux_text_weights(
        dest_dir=args.dest,
        model_type=args.type,
        token=args.token,
        download_base=args.download_base,
    )


if __name__ == "__main__":
    main()
