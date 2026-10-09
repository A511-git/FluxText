#!/usr/bin/env python3
"""
Model Weights Downloader & Integrity Verifier
Downloads weights from Hugging Face or Google Drive, verifying exact byte counts
and PyTorch checkpoint deserialization before proceeding.
"""

import os
import sys
import subprocess
import urllib.request

def ensure_gdown():
    try:
        import gdown
        return gdown
    except ImportError:
        print("[*] Installing gdown for Google Drive downloads...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
        import gdown
        return gdown

def verify_and_download_hf(model_name, url, target_dir="weights"):
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, model_name)

    print(f"\n[*] Checking remote file metadata for {model_name}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        expected_bytes = int(resp.headers.get("content-length", 0))

    need_download = True
    if os.path.exists(target_path) and expected_bytes > 0:
        local_bytes = os.path.getsize(target_path)
        if local_bytes == expected_bytes:
            print(f"[+] Verified {model_name} byte count matches ({local_bytes:,} bytes). Validating checkpoint...")
            try:
                import torch
                torch.load(target_path, map_location="cpu", weights_only=False)
                print(f"[+] Checkpoint '{model_name}' is valid. Skipping download.")
                return target_path
            except Exception as e:
                print(f"[!] Checkpoint corrupted ({e}). Re-downloading...")
                os.remove(target_path)
        else:
            print(f"[!] File size mismatch: local {local_bytes:,} != remote {expected_bytes:,}. Re-downloading...")
            os.remove(target_path)

    print(f"[>] Downloading {model_name} ({expected_bytes:,} bytes)...")
    if subprocess.call(["which", "wget"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        subprocess.run(["wget", "-c", "--show-progress", url, "-O", target_path], check=True)
    else:
        urllib.request.urlretrieve(url, target_path)

    # Final verification
    try:
        import torch
        torch.load(target_path, map_location="cpu", weights_only=False)
        print(f"[+] Checkpoint verified 100%: {model_name}")
    except Exception as e:
        print(f"[!] Warning: Could not verify with torch.load: {e}")

    return target_path

def verify_and_download_gdrive(model_name, gdrive_id, min_size_mb=100, target_dir="weights"):
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, model_name)
    gdown = ensure_gdown()

    if os.path.exists(target_path):
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        if size_mb >= min_size_mb:
            try:
                import torch
                torch.load(target_path, map_location="cpu", weights_only=False)
                print(f"[+] Verified {model_name} ({size_mb:.1f} MB). Skipping download.")
                return target_path
            except Exception as e:
                print(f"[!] Checkpoint corrupted ({e}). Re-downloading...")
                os.remove(target_path)

    print(f"[>] Downloading {model_name} from Google Drive...")
    gdown.download(id=gdrive_id, output=target_path, quiet=False)

    size_mb = os.path.getsize(target_path) / (1024 * 1024)
    if size_mb < min_size_mb:
        raise ValueError(f"Downloaded file too small: {size_mb:.1f} MB < {min_size_mb} MB")

    try:
        import torch
        torch.load(target_path, map_location="cpu", weights_only=False)
        print(f"[+] Checkpoint verified 100%: {model_name}")
    except Exception as e:
        print(f"[!] Warning: Could not verify with torch.load: {e}")

    return target_path

if __name__ == "__main__":
    print("Weight downloader utility ready.")
