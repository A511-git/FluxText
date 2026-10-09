#!/usr/bin/env python3
"""
Universal Batch Inference Runner & Image Processing Utility
Provides robust folder loading, multi-extension support, RGB normalization,
and batch generation for testing vision/generative models.
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image

VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")

def find_valid_images(folder_path, recursive=False):
    """
    Scans a folder and returns sorted paths of all valid image files.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Input directory does not exist: {folder_path}")

    pattern = "**/*" if recursive else "*"
    images = [
        str(p) for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]
    return sorted(images)

def load_image_rgb(image_path):
    """
    Safely opens an image and converts it to 3-channel RGB (preventing RGBA/grayscale issues).
    """
    with Image.open(image_path) as img:
        return img.convert("RGB")

def run_batch_inference_pipeline(input_dir, output_dir, inference_fn=None, recursive=False):
    """
    Loads images, calls the inference callback, and saves outputs.
    """
    images = find_valid_images(input_dir, recursive=recursive)
    print(f"[*] Found {len(images)} valid images in '{input_dir}'")
    if not images:
        print("[!] No images found to process.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    results = []

    try:
        from tqdm import tqdm
        iterator = tqdm(images, desc="Inference")
    except ImportError:
        iterator = images

    for img_path in iterator:
        img_name = os.path.basename(img_path)
        out_path = os.path.join(output_dir, img_name)

        try:
            pil_img = load_image_rgb(img_path)
            if inference_fn is not None:
                out_img = inference_fn(pil_img, img_path)
                if isinstance(out_img, Image.Image):
                    out_img.save(out_path)
                results.append((img_path, out_path, True))
            else:
                # Default demonstration pass-through
                pil_img.save(out_path)
                results.append((img_path, out_path, True))
        except Exception as e:
            print(f"[!] Error processing {img_name}: {e}")
            results.append((img_path, out_path, False))

    print(f"[+] Completed! Outputs saved to '{output_dir}'.")
    return results

def main():
    parser = argparse.ArgumentParser(description="Universal Batch Image Inference Helper")
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Path to input image folder")
    parser.add_argument("--output_dir", "-o", type=str, default="./results", help="Directory to save output images")
    parser.add_argument("--recursive", "-r", action="store_true", help="Search images recursively")
    args = parser.parse_args()

    run_batch_inference_pipeline(args.input_dir, args.output_dir, recursive=args.recursive)

if __name__ == "__main__":
    main()
