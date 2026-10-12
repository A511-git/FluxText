import argparse
import math
import json
import os
import os.path as osp
from pathlib import Path
from shutil import copyfile
import sys
import random
import yaml

import cv2
import gradio as gr
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import torchvision.transforms as T

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'eval')))
from eval.t3_dataset import draw_glyph2
from src.flux.condition import Condition
from src.flux.generate_fill import generate_fill, generate_fill_low_RAM
from src.train.model import OminiModelFIll

ASPECT_RATIO_LD_LIST = [  # width:height
    "2.39:1",  # cinemascope, 2.39
    "2:1",  # rare, 2
    "16:9",  # rare, 1.89
    "1.85:1",  # american widescreen, 1.85
    "9:16",  # popular, 1.78
    "5:8",  # rare, 1.6
    "3:2",  # rare, 1.5
    "4:3",  # classic, 1.33
    "1:1",  # square
]

RESOLUTIONS = [512, 768, 1024]
PIXELS = [512 * 512, 768 * 768, 1024 * 1024]

def get_ratio(name: str) -> float:
    width, height = map(float, name.split(":"))
    return height / width

def get_closest_ratio(height: float, width: float, ratios: dict) -> str:
    aspect_ratio = height / width
    closest_ratio = min(
        ratios, key=lambda ratio: abs(aspect_ratio - get_ratio(ratio))
    )
    return closest_ratio

def get_aspect_ratios_dict(
    total_pixels: int = 256 * 256, training: bool = True
) -> dict[str, tuple[int, int]]:
    D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 16))
    aspect_ratios_dict = {}
    aspect_ratios_vertical_dict = {}
    for ratio in ASPECT_RATIO_LD_LIST:
        width_ratio, height_ratio = map(float, ratio.split(":"))
        width = int(math.sqrt(total_pixels * (width_ratio / height_ratio)) // D) * D
        height = int((total_pixels / width) // D) * D

        if training:
            # adjust aspect ratio to match total pixels
            diff = abs(height * width - total_pixels)
            candidate = [
                (height - D, width),
                (height + D, width),
                (height, width - D),
                (height, width + D),
            ]
            for h, w in candidate:
                if abs(h * w - total_pixels) < diff:
                    height, width = h, w
                    diff = abs(h * w - total_pixels)

        # remove duplicated aspect ratio
        if (height, width) not in aspect_ratios_dict.values() or not training:
            aspect_ratios_dict[ratio] = (height, width)
            vertial_ratios = ":".join(ratio.split(":")[::-1])
            aspect_ratios_vertical_dict[vertial_ratios] = (width, height)

    aspect_ratios_dict.update(aspect_ratios_vertical_dict)

    return aspect_ratios_dict

def load_app_config(config_file="config.yaml"):
    """Loads root config.yaml if present."""
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[!] Notice reading {config_file}: {e}")
    return {}

APP_CONFIG = load_app_config()

def ensure_checkpoint(model_path: str = None, model_type: str = "512") -> str:
    """Ensures model checkpoint exists, auto-downloading from Hugging Face if missing."""
    if model_path and os.path.exists(model_path):
        return model_path

    # Check common alternative locations
    candidates = [
        f"models/model_{model_type}/pytorch_lora_weights.safetensors",
        "models/model_512/pytorch_lora_weights.safetensors",
        "models/model_multisize/pytorch_lora_weights.safetensors",
        "models/pytorch_lora_weights.safetensors",
        "models/model_512/epoch_100.safetensors",
        "models/model_multisize/epoch_100.safetensors",
        "models/epoch_100.safetensors",
        "models/epoch_100.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"[+] Found existing checkpoint: {c}")
            return c

    # Search recursively for any safetensors in models/
    import glob
    found = sorted(glob.glob("models/**/*.safetensors", recursive=True))
    if found:
        print(f"[+] Found existing checkpoint: {found[0]}")
        return found[0]

    # Auto-download from Hugging Face
    target_sub = f"model_{model_type}" if model_type in ("512", "multisize") else "model_512"
    filename = f"{target_sub}/pytorch_lora_weights.safetensors"
    print("=" * 72)
    print(f"[*] FLUX-Text checkpoint '{model_path or filename}' not found locally.")
    print(f"[*] Auto-downloading '{filename}' from Hugging Face (GD-ML/FLUX-Text)...")
    print("=" * 72)

    try:
        from huggingface_hub import hf_hub_download
        os.makedirs("models", exist_ok=True)
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        downloaded = hf_hub_download(
            repo_id="GD-ML/FLUX-Text",
            filename=filename,
            local_dir="models",
            token=token,
        )
        print(f"[+] Checkpoint successfully downloaded to: {downloaded}")
        return downloaded
    except Exception as e:
        print(f"[!] Auto-download failed: {e}")
        print(f"[*] You can manually download via: python scripts/download_flux_weights.py --type {model_type}")
        raise FileNotFoundError(f"Checkpoint {filename} could not be found or downloaded.")


def parse_args():
    cfg_model = APP_CONFIG.get("model", {})
    cfg_hw = APP_CONFIG.get("hardware", {})

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to master config.yaml"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=cfg_model.get("checkpoint_path", None),
        help='path of model checkpoint (defaults to config.yaml setting)'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default=cfg_model.get("config_path", 'train/config/word_512_size.yaml'),
        help='path of config file (default: train/config/word_512_size.yaml)'
    )
    parser.add_argument(
        '--type',
        type=str,
        default=cfg_model.get("type", "512"),
        choices=["512", "multisize"],
        help='model type/resolution (default: 512)'
    )
    parser.add_argument(
        '--low_vram',
        action='store_true',
        default=True if cfg_hw.get("low_vram") is True else False,
        help='force low VRAM CPU-GPU offload mode (auto-enabled if GPU VRAM <= 36GB)'
    )
    args = parser.parse_args()

    # Automatically ensure checkpoint exists (downloading if missing)
    args.model_path = ensure_checkpoint(args.model_path, model_type=args.type)
    return args


USE_LOW_VRAM = False

def init_pipeline(args, config, is_low_vram=False):
    training_config = config["train"]
    target_device = "cpu" if is_low_vram else "cuda"
    trainable_model = OminiModelFIll(
            flux_pipe_id=config["flux_path"],
            lora_config=training_config["lora_config"],
            device=target_device,
            dtype=getattr(torch, config["dtype"]),
            optimizer_config=training_config["optimizer"],
            model_config=config.get("model", {}),
            gradient_checkpointing=training_config.get("gradient_checkpointing", False),
            byt5_encoder_config=training_config.get("byt5_encoder", None),
        )

    from safetensors.torch import load_file
    print(f"[*] Loading FLUX-Text LoRA weights from: {args.model_path}")
    state_dict = load_file(args.model_path)
    state_dict1 = {x.replace('lora_A', 'lora_A.default').replace('lora_B', 'lora_B.default').replace('transformer.', ''): v for x, v in state_dict.items()}
    trainable_model.transformer.load_state_dict(state_dict1, strict=False)

    pipe = trainable_model.flux_pipe
    return pipe, trainable_model

def get_captions(ori_image, _input_file):
    image = Image.fromarray(ori_image)
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)

    if USE_LOW_VRAM:
        blipmodel.to(device)
    generated_ids = blipmodel.generate(**inputs, max_new_tokens=20)
    if USE_LOW_VRAM:
        blipmodel.to('cpu')
        torch.cuda.empty_cache()

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    caption = f'{generated_text}, that reads "{_input_file}"'
    return caption

def get_glyph_pos(mask, _input_file, width, height):
    mask = mask.astype('uint8')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise gr.Error("No brush stroke detected! Please draw a visible white stroke over the text area.")
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    hint = mask / 255
    glyph_scale = 1
    glyphs = draw_glyph2(selffont, _input_file, contours[0], scale=glyph_scale, width=width, height=height)

    return hint, glyphs

def brush_button_func(brush_image):
    if not brush_image or 'layers' not in brush_image or not brush_image['layers']:
        raise gr.Error("No brush stroke found! Please paint over the text region you want to edit.")
    _mask = brush_image['layers'][0][:, :, :3]
    _mask = np.where(_mask > 0, 255, 0).astype(np.uint8)
    return _mask[:,:,0], [_mask[:,:,0]]

def update_mask_func(edit_mask, edit_text):
    if not edit_mask or 'background' not in edit_mask or edit_mask['background'] is None:
        raise gr.Error("Please upload an image into the editor first!")
    if not edit_text or not str(edit_text).strip():
        raise gr.Error("Please enter text into 'Input text' before clicking 'Generate glyph and mask'.")

    background = edit_mask['background']
    background = background[:, :, :3]
    mask, _ = brush_button_func(edit_mask)

    hint, glyphs = get_glyph_pos(mask, edit_text, background.shape[1], background.shape[0])
    hint = hint.astype('uint8') * 255
    glyphs = (1 - glyphs.astype('uint8') ) * 255
    glyphs = glyphs[:,:,0]
    caption = get_captions(background, edit_text)
    return hint, glyphs, caption

def generate_image_func(prompt, img, glyph_img=None, mask_img=None, seed=42, edit_mask=None, edit_text=""):
    # 1. Ensure base image exists (fallback to edit_mask background if present)
    if img is None and edit_mask is not None and isinstance(edit_mask, dict) and edit_mask.get('background') is not None:
        bg = edit_mask['background']
        if isinstance(bg, np.ndarray):
            img = Image.fromarray(bg[:, :, :3].astype(np.uint8))
        elif isinstance(bg, Image.Image):
            img = bg

    if img is None:
        raise gr.Error("Please upload an Image, or select one of the Examples below.")

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if isinstance(mask_img, np.ndarray):
        mask_img = Image.fromarray(mask_img)
    if isinstance(glyph_img, np.ndarray):
        glyph_img = Image.fromarray(glyph_img)

    ori_width, ori_height = img.size
    num_pixel = min(PIXELS, key=lambda x: abs(x - ori_width * ori_height))
    aspect_ratio_dict = get_aspect_ratios_dict(num_pixel)
    close_ratio = get_closest_ratio(ori_height, ori_width, ASPECT_RATIO_LD_LIST)
    tgt_height, tgt_width = aspect_ratio_dict[close_ratio]

    # 2. Check if mask can be recovered from manual edit canvas
    if mask_img is None and edit_mask is not None and isinstance(edit_mask, dict):
        layers = edit_mask.get('layers', [])
        if layers and len(layers) > 0 and layers[0] is not None:
            raw_layer = layers[0][:, :, :3]
            if np.any(raw_layer > 0):
                mask_np = np.where(raw_layer[:, :, 0] > 0, 255, 0).astype(np.uint8)
                mask_img = Image.fromarray(mask_np)

    # 3. Detect text specified in edit_text or prompt
    import re
    text_to_draw = None
    if edit_text and str(edit_text).strip():
        text_to_draw = str(edit_text).strip()
    elif prompt:
        p_str = str(prompt).strip()
        p_lower = p_str.lower()
        if "that reads" in p_lower:
            idx = p_lower.find("that reads") + len("that reads")
            text_to_draw = p_str[idx:].strip().strip('.:,"\'')
        elif "written materials on the picture:" in p_lower:
            idx = p_lower.find("written materials on the picture:") + len("written materials on the picture:")
            text_to_draw = p_str[idx:].strip().strip('.:,"\'')
        elif "with the text" in p_lower:
            idx = p_lower.find("with the text") + len("with the text")
            text_to_draw = p_str[idx:].strip().strip('.:,"\'')
        elif "with text" in p_lower:
            idx = p_lower.find("with text") + len("with text")
            text_to_draw = p_str[idx:].strip().strip('.:,"\'')
        else:
            match = re.search(r'["\']([^"\']+)["\']', p_str)
            if match:
                text_to_draw = match.group(1).strip()

    # 4. If glyph is missing but we have mask and text, auto-generate glyph
    if glyph_img is None and mask_img is not None and text_to_draw:
        try:
            m_np = np.array(mask_img.convert('L'))
            contours, _ = cv2.findContours(m_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                _, g_np = get_glyph_pos(m_np, text_to_draw, m_np.shape[1], m_np.shape[0])
                g_np = (1 - g_np.astype('uint8')) * 255
                glyph_img = Image.fromarray(g_np[:, :, 0])
        except Exception as e:
            print(f"[*] Notice auto-generating glyph: {e}")

    # 5. If mask is STILL missing:
    if mask_img is None:
        if text_to_draw:
            font_to_use = selffont
            try:
                target_text_w = min(tgt_width * 0.75, 500)
                font_size = max(int(target_text_w / max(len(text_to_draw), 1)), 24)
                font_to_use = selffont.font_variant(size=font_size)
            except Exception:
                font_to_use = selffont

            # Auto-create centered text layout & mask
            glyph_canvas = Image.new("RGB", (tgt_width, tgt_height), (255, 255, 255))
            d = ImageDraw.Draw(glyph_canvas)
            bbox = d.textbbox((0, 0), text_to_draw, font=font_to_use)
            tw, th = max(bbox[2] - bbox[0], 10), max(bbox[3] - bbox[1], 10)
            tx = max((tgt_width - tw) // 2, 10)
            ty = max((tgt_height - th) // 2, 10)
            d.text((tx, ty), text_to_draw, font=font_to_use, fill=(0, 0, 0))
            glyph_img = glyph_canvas

            mask_canvas = Image.new("L", (tgt_width, tgt_height), 0)
            dm = ImageDraw.Draw(mask_canvas)
            pad_x = max(int(tw * 0.08), 20)
            pad_y = max(int(th * 0.15), 15)
            dm.rectangle([max(tx - pad_x, 0), max(ty - pad_y, 0), min(tx + tw + pad_x, tgt_width), min(ty + th + pad_y, tgt_height)], fill=255)
            mask_img = mask_canvas
        else:
            # Full inpainting/generative mode: entire image is masked
            mask_img = Image.new("L", (tgt_width, tgt_height), 255)

    # 6. If glyph is STILL missing:
    if glyph_img is None:
        # Default blank white canvas (no constraint on generation)
        glyph_img = Image.new("RGB", (tgt_width, tgt_height), (255, 255, 255))

    hint = mask_img.resize((tgt_width, tgt_height)).convert('RGB')
    img = img.resize((tgt_width, tgt_height))
    condition_img = glyph_img.resize((tgt_width, tgt_height)).convert('RGB')
    hint = np.array(hint) / 255
    condition_img = np.array(condition_img)
    condition_img = (255 - condition_img) / 255
    condition_img = [condition_img, hint, img]
    position_delta = [0, 0]
    condition = Condition(
                    condition_type='word_fill',
                    condition=condition_img,
                    position_delta=position_delta,
                )
    seed_val = int(seed) if seed is not None else 42
    generator.manual_seed(seed_val)
    gen_func = generate_fill_low_RAM if USE_LOW_VRAM else generate_fill
    res = gen_func(
        pipe,
        prompt=prompt,
        conditions=[condition],
        height=tgt_height,
        width=tgt_width,
        generator=generator,
        model_config=config.get("model", {}),
        default_lora=True,
    )
    return res.images[0]

def mode_select_change_func(mode_select):
    if mode_select:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)
    
def image_upload_func(image):
    return gr.update(value=image)

single_examples = [
    ["lepto college of education, the written materials on the picture: LESOTHO , COLLEGE OF , RE BONA LESELI LESEL , EDUCATION .", Image.open("assets/hint_imgs.jpg"), Image.open("assets/hint_imgs_word.png"), Image.open("assets/hint.png"), 42],
    ["keda group logo, that reads KDG , 科达股份 , 证券代码：600986 , 数字营销领军集团 .", Image.open("assets/hint_imgs1.jpg"), Image.open("assets/hint_imgs_word1.png"), Image.open("assets/hint1.png"), 42],
    ["chinese calligraphy font with the word 'love' written in it, that reads 精神食粮 .", Image.open("assets/hint_imgs2.jpg"), Image.open("assets/hint_imgs_word2.png"), Image.open("assets/hint2.png"), 42],
]
# Create the Gradio Blocks interface
with gr.Blocks() as demo:
    gr.Markdown("# Image Generation with FLUX-Text")
    gr.Markdown("Generate images using FLUX-Text with a lightweight Condition Injection LoRA.")

    with gr.Row():
        with gr.Column():
            with gr.Row():
                img = gr.Image(label="Image", type="pil")  # 上传图像文件
            with gr.Row():
                glyph_img = gr.Image(label="glyph Image", type="pil")  # 上传图像文件
                mask_img = gr.Image(label="mask Image", type="pil")  # 上传图像文件

            with gr.Row():
                mode_select = gr.Checkbox(label='manual edit', info='manual checkbox')

            with gr.Row(visible=False) as edit_image:
                with gr.Column():
                    edit_mask = gr.ImageEditor(type="numpy", interactive=True)
                with gr.Column():
                    edit_text = gr.Textbox(label="Input text")
                    edit_btn = gr.Button("Generate glyph and mask")

            prompt = gr.Textbox(label="Prompt")
            seed = gr.Number(label="Seed", value=42)
            single_generate_btn = gr.Button("Generate Image")
        with gr.Column():
            single_output_image = gr.Image(label="Generated Image")

    # Add examples for Single Condition Generation
    gr.Examples(
        examples=single_examples,
        inputs=[prompt, img, glyph_img, mask_img, seed],
        outputs=single_output_image,
        fn=generate_image_func,
        cache_examples=False,  # 缓存示例结果以加快加载速度
        label="Examples"
    )

    # Link the buttons to the functions
    img.upload(
        image_upload_func,
        inputs=[img],
        outputs=[edit_mask],
    )
    single_generate_btn.click(
        generate_image_func,
        inputs=[prompt, img, glyph_img, mask_img, seed, edit_mask, edit_text],
        outputs=single_output_image
    )
    mode_select.change(
        mode_select_change_func,
        inputs=[mode_select],
        outputs=[edit_image],
    )
    edit_btn.click(
        update_mask_func,
        inputs=[edit_mask, edit_text],
        outputs=[mask_img, glyph_img, prompt],
    )

if __name__ == '__main__':
    args = parse_args()
    cfg_server = APP_CONFIG.get("server", {})
    cfg_model = APP_CONFIG.get("model", {})

    font_path = cfg_model.get("font_path", './font/Arial_Unicode.ttf')
    if not os.path.exists(font_path):
        candidates = ['./font/Arial_Unicode.ttf', 'font/Arial_Unicode.ttf']
        for c in candidates:
            if os.path.exists(c):
                font_path = c
                break
    try:
        selffont = ImageFont.truetype(font_path, size=60)
    except Exception as e:
        print(f"[!] Notice loading font '{font_path}': {e}. Using default font.")
        selffont = ImageFont.load_default()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    generator = torch.Generator(device="cuda")
    to_tensor = T.ToTensor()

    blip_id = cfg_model.get("blip_path", "Salesforce/blip2-opt-2.7b")
    processor = AutoProcessor.from_pretrained(blip_id)
    blipmodel = Blip2ForConditionalGeneration.from_pretrained(blip_id, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    USE_LOW_VRAM = args.low_vram or (torch.cuda.is_available() and vram_gb < 36)
    if USE_LOW_VRAM:
        print(f"[*] Low-VRAM CPU offloading mode active (GPU VRAM: {vram_gb:.1f}GB <= 36GB).")
        blipmodel.to(torch.float16)
    else:
        blipmodel.to(device, torch.float16)

    pipe, trainable_model = init_pipeline(args, config, is_low_vram=USE_LOW_VRAM)

    server_host = cfg_server.get("host", "0.0.0.0")
    server_port = int(cfg_server.get("port", 6681))
    server_share = cfg_server.get("share", False)

    # Launch the Gradio app
    print(f"\n[+] Launching Gradio UI on http://{server_host}:{server_port}...")
    demo.queue().launch(server_name=server_host, server_port=server_port, share=server_share)