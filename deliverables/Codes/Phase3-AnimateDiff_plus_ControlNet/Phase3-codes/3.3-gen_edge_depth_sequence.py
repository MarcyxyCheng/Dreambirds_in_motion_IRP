#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3-3: generate edges_depth_sequence
Generate multi-frame edge/depth sequences (default 64 frames) from a single image
for ControlNet conditioning.

Dependencies:
  pip install opencv-python pillow numpy transformers torch torchvision

Example:
  python gen_edges_depth_sequence.py \
    --image dreambird.jpg \
    --out_root reportcontrolnet/BirdLand \
    --frames 64 \
    --size 512 \
    --device cpu \
    --canny_t1 100 --canny_t2 200
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
from transformers import pipeline


def load_image_rgb(path: str) -> Image.Image:
    """Load an image as RGB PIL.Image."""
    img = Image.open(path).convert("RGB")
    return img


def resize_square(img_pil: Image.Image, size: int) -> Image.Image:
    """Resize to a square (size x size) using bicubic resampling."""
    if size is None:
        return img_pil
    return img_pil.resize((size, size), Image.BICUBIC)


def make_edges(
    img_pil: Image.Image,
    t1: int = 100,
    t2: int = 200,
    aperture: int = 3,
    out_size: int = 512
) -> Image.Image:
    """
    Create a Canny edge map (grayscale) from a PIL RGB image and resize to out_size.
    """
    arr = np.array(img_pil)  # RGB uint8
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, t1, t2, apertureSize=aperture, L2gradient=True)
    # Save as 1-channel (L). If 3-channel is needed, stack the array.
    edges_img = Image.fromarray(edges, mode="L").resize((out_size, out_size), Image.BILINEAR)
    return edges_img


def depth_pipeline(device_str: str):
    """
    Create a transformers depth-estimation pipeline.
    Use device=0 for GPU if available and device_str == 'cuda', otherwise CPU.
    """
    if device_str.lower() == "cuda" and torch.cuda.is_available():
        device = 0
    else:
        device = "cpu"
    return pipeline(
        task="depth-estimation",
        model="LiheYoung/depth-anything-large-hf",
        device=device
    )


def make_depth(img_path: str, out_size: int = 512, device_str: str = "cpu") -> Image.Image:
    """
    Run depth estimation on the input image path.
    Output is normalized to 0–255 and resized to out_size as a grayscale PNG-ready PIL image.
    """
    pipe = depth_pipeline(device_str)
    out = pipe(img_path)
    depth_img = out.get("depth") or out.get("predicted_depth")  # PIL (mode 'F' or 'L')
    d_np = np.array(depth_img, dtype=np.float32)
    m = float(d_np.max())
    if m <= 0:
        # Avoid division by zero
        d_np = np.zeros_like(d_np, dtype=np.float32)
        m = 1.0
    d_u8 = (255.0 * (d_np / m)).astype(np.uint8)
    depth_png = Image.fromarray(d_u8, mode="L").resize((out_size, out_size), Image.BILINEAR)
    return depth_png


def replicate_frames(img: Image.Image, out_dir: Path, frames: int, prefix: str = "frames"):
    """
    Save the same image replicated N times as frames0000.png ... frames00NN.png.
    This script does not create temporal variation by itself.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(frames):
        img.save(out_dir / f"{prefix}{i:04d}.png")


def main():
    ap = argparse.ArgumentParser(description="Generate multi-frame edge/depth sequences from a single image.")
    ap.add_argument("--image", required=True, help="Path to input image, e.g., dreambird.jpg")
    ap.add_argument("--out_root", required=True, help="Output root directory, e.g., reportcontrolnet/BirdLand")
    ap.add_argument("--frames", type=int, default=64, help="Number of frames to replicate (default: 64)")
    ap.add_argument("--size", type=int, default=512, help="Output resolution (square; default: 512)")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                    help="Device for depth inference (default: cpu)")
    ap.add_argument("--canny_t1", type=int, default=100, help="Canny threshold 1 (default: 100)")
    ap.add_argument("--canny_t2", type=int, default=200, help="Canny threshold 2 (default: 200)")
    ap.add_argument("--canny_aperture", type=int, default=3, choices=[3, 5, 7],
                    help="Canny aperture size (default: 3)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    edge_dir = out_root / "edge"
    depth_dir = out_root / "depth"

    # Load and resize image (for edges)
    img = load_image_rgb(args.image)
    img = resize_square(img, args.size)

    # Generate edge sequence
    edge_img = make_edges(
        img,
        t1=args.canny_t1,
        t2=args.canny_t2,
        aperture=args.canny_aperture,
        out_size=args.size
    )
    replicate_frames(edge_img, edge_dir, args.frames, prefix="frames")

    # Generate depth sequence (Depth Anything)
    depth_img = make_depth(args.image, out_size=args.size, device_str=args.device)
    replicate_frames(depth_img, depth_dir, args.frames, prefix="frames")

    print("Done.")
    print(f"Edge frames directory:  {edge_dir}")
    print(f"Depth frames directory: {depth_dir}")
    print(f"Frames per sequence:    {args.frames}")
    print("File naming: frames0000.png ... frames00NN.png")


if __name__ == "__main__":
    main()
