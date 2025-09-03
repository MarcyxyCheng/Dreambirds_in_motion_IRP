#!/usr/bin/env python
"""
Phase 3-5: run_mybird_combo_pose.py ──────────────────────────────────────────────────────
Animate a bird photograph into a short clip **driven by three ControlNets**:
  • Canny edges   (global silhouette)
  • Depth map     (basic volume)
  • OpenPose      (per‑frame skeleton / keypoints)

Usage (example) ──────────────────────────────────────────────────────────────
    python run_mybird_combo_pose.py \
        --ckpt SG161222/Realistic_Vision_V5.1_noVAE \
        --canny_dir demo/canny_png \
        --depth_dir demo/depth_png \
        --pose_dir  demo/pose_png  \
        --frames 16 --fps 8 \
        --prompt "A red‑crowned crane flapping wings, cinematic lighting" \
        --out_dir demo/out

Prerequisites ────────────────────────────────────────────────────────────────
• diffusers >= 0.28.2, transformers, accelerate, safetensors
• CUDA‑enabled PyTorch (FP16 recommended)  
• Pre‑generated PNG sequences in the three directories above, with filenames
  like frame_0000.png … frame_00XX.png (same dimensions as --width/--height).

The script will
  1. load MotionAdapter + 3 ControlNets
  2. assemble conditioning_frames  =  [Canny_seq, Depth_seq, Pose_seq]
  3. run AnimateDiffControlNetPipeline
  4. save GIF + individual PNG frames to --out_dir
"""

import argparse, glob, os, pathlib, sys

import torch
from PIL import Image
from diffusers import (
    AnimateDiffControlNetPipeline,  # diffusers >= 0.28.2
    MotionAdapter,
    ControlNetModel,
    DDIMScheduler,
)
from diffusers.utils import export_to_gif

# ────────────────────────── Helper functions ──────────────────────────────

def gather_frame_paths(folder: str):
    """Return sorted list of frame_*.png paths inside *folder*"""
    paths = sorted(glob.glob(os.path.join(folder, "frame_*.png")))
    if not paths:
        raise FileNotFoundError(f"No frame_*.png files in {folder!r}")
    return paths


def load_and_resize(path: str, size: tuple[int, int], mode: str = "RGB") -> Image.Image:
    """Open *path*, convert to mode, resize to *size* (nearest)."""
    return Image.open(path).convert(mode).resize(size, Image.BICUBIC)


# ───────────────────────────────  CLI  ─────────────────────────────────────

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--ckpt", default="SG161222/Realistic_Vision_V5.1_noVAE",
                    help="Base SD‑1.5 checkpoint or local dir")
parser.add_argument("--canny_dir", required=True,  help="Dir with Canny PNG sequence")
parser.add_argument("--depth_dir", required=True,  help="Dir with Depth PNG sequence")
parser.add_argument("--pose_dir",  required=True,  help="Dir with Pose  PNG sequence")
parser.add_argument("--frames", type=int, default=16, help="#frames to render")
parser.add_argument("--fps",     type=int, default=8,  help="Output GIF FPS (metadata)")
parser.add_argument("--width",   type=int, default=512)
parser.add_argument("--height",  type=int, default=512)
parser.add_argument("--prompt",  required=True)
parser.add_argument("--negative_prompt", default=(
    "blurry, distorted anatomy, low detail, watermark, text, jpeg artifacts"
))
parser.add_argument("--out_dir", default="demo/out", help="Where to write GIF & frames")
parser.add_argument("--seed", type=int, default=42)
# ControlNet scales
parser.add_argument("--scale_canny", type=float, default=0.5)
parser.add_argument("--scale_depth", type=float, default=0.8)
parser.add_argument("--scale_pose",  type=float, default=1.2)
args = parser.parse_args()

out_dir = pathlib.Path(args.out_dir).expanduser()
out_dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────  Device  ────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
print(f" Device: {device} | dtype: {dtype}  | seed: {args.seed}")

# ────────────────────────  Load models  ────────────────────────────────────
print("Loading models …")

motion_adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=dtype
)

cn_canny = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=dtype
)
cn_depth = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=dtype
)
cn_pose  = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", torch_dtype=dtype
)

pipe = AnimateDiffControlNetPipeline.from_pretrained(
    args.ckpt,
    motion_adapter=motion_adapter,
    controlnet=[cn_canny, cn_depth, cn_pose],
    torch_dtype=dtype,
).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()


if torch.cuda.is_available():
    try:
        pipe.enable_model_cpu_offload()
    except Exception as e:
        print(f"Could not enable model CPU offload: {e}. Continuing without it.")
else:
    pipe.to("cpu")

# ──────────────────  Prepare conditioning frames  ─────────────────────────
print("Preparing conditioning sequences …")

# Load *one* Canny + Depth image, repeat for all frames (saves VRAM)
canny_first = load_and_resize(gather_frame_paths(args.canny_dir)[0], (args.width, args.height))
depth_first = load_and_resize(gather_frame_paths(args.depth_dir)[0], (args.width, args.height), mode="L")

# Pose needs one image per frame
pose_paths = gather_frame_paths(args.pose_dir)[: args.frames]
if len(pose_paths) < args.frames:
    raise ValueError(f"Only {len(pose_paths)} pose frames found, but --frames = {args.frames}")
pose_imgs = [load_and_resize(p, (args.width, args.height)) for p in pose_paths]

conditioning_frames = [
    [canny_first] * args.frames,       # Canny  (RGB)
    [depth_first] * args.frames,       # Depth  (L)
    pose_imgs,                        # Pose   (RGB per‑frame)
]

# ───────────────────────────  Generate  ────────────────────────────────────
print("Generating clip …")

result = pipe(
    prompt=args.prompt,
    negative_prompt=args.negative_prompt,
    width=args.width,
    height=args.height,
    num_frames=args.frames,
    conditioning_frames=conditioning_frames,
    controlnet_conditioning_scale=[args.scale_canny, args.scale_depth, args.scale_pose],
    guidance_scale=7.0,
    num_inference_steps=25,
    generator=torch.manual_seed(args.seed),
)
frames = result.frames[0]
print(f"Generated {len(frames)} frames")

# ────────────────────────────  Save  ──────────────────────────────────────
gif_path = out_dir / "mybird_control_combo33333.gif"
export_to_gif(frames, gif_path, fps=args.fps)
print(f" GIF saved → {gif_path}")

# Optional: also save individual PNG frames
for idx, frame in enumerate(frames):
    frame.save(out_dir / f"frame_{idx:04d}.png")

print("Done.")
