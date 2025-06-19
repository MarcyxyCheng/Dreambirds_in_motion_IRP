#!/usr/bin/env python
# Canny edge + Depth
"""
run_mybird_combo.py ────────────────────────────────────────────────────────────
Animate "mybird.jpg" into a 16‑frame GIF with **AnimateDiff + two ControlNets**
(Canny edges + monocular depth).  This variant uses the official
`AnimateDiffControlNetPipeline`, whose `__call__` signature takes
`conditioning_frames` **instead of** the usual `image/control_image` args.

Prerequisites
-------------
1. Generate the guidance maps once (512×512):
     $ python gen_canny.py   # → canny_map.png
     $ python run_depth.py   # → depth_map.png
2. A CUDA build of PyTorch plus `diffusers>=0.26`, `transformers`, `accelerate`.

Run
---
     CUDA_VISIBLE_DEVICES=0 python run_mybird_combo.py
"""

from diffusers import (
    AnimateDiffControlNetPipeline,
    MotionAdapter,
    ControlNetModel,
    DDIMScheduler,
)
from diffusers.utils import export_to_gif
from PIL import Image
import torch, os, sys

# ── Helpers ────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"🖥️  Device: {device} | dtype: {dtype}")

def here(*p):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *p)

# ── 1.  Load motion module and ControlNets ──────────────────────────────────
print("🔄 Loading models …")
motion_adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=dtype
)
cn_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=dtype)
cn_depth = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=dtype)

# ── 2.  Build the pipeline — **note the class!** ───────────────────────────
pipe = AnimateDiffControlNetPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    motion_adapter=motion_adapter,
    controlnet=[cn_canny, cn_depth],          # multi‑ControlNet
    torch_dtype=dtype,
).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# ── 3.  Inputs ─────────────────────────────────────────────────────────────
try:
    ref_img = Image.open(here("mybird.jpg")).convert("RGB")  # just for prompt crafting
except FileNotFoundError:
    sys.exit("❌  mybird.jpg not found next to the script.")

canny = Image.open(here("canny_map.png")).convert("RGB").resize((512, 512))
depth = Image.open(here("depth_map.png")).convert("L").resize((512, 512))

num_frames = 16
# each element in the outer list = frame‑t conditionings;
# inner list order ↔ order of ControlNets passed above
# structure: 2 (CNs) × 16 (frames)
conditioning_frames = [
    [canny] * num_frames,  # Canny map repeated for every frame
    [depth] * num_frames,  # Depth map repeated likewise
]

# ── 4.  Prompts ────────────────────────────────────────────────────────────
prompt = (
    "Sun conure parrot from the reference photo, bright orange head, yellow body, "
    "green wing edges, perched on the same grey cloth. "
    "The bird **starts to turn its head**, feathers gently ruffle. "
    "Cinematic camera slowly pans and orbits, subtle body sway, dynamic natural lighting, "
    "shallow depth of field, crisp 4K."
)
negative_prompt = (
    "blurry, distorted anatomy, extra wings, split beak, low detail, low resolution, watermark"
)

# ── 5.  Generate ───────────────────────────────────────────────────────────
print("🎬  Generating clip …")
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=512,
    height=512,
    num_frames=num_frames,
    conditioning_frames=conditioning_frames,
    controlnet_conditioning_scale=[0.6, 1.0],  # Canny weaker, Depth stronger
    guidance_scale=7.0,
    num_inference_steps=25,
    generator=torch.manual_seed(42),
)
frames = result.frames[0]
print(f"✅  Generated {len(frames)} frames")

# ── 6.  Save ───────────────────────────────────────────────────────────────
out_path = here("mybird_control_combo.gif")
export_to_gif(frames, out_path)
print(f"🎉  GIF saved → {out_path}")
