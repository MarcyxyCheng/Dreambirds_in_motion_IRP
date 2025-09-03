#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
corrected_cub_to_op9_png.py
Render per-frame OP-9 PNGs derived from CUB-15 using a corrected, hand-verified mapping
- Uses the corrected index mapping; ignores potentially wrong YAML labels
- OP-9 joints: 0=Nose, 1=Neck, 2=RShoulder, 3=RElbow, 4=RWrist, 5=LShoulder, 6=LElbow, 7=LWrist, 8=MidHip
- Optional white/black background
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw


def resample_sequence(seq, T_out):
    """Linear interpolation resampling along time."""
    T_in = seq.shape[0]
    if not T_out or T_out == T_in:
        return seq
    xs = np.linspace(0, T_in - 1, num=T_out)
    x0 = np.floor(xs).astype(int)
    x1 = np.clip(x0 + 1, 0, T_in - 1)
    w = xs - x0
    return (1 - w)[:, None, None] * seq[x0] + w[:, None, None] * seq[x1]


def fit_to_canvas(points_xy, W, H, margin=0.05):
    """Scale and center points into the canvas while preserving aspect ratio."""
    xs, ys = points_xy[:, 0], points_xy[:, 1]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    bw, bh = max(1e-6, xmax - xmin), max(1e-6, ymax - ymin)
    sx = (1 - 2 * margin) * W / bw
    sy = (1 - 2 * margin) * H / bh
    s = min(sx, sy)
    cx_src, cy_src = (xmin + xmax) / 2, (ymin + ymax) / 2
    cx_dst, cy_dst = W / 2, H / 2
    out = points_xy.copy()
    out[:, 0] = (points_xy[:, 0] - cx_src) * s + cx_dst
    out[:, 1] = (points_xy[:, 1] - cy_src) * s + cy_dst
    return out


def clamp_canvas(points_xy, W, H):
    """Clamp points to the canvas bounds."""
    out = points_xy.copy()
    out[:, 0] = np.clip(out[:, 0], 0, W - 1)
    out[:, 1] = np.clip(out[:, 1], 0, H - 1)
    return out


def derive_op9_from_cub15(frame_cub_xy, swap_lr=False, shoulder_offset=0.04, elbow_ratio=0.55):
    """
    Derive OP-9 joints from one CUB-15 frame using a corrected index mapping.

    Args:
        frame_cub_xy: (K, 2) CUB pixel coordinates for one frame
        swap_lr:      if True, swap left_wing / right_wing
        shoulder_offset: shoulder offset magnitude relative to canvas size (0..1)
        elbow_ratio:  position of elbow along the shoulder→wrist segment (0..1)

    Returns:
        op9_xy: (9, 2) in the order:
                [Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist, MidHip]
    """
    # Corrected mapping (indices refer to CUB array)
    CORRECTED_MAPPING = {
        "beak": 7,        # index that actually corresponds to the beak
        "neck": 5,        # index that actually corresponds to the throat/neck
        "left_wing": 10,  # index that actually corresponds to left wing
        "right_wing": 11, # index that actually corresponds to right wing
        "back": 0,        # back
        "belly": 14,      # belly
    }

    beak = frame_cub_xy[CORRECTED_MAPPING["beak"]]
    neck = frame_cub_xy[CORRECTED_MAPPING["neck"]]
    back = frame_cub_xy[CORRECTED_MAPPING["back"]]
    belly = frame_cub_xy[CORRECTED_MAPPING["belly"]]

    # Wing points (optionally swap)
    if swap_lr:
        rwing = frame_cub_xy[CORRECTED_MAPPING["left_wing"]]
        lwing = frame_cub_xy[CORRECTED_MAPPING["right_wing"]]
    else:
        rwing = frame_cub_xy[CORRECTED_MAPPING["right_wing"]]
        lwing = frame_cub_xy[CORRECTED_MAPPING["left_wing"]]

    # MidHip: midpoint of back and belly
    original_midhip = 0.5 * back + 0.5 * belly

    # Shorten neck→midhip distance to half (consistent with an OP-5 style heuristic)
    neck_to_midhip_vector = original_midhip - neck
    shortened_midhip = neck + 0.5 * neck_to_midhip_vector

    # Shoulders: offset from neck in directions of the wings
    canvas_size = 512  # used only to turn a relative ratio into pixels
    offset_dist = shoulder_offset * canvas_size

    # Right shoulder from neck toward right wing
    neck_to_rwing = rwing - neck
    rwing_dist = np.linalg.norm(neck_to_rwing) + 1e-6
    rwing_unit = neck_to_rwing / rwing_dist
    rshoulder = neck + rwing_unit * offset_dist

    # Left shoulder from neck toward left wing
    neck_to_lwing = lwing - neck
    lwing_dist = np.linalg.norm(neck_to_lwing) + 1e-6
    lwing_unit = neck_to_lwing / lwing_dist
    lshoulder = neck + lwing_unit * offset_dist

    # Elbows along shoulder→wing
    relbow = rshoulder + elbow_ratio * (rwing - rshoulder)
    lelbow = lshoulder + elbow_ratio * (lwing - lshoulder)

    # Wrists at wing tips
    rwrist = rwing
    lwrist = lwing

    # Assemble OP-9
    op9 = np.zeros((9, 2), dtype=np.float32)
    op9[0] = beak          # Nose
    op9[1] = neck          # Neck
    op9[2] = rshoulder     # RShoulder
    op9[3] = relbow        # RElbow
    op9[4] = rwrist        # RWrist
    op9[5] = lshoulder     # LShoulder
    op9[6] = lelbow        # LElbow
    op9[7] = lwrist        # LWrist
    op9[8] = shortened_midhip  # MidHip

    return op9


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Derive OP-9 from CUB-15 and render PNG frames (using a corrected mapping)."
    )

    # I/O
    ap.add_argument("--npy", required=True, help=".npy skeleton sequence (T×K×2 or T×K×3)")
    ap.add_argument("--out_dir", default="op9_corrected_png", help="Output directory for OP-9 PNGs")

    # Canvas
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)

    # Temporal
    ap.add_argument("--frames", type=int, default=None, help="Resampled frame count")

    # Coordinate handling
    ap.add_argument("--coord_mode", choices=["neg1to1", "unit", "pixels"], default="unit",
                    help="Input coordinate mode")
    ap.add_argument("--src_w", type=int, default=512, help="Original width if coord_mode=pixels")
    ap.add_argument("--src_h", type=int, default=512, help="Original height if coord_mode=pixels")

    # Scaling and centering
    ap.add_argument("--fit_to_canvas", dest="fit_to_canvas", action="store_true")
    ap.add_argument("--no-fit_to_canvas", dest="fit_to_canvas", action="store_false")
    ap.set_defaults(fit_to_canvas=True)
    ap.add_argument("--fit_global", action="store_true", help="Fit the whole sequence once (uniform scale/center)")
    ap.add_argument("--margin", type=float, default=0.05)

    # OP-9 specific controls
    ap.add_argument("--y_up_op9", action="store_true", help="Apply Y-up to image-coordinate flip to OP-9 only")
    ap.add_argument("--swap_lr_op9", action="store_true", help="Swap left/right wings for OP-9 only")
    ap.add_argument("--shoulder_offset", type=float, default=0.04, help="Shoulder offset ratio (relative to 512)")
    ap.add_argument("--elbow_ratio", type=float, default=0.55, help="Elbow position along shoulder→wrist")

    # Drawing
    ap.add_argument("--line_w", type=int, default=6, help="Line width")
    ap.add_argument("--point_r", type=int, default=6, help="Joint dot radius")
    ap.add_argument("--points_only", action="store_true", help="Draw points only (debug)")
    ap.add_argument("--bg_color", choices=["white", "black"], default="black", help="Background color")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load sequence
    seq = np.load(args.npy)
    assert seq.ndim == 3 and seq.shape[-1] in (2, 3), "Expected (T,K,2|3)"
    T, K, D = seq.shape
    has_vis = (D == 3)
    print(f"Loaded sequence: {T} frames, {K} keypoints, {'with' if has_vis else 'without'} visibility")

    if K != 15:
        print(f"Warning: expected 15 keypoints, got {K}")

    # Map coordinates into pixel domain
    if args.coord_mode == "neg1to1":
        seq_xy = (seq[..., :2] + 1.0) / 2.0
        seq_xy[..., 0] *= args.width
        seq_xy[..., 1] *= args.height
    elif args.coord_mode == "unit":
        seq_xy = seq[..., :2].copy()
        seq_xy[..., 0] *= args.width
        seq_xy[..., 1] *= args.height
    else:  # pixels
        seq_xy = seq[..., :2].copy()
        seq_xy[..., 0] = seq_xy[..., 0] / max(1, args.src_w) * args.width
        seq_xy[..., 1] = seq_xy[..., 1] / max(1, args.src_h) * args.height

    # Resample
    seq_xy = resample_sequence(seq_xy, args.frames)
    vis = resample_sequence(seq[..., 2:], args.frames)[..., 0] if has_vis else None
    T_out = seq_xy.shape[0]
    print(f"Output frames: {T_out}")

    # Global fit parameters (if enabled)
    if args.fit_global:
        all_pts = seq_xy.reshape(-1, 2)
        xmin, ymin = all_pts[:, 0].min(), all_pts[:, 1].min()
        xmax, ymax = all_pts[:, 0].max(), all_pts[:, 1].max()
        bw, bh = max(1e-6, xmax - xmin), max(1e-6, ymax - ymin)
        sx = (1 - 2 * args.margin) * args.width / bw
        sy = (1 - 2 * args.margin) * args.height / bh
        s = min(sx, sy)
        cx_src, cy_src = (xmin + xmax) / 2, (ymin + ymax) / 2
        cx_dst, cy_dst = args.width / 2, args.height / 2
        print(f"Global scale: {s:.3f}, center: ({cx_src:.1f},{cy_src:.1f}) -> ({cx_dst:.1f},{cy_dst:.1f})")

    # OP-9 edge list
    op9_edges = [
        (8, 1),  # midhip-neck
        (1, 0),  # neck-nose
        (1, 2),  # neck-rshoulder
        (2, 3),  # rshoulder-relbow
        (3, 4),  # relbow-rwrist
        (1, 5),  # neck-lshoulder
        (5, 6),  # lshoulder-lelbow
        (6, 7),  # lelbow-lwrist
    ]

    # Colors (cycled)
    base_colors = [
        (255, 0, 0),
        (255, 128, 0),
        (255, 255, 0),
        (0, 255, 0),
        (0, 255, 255),
        (0, 128, 255),
        (0, 0, 255),
        (255, 0, 255),
    ]

    print("Rendering OP-9 frames with corrected mapping...")
    for t in range(T_out):
        pts_cub = seq_xy[t]  # (K, 2)

        try:
            # Derive OP-9 using the corrected mapping
            op9 = derive_op9_from_cub15(
                pts_cub,
                swap_lr=args.swap_lr_op9,
                shoulder_offset=args.shoulder_offset,
                elbow_ratio=args.elbow_ratio
            )

            # Fit/center
            if args.fit_global:
                op9_draw = op9.copy()
                op9_draw[:, 0] = (op9[:, 0] - cx_src) * s + cx_dst
                op9_draw[:, 1] = (op9[:, 1] - cy_src) * s + cy_dst
            elif args.fit_to_canvas:
                op9_draw = fit_to_canvas(op9, args.width, args.height, margin=args.margin)
            else:
                op9_draw = op9.copy()

            # Y-up flip (OP-9 only)
            if args.y_up_op9:
                op9_draw[:, 1] = args.height - 1 - op9_draw[:, 1]

            op9_draw = clamp_canvas(op9_draw, args.width, args.height)

            # Render
            img = Image.new("RGB", (args.width, args.height), args.bg_color)
            draw = ImageDraw.Draw(img)

            # Edges
            if not args.points_only:
                for edge_idx, (a, b) in enumerate(op9_edges):
                    color = base_colors[edge_idx % len(base_colors)]
                    pt_a = tuple(op9_draw[a].astype(int))
                    pt_b = tuple(op9_draw[b].astype(int))
                    draw.line([pt_a, pt_b], fill=color, width=args.line_w)

            # Points
            for i in range(9):
                x, y = int(op9_draw[i, 0]), int(op9_draw[i, 1])
                r = args.point_r
                # outer white
                draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255))
                # colored inner core
                inner_color = base_colors[i % len(base_colors)]
                draw.ellipse([x - r // 2, y - r // 2, x + r // 2, y + r // 2], fill=inner_color)

            # Save
            img.save(os.path.join(args.out_dir, f"frame_{t:04d}.png"))

            if (t + 1) % 10 == 0 or t == T_out - 1:
                print(f"  Finished {t + 1}/{T_out} frames")

        except Exception as e:
            print(f"Warning: failed to process frame {t}: {e}")
            continue

    print(f"Wrote {T_out} frames to {args.out_dir}")
    print("Note: corrected index mapping used; YAML labels are ignored.")
    print("Corrected mapping summary: beak=7, neck=5, left_wing=10, right_wing=11, back=0, belly=14")


if __name__ == "__main__":
    main()
