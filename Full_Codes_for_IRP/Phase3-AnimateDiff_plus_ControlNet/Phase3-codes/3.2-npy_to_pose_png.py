#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3-2: npy_to_pose_png
Render per-frame PNGs in a ControlNet/OpenPose style from a .npy skeleton sequence.

Supports:
- CUB-15 edges (same as your 2.6 visualization)
- YAML-defined custom edges (CUB-style keypoints + connections)
- CUB-15 → OpenPose-18 (standard human topology):
  * --op18_from_yaml  Automatically build a mapping from YAML keypoint names (smart mapping)
  * --op18_map        Manual 18-index mapping (indices may repeat)

Example (OpenPose18 + auto-mapping + 16 frames):
  python 5.2-npy_to_pose_png.py \
    --npy seq.npy --yaml skeleton.yaml \
    --out_dir pose_op18 \
    --width 512 --height 512 \
    --coord_mode pixels --src_w 512 --src_h 512 \
    --frames 16 --y_up --fit_global \
    --edge_mode openpose18 --op18_from_yaml
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw

# optional yaml
try:
    import yaml
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

# ---------- CUB-15 connections (same as your 2.6) ----------
CUB_CONNECTIONS = [
    (0,1),(1,2),(1,3),(1,4),(2,5),       # head block
    (1,6),(6,7),(7,8),(8,9),             # spine
    (7,10),(7,11),                       # wings
    (8,12),                              # tail
    (9,13),(9,14),                       # legs
]

# ---------- OpenPose18 standard edges ----------
OPENPOSE18_EDGES = [
    (0,1), (1,2), (2,3), (3,4),          # nose-neck-Rshoulder-Relbow-Rwrist
    (1,5), (5,6), (6,7),                 # neck-Lshoulder-Lelbow-Lwrist
    (1,8), (8,9), (9,10),                # neck-Rhip-Rknee-Rankle
    (1,11),(11,12),(12,13),              # neck-Lhip-Lknee-Lankle
    (0,14),(14,16), (0,15),(15,17)       # nose-REye-REar, nose-LEye-LEar
]

# ---------- utils ----------
def resample_sequence(seq, T_out):
    T_in = seq.shape[0]
    if not T_out or T_out == T_in:
        return seq
    xs = np.linspace(0, T_in - 1, num=T_out)
    x0 = np.floor(xs).astype(int)
    x1 = np.clip(x0 + 1, 0, T_in - 1)
    w = xs - x0
    return (1 - w)[:, None, None] * seq[x0] + w[:, None, None] * seq[x1]

def fit_to_canvas(points_xy, W, H, margin=0.05):
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
    out = points_xy.copy()
    out[:, 0] = np.clip(out[:, 0], 0, W - 1)
    out[:, 1] = np.clip(out[:, 1], 0, H - 1)
    return out

def load_yaml_edges(yaml_path):
    """
    Read a CUB-style skeleton.yaml with shape like:
      keypoints: [{id:1, name: back}, ...]  # 1-based indexing
      connections: rigid / flexible
    Returns (kp_names, edges, edge_types)
    """
    if not HAVE_YAML:
        raise RuntimeError("pyyaml is not installed; cannot read --yaml. Please `pip install pyyaml`.")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    kp_raw = data.get("keypoints")
    if not isinstance(kp_raw, list) or len(kp_raw) == 0:
        raise ValueError("YAML missing a non-empty keypoints list.")
    items = []
    for it in kp_raw:
        if not isinstance(it, dict) or "id" not in it or "name" not in it:
            raise ValueError(f"Cannot parse keypoint entry: {it}")
        items.append((int(it["id"]), str(it["name"])))
    items.sort(key=lambda x: x[0])
    kp_names = [n for (_i, n) in items]
    id1_to_idx0 = {i: i-1 for (i, _n) in items}

    conns = data.get("connections", {})
    rigid = conns.get("rigid", []) or []
    flex  = conns.get("flexible", []) or []

    def convert(pairs, etype):
        out = []
        for p in pairs:
            a_id, b_id = int(p[0]), int(p[1])
            if a_id not in id1_to_idx0 or b_id not in id1_to_idx0:
                raise KeyError(f"Connection references unknown keypoint id: {p}")
            out.append((id1_to_idx0[a_id], id1_to_idx0[b_id], etype))
        return out

    edges_all = convert(rigid, "rigid") + convert(flex, "flexible")

    seen = set()
    edges, types = [], []
    for a, b, et in edges_all:
        key = (a, b) if a <= b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        edges.append((a, b))
        types.append(et)
    return kp_names, edges, types

def create_smart_cub_to_openpose_mapping(cub_keypoints, cub_names=None):
    """
    Smart CUB → OpenPose-18 mapping.
    Mix of dropping unsuitable points and interpolating missing ones.

    Args:
        cub_keypoints: (T, K, 2) array of CUB points
        cub_names: list of CUB keypoint names (optional)

    Returns:
        openpose_points: (T, 18, 2) mapped points
        visibility:      (T, 18)   visibility array
    """
    T, K, _ = cub_keypoints.shape
    openpose_points = np.zeros((T, 18, 2))
    visibility = np.ones((T, 18))  # default visible

    meaningful_mappings = {
        # direct plausible mappings
        0: "beak",        # nose <- beak
        1: "nape",        # neck <- nape
        14: "right_eye",  # REye
        15: "left_eye",   # LEye
        # legs
        9: "right_leg",   # RKnee
        10: "right_leg",  # RAnkle (extended from knee)
        12: "left_leg",   # LKnee
        13: "left_leg",   # LAnkle (extended from knee)
    }

    if cub_names:
        name_to_idx = {name: i for i, name in enumerate(cub_names)}
        print(f"CUB keypoints: {cub_names}")
    else:
        # default CUB-15 names (adjust to your true ordering if needed)
        name_to_idx = {
            "back": 0, "beak": 1, "belly": 2, "breast": 3, "crown": 4,
            "forehead": 5, "left_eye": 6, "left_leg": 7, "left_wing": 8,
            "nape": 9, "right_eye": 10, "right_leg": 11, "right_wing": 12,
            "tail": 13, "throat": 14
        }
        print("Using default CUB-15 name-to-index mapping.")

    for t in range(T):
        frame_cub = cub_keypoints[t]

        # (1) direct mappings
        for op_idx, cub_name in meaningful_mappings.items():
            if cub_name in name_to_idx:
                cub_idx = name_to_idx[cub_name]
                openpose_points[t, op_idx] = frame_cub[cub_idx]
                # extend ankle a bit away from knee, if applicable
                if op_idx in [10, 13]:  # RAnkle, LAnkle
                    knee_idx = op_idx - 1  # paired knee index
                    if knee_idx in [9, 12]:
                        leg_direction = openpose_points[t, op_idx] - openpose_points[t, knee_idx]
                        n = np.linalg.norm(leg_direction)
                        if n > 1e-6:
                            leg_direction = leg_direction / n
                            openpose_points[t, op_idx] = openpose_points[t, op_idx] + leg_direction * 10
            else:
                visibility[t, op_idx] = 0.1  # low visibility if missing

        # (2) interpolate missing joints from available body landmarks
        neck_pos = frame_cub[name_to_idx["nape"]] if "nape" in name_to_idx else None
        breast_pos = (frame_cub[name_to_idx["breast"]]
                      if "breast" in name_to_idx else
                      (frame_cub[name_to_idx["throat"]] if "throat" in name_to_idx else None))
        belly_pos = frame_cub[name_to_idx["belly"]] if "belly" in name_to_idx else None

        # shoulders from neck and breast
        if (neck_pos is not None) and (breast_pos is not None):
            shoulder_center = neck_pos * 0.3 + breast_pos * 0.7
            body_scale = np.linalg.norm(breast_pos - neck_pos)
            shoulder_width = body_scale * 0.8
            openpose_points[t, 2] = shoulder_center + np.array([shoulder_width/2, 0])   # RShoulder
            openpose_points[t, 5] = shoulder_center + np.array([-shoulder_width/2, 0])  # LShoulder
        else:
            visibility[t, 2] = visibility[t, 5] = 0.2

        # elbows and wrists from wings
        if "right_wing" in name_to_idx and visibility[t, 2] > 0.5:
            right_wing = frame_cub[name_to_idx["right_wing"]]
            right_shoulder = openpose_points[t, 2]
            openpose_points[t, 3] = right_shoulder * 0.3 + right_wing * 0.7  # RElbow
            openpose_points[t, 4] = right_wing                               # RWrist
            visibility[t, 3] = visibility[t, 4] = 0.7
        else:
            visibility[t, 3] = visibility[t, 4] = 0.1

        if "left_wing" in name_to_idx and visibility[t, 5] > 0.5:
            left_wing = frame_cub[name_to_idx["left_wing"]]
            left_shoulder = openpose_points[t, 5]
            openpose_points[t, 6] = left_shoulder * 0.3 + left_wing * 0.7   # LElbow
            openpose_points[t, 7] = left_wing                               # LWrist
            visibility[t, 6] = visibility[t, 7] = 0.7
        else:
            visibility[t, 6] = visibility[t, 7] = 0.1

        # hips from belly
        if belly_pos is not None:
            hip_width = body_scale * 0.6 if 'body_scale' in locals() else 20
            openpose_points[t, 8]  = belly_pos + np.array([hip_width/2, 0])   # RHip
            openpose_points[t, 11] = belly_pos + np.array([-hip_width/2, 0])  # LHip
        else:
            visibility[t, 8] = visibility[t, 11] = 0.2

        # ears from eyes
        if visibility[t, 14] > 0.5 and visibility[t, 15] > 0.5:
            right_eye = openpose_points[t, 14]
            left_eye = openpose_points[t, 15]
            eye_distance = np.linalg.norm(right_eye - left_eye)
            if eye_distance > 1e-6:
                openpose_points[t, 16] = right_eye + np.array([eye_distance*0.4, 0])  # REar
                openpose_points[t, 17] = left_eye  + np.array([-eye_distance*0.4, 0]) # LEar
                visibility[t, 16] = visibility[t, 17] = 0.5
            else:
                visibility[t, 16] = visibility[t, 17] = 0.1
        else:
            visibility[t, 16] = visibility[t, 17] = 0.1

    return openpose_points, visibility

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--npy", required=True, help=".npy skeleton sequence (T×K×2 or T×K×3)")
    ap.add_argument("--yaml", default=None, help="Required for edge_mode=yaml or --op18_from_yaml")
    ap.add_argument("--out_dir", default="pose_png", help="Output directory")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--frames", type=int, default=None, help="Resampled frame count")
    ap.add_argument("--coord_mode", choices=["neg1to1", "unit", "pixels"], default="unit",
                    help="Input coordinate mode")
    ap.add_argument("--src_w", type=int, default=512, help="Original width if coord_mode=pixels")
    ap.add_argument("--src_h", type=int, default=512, help="Original height if coord_mode=pixels")
    ap.add_argument("--line_w_rigid", type=int, default=6)
    ap.add_argument("--line_w_flex",  type=int, default=5)
    ap.add_argument("--point_r", type=int, default=4)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--fit_to_canvas", dest="fit_to_canvas", action="store_true")
    ap.add_argument("--no-fit_to_canvas", dest="fit_to_canvas", action="store_false")
    ap.set_defaults(fit_to_canvas=True)
    ap.add_argument("--y_up", action="store_true", help="Input uses math coords (y up); flip for image coords before saving")
    ap.add_argument("--fit_global", action="store_true", help="Fit the whole sequence at once to reduce per-frame drift")
    ap.add_argument("--points_only", action="store_true", help="Draw points only (for debugging)")
    ap.add_argument("--idx_map", type=str, default=None,
                    help="Comma-separated integers to reorder input keypoints to the target edge order (length=K)")
    ap.add_argument("--edge_mode", choices=["cub","yaml","openpose18"], default="cub",
                    help="Where to get edges: CUB-15 / YAML / OpenPose-18")
    ap.add_argument("--op18_map", type=str, default=None,
                    help="Manual 18-index mapping (0-based) from OpenPose18 to input indices; repeats allowed")
    ap.add_argument("--op18_from_yaml", action="store_true",
                    help="Build OpenPose18 mapping from YAML keypoint names (smart mapping)")
    ap.add_argument("--vis_threshold", type=float, default=0.05,
                    help="Visibility threshold; points/edges below this will not be drawn")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load seq
    seq = np.load(args.npy)  # (T,K,2|3)
    assert seq.ndim == 3 and seq.shape[-1] in (2, 3), "Expected (T,K,2|3)"
    T, K, D = seq.shape
    has_vis = (D == 3)
    print(f"Loaded sequence: {T} frames, {K} keypoints, {'with' if has_vis else 'without'} visibility channel")

    # optional re-indexing
    if args.idx_map:
        mapping = [int(x.strip()) for x in args.idx_map.split(",") if x.strip() != ""]
        assert len(mapping) == K, f"idx_map length must be {K}, got {len(mapping)}"
        seq = seq[:, mapping, :]
        print(f"Applied index reordering: {mapping}")

    # choose edges & optional OP18 mapping
    EDGES = CUB_CONNECTIONS
    EDGE_TYPES = ["rigid"] * len(EDGES)
    vis = None

    if args.edge_mode == "yaml":
        if not args.yaml:
            raise ValueError("edge_mode=yaml requires --yaml")
        kp_names, yaml_edges, yaml_types = load_yaml_edges(args.yaml)
        EDGES = [(a, b) for (a, b) in yaml_edges if a < K and b < K]
        EDGE_TYPES = []
        ti = 0
        for (a, b) in yaml_edges:
            if a < K and b < K:
                t = yaml_types[ti] if ti < len(yaml_types) else "rigid"
                EDGE_TYPES.append(t)
            ti += 1
        print(f"Using YAML edges: {len(EDGES)} segments")

    elif args.edge_mode == "openpose18":
        # smart mapping or manual mapping
        if args.op18_from_yaml:
            if not args.yaml:
                raise ValueError("--op18_from_yaml requires --yaml")
            kp_names, _, _ = load_yaml_edges(args.yaml)
            print("Using smart CUB→OpenPose18 mapping...")

            # map coordinates to pixel space before mapping
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

            # apply smart mapping
            openpose_points, op_vis = create_smart_cub_to_openpose_mapping(seq_xy, kp_names)

            # re-pack to original format (with or without visibility)
            if has_vis:
                seq = np.concatenate([openpose_points, op_vis[..., None]], axis=-1)
            else:
                seq = openpose_points

        else:
            # manual mapping path (kept from your original)
            default_op18_map = [1,9,0,12,12,0,8,8,2,11,11,2,7,7,10,6,10,6]
            if args.op18_map:
                op18_map = [int(x.strip()) for x in args.op18_map.split(",")]
                assert len(op18_map) == 18, "op18_map must have 18 integers"
            else:
                op18_map = default_op18_map
            assert max(op18_map) < K, "op18_map index out of range"
            seq = seq[:, op18_map, :]
            print(f"Using manual mapping: {op18_map}")

        K = 18
        EDGES = OPENPOSE18_EDGES
        EDGE_TYPES = ["rigid"] * len(EDGES)
        print(f"Converted to OpenPose18 format: {K} keypoints")

    # map to pixel domain if not already done
    if args.edge_mode != "openpose18" or not args.op18_from_yaml:
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
    else:
        seq_xy = seq[..., :2]

    # resample
    seq_xy = resample_sequence(seq_xy, args.frames)
    T_out = seq_xy.shape[0]
    vis = resample_sequence(seq[..., 2:], args.frames)[..., 0] if has_vis else None
    print(f"Output frames: {T_out}")

    # global fit (optional)
    if args.fit_global:
        all_pts = seq_xy.reshape(-1, 2)
        xmin, ymin = all_pts[:, 0].min(), all_pts[:, 1].min()
        xmax, ymax = all_pts[:, 0].max(), all_pts[:, 1].max()
        bw, bh = max(1e-6, xmax-xmin), max(1e-6, ymax-ymin)
        sx = (1-2*args.margin)*args.width  / bw
        sy = (1-2*args.margin)*args.height / bh
        s = min(sx, sy)
        cx_src, cy_src = (xmin+xmax)/2, (ymin+ymax)/2
        cx_dst, cy_dst = args.width/2,  args.height/2
        print(f"Global scale: {s:.3f}, center: ({cx_src:.1f}, {cy_src:.1f}) -> ({cx_dst:.1f}, {cy_dst:.1f})")

    # loop colors
    base_colors = [
        (255, 0, 0), (255,128, 0), (255,255, 0),
        (0, 255, 0), (0, 255,255), (0,128,255),
        (0, 0,255), (255, 0,255)
    ]

    # render
    print("Rendering frames...")
    for t in range(T_out):
        pts = seq_xy[t]

        if args.fit_global:
            pts_draw = pts.copy()
            pts_draw[:,0] = (pts[:,0]-cx_src)*s + cx_dst
            pts_draw[:,1] = (pts[:,1]-cy_src)*s + cy_dst
        elif args.fit_to_canvas:
            pts_draw = fit_to_canvas(pts, args.width, args.height, margin=args.margin)
        else:
            pts_draw = pts

        if args.y_up:
            pts_draw[:,1] = args.height - 1 - pts_draw[:,1]

        pts_draw = clamp_canvas(pts_draw, args.width, args.height)

        img = Image.new("RGB", (args.width, args.height), "black")
        draw = ImageDraw.Draw(img)

        def kp_visible(i):
            if vis is None:
                return True
            return vis[t, i] > args.vis_threshold

        # draw edges
        if not args.points_only:
            for ei, (a, b) in enumerate(EDGES):
                if a >= K or b >= K:
                    continue
                if not (kp_visible(a) and kp_visible(b)):
                    continue
                color = base_colors[ei % len(base_colors)]
                if args.edge_mode == "yaml" and ei < len(EDGE_TYPES):
                    lw = args.line_w_rigid if EDGE_TYPES[ei] == "rigid" else args.line_w_flex
                else:
                    lw = args.line_w_rigid
                xa, ya = int(pts_draw[a,0]), int(pts_draw[a,1])
                xb, yb = int(pts_draw[b,0]), int(pts_draw[b,1])
                draw.line([(xa,ya),(xb,yb)], fill=color, width=lw)

        # draw points
        for i in range(K):
            if not kp_visible(i):
                continue
            x, y = int(pts_draw[i,0]), int(pts_draw[i,1])
            r = args.point_r
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(255,255,255))

        img.save(os.path.join(args.out_dir, f"frame_{t:04d}.png"))

        if (t + 1) % 10 == 0 or t == T_out - 1:
            print(f"  Finished {t+1}/{T_out} frames")

    print(f"Saved {T_out} frames to {args.out_dir}")
    if args.edge_mode == "openpose18":
        print("OpenPose18 tip: validate Pose-only first, then gradually add Depth/Canny.")
        if args.op18_from_yaml:
            print("Smart mapping used; interpolated joints have adjusted visibilities for better appearance.")

if __name__ == "__main__":
    main()
