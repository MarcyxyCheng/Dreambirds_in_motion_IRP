#!/usr/bin/env python3
"""
Phase 3-1 smooth_pose 
Skeleton visualization version 
Adapted for CUB-15 point MDM output format, with skeleton visualization utilities.
"""

import argparse
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt  # (not strictly required unless you add plots)
import cv2
from PIL import Image
import os
import json


def cub15_to_skeleton_image(keypoints, img_size=512, title="CUB-15 Skeleton"):
    """
    Convert a single CUB-15 keypoint set to a skeleton visualization image.

    Args:
        keypoints: np.ndarray of shape (15, 3) with (x, y, visibility)
        img_size: output square image size (pixels)
        title: title text to draw at the top-left
    Returns:
        PIL.Image RGB image
    """
    # Create canvas
    pose_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Extract coordinates and visibility
    coords = keypoints[:, :2]      # (15, 2)
    visibility = keypoints[:, 2]   # (15,)

    # Normalize coordinates into [0,1] (with safe margins)
    coord_min, coord_max = coords.min(axis=0), coords.max(axis=0)
    coord_range = coord_max - coord_min

    if np.any(coord_range > 0):
        coords_norm = (coords - coord_min) / coord_range
        # Keep margins to avoid touching borders
        coords_norm[:, 0] = coords_norm[:, 0] * 0.7 + 0.15  # X: 0.15-0.85
        coords_norm[:, 1] = coords_norm[:, 1] * 0.7 + 0.15  # Y: 0.15-0.85
    else:
        coords_norm = np.full_like(coords, 0.5)

    # To pixel coordinates
    points_2d = (coords_norm * img_size).astype(int)

    # CUB-15 connections (indices must match your CUB-15 definition)
    cub_connections = [
        # main spine chain: beak -> head center -> neck -> body front -> body back -> tail
        (0, 1), (1, 6), (6, 7), (7, 8), (8, 12),

        # head structure
        (1, 2), (2, 5),      # head contour
        (1, 3), (1, 4),      # eyes region

        # wings
        (7, 10), (7, 11),

        # legs
        (8, 13), (8, 14),

        # lower body
        (8, 9),
    ]

    # Colors (BGR for OpenCV)
    colors = {
        'head': (255, 100, 100),       # red-ish
        'body': (100, 255, 100),       # green-ish
        'wings': (100, 100, 255),      # blue-ish
        'tail': (255, 255, 100),       # yellow-ish
        'legs': (255, 100, 255),       # magenta-ish
        'connection': (255, 255, 255)  # white
    }

    # Draw connections
    for start_idx, end_idx in cub_connections:
        if (start_idx < len(points_2d) and end_idx < len(points_2d) and
                visibility[start_idx] > 0.5 and visibility[end_idx] > 0.5):

            start_point = tuple(points_2d[start_idx])
            end_point = tuple(points_2d[end_idx])

            if (start_idx, end_idx) in [(0, 1), (1, 6), (6, 7), (7, 8), (8, 12)]:
                color = colors['connection']
                thickness = 4
            elif (start_idx, end_idx) in [(7, 10), (7, 11)]:
                color = colors['wings']
                thickness = 3
            elif (start_idx, end_idx) in [(8, 13), (8, 14)]:
                color = colors['legs']
                thickness = 3
            else:
                color = colors['connection']
                thickness = 2

            cv2.line(pose_img, start_point, end_point, color, thickness)

    # Draw keypoints
    for i, (point, vis) in enumerate(zip(points_2d, visibility)):
        if vis > 0.5:
            if i <= 5:             # head points (0-5)
                color = colors['head']
                radius = 6
            elif 6 <= i <= 9:      # body points (6-9)
                color = colors['body']
                radius = 7
            elif i in [10, 11]:    # wings (10-11)
                color = colors['wings']
                radius = 6
            elif i == 12:          # tail (12)
                color = colors['tail']
                radius = 5
            else:                   # legs (13-14)
                color = colors['legs']
                radius = 5

            cv2.circle(pose_img, tuple(point), radius, color, -1)
            cv2.circle(pose_img, tuple(point), radius, (255, 255, 255), 1)
            cv2.putText(pose_img, str(i), (point[0] + 8, point[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Title
    cv2.putText(pose_img, title, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Legend
    legend_y = img_size - 120
    cv2.putText(pose_img, "CUB-15 Parts:", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    legend_items = [
        ("0-5: Head", colors['head']),
        ("6-9: Body", colors['body']),
        ("10-11: Wings", colors['wings']),
        ("12: Tail", colors['tail']),
        ("13-14: Legs", colors['legs'])
    ]

    for j, (text, color) in enumerate(legend_items):
        y_pos = legend_y + 15 + j * 15
        cv2.circle(pose_img, (15, y_pos - 3), 4, color, -1)
        cv2.putText(pose_img, text, (25, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return Image.fromarray(pose_img)


def create_skeleton_comparison(original_seq, smoothed_seq, output_dir, num_samples=8):
    """
    Create side-by-side static skeleton comparison images for sampled frames.

    Args:
        original_seq: np.ndarray (T, 15, 3)
        smoothed_seq: np.ndarray (T, 15, 3)
        output_dir: base directory for visualization
        num_samples: number of frames to sample uniformly across the sequence
    Returns:
        vis_dir: directory path where images are saved
    """
    print("Creating skeleton comparison visualization...")

    vis_dir = os.path.join(output_dir, 'skeleton_visualization')
    os.makedirs(vis_dir, exist_ok=True)

    total_frames = len(original_seq)
    sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    print(f"Sampled frame indices: {sample_indices}")

    for i, frame_idx in enumerate(sample_indices):
        print(f"Processing frame {frame_idx + 1}/{total_frames} ({i + 1}/{num_samples})")

        original_img = cub15_to_skeleton_image(
            original_seq[frame_idx],
            title=f"Original Frame {frame_idx}"
        )
        smoothed_img = cub15_to_skeleton_image(
            smoothed_seq[frame_idx],
            title=f"Smoothed Frame {frame_idx}"
        )

        comparison = Image.new('RGB', (original_img.width * 2, original_img.height))
        comparison.paste(original_img, (0, 0))
        comparison.paste(smoothed_img, (original_img.width, 0))

        comparison_path = os.path.join(vis_dir, f'comparison_frame_{frame_idx:03d}.png')
        comparison.save(comparison_path)

        original_img.save(os.path.join(vis_dir, f'original_frame_{frame_idx:03d}.png'))
        smoothed_img.save(os.path.join(vis_dir, f'smoothed_frame_{frame_idx:03d}.png'))

    print(f"Skeleton comparison saved to: {vis_dir}")
    return vis_dir


def create_animation_comparison(original_seq, smoothed_seq, output_dir, fps=10):
    """
    Create side-by-side skeleton animations (GIF/MP4) comparing original vs smoothed.

    Args:
        original_seq: np.ndarray (T, 15, 3)
        smoothed_seq: np.ndarray (T, 15, 3)
        output_dir: base directory for visualization
        fps: frames per second for animations
    """
    try:
        import imageio

        print("Creating skeleton animation comparison...")

        vis_dir = os.path.join(output_dir, 'skeleton_visualization')
        os.makedirs(vis_dir, exist_ok=True)

        original_frames = []
        smoothed_frames = []
        comparison_frames = []

        step = max(1, len(original_seq) // 32)  # cap to ~32 frames
        frame_indices = list(range(0, len(original_seq), step))
        print(f"Animation frames: {len(frame_indices)} (step: {step})")

        for i, frame_idx in enumerate(frame_indices):
            if i % 5 == 0:
                print(f"Generating animation frame {i + 1}/{len(frame_indices)}")

            try:
                original_img = cub15_to_skeleton_image(
                    original_seq[frame_idx],
                    title=f"Original Frame {frame_idx}"
                )
                smoothed_img = cub15_to_skeleton_image(
                    smoothed_seq[frame_idx],
                    title=f"Smoothed Frame {frame_idx}"
                )

                comparison = Image.new('RGB', (original_img.width * 2, original_img.height))
                comparison.paste(original_img, (0, 0))
                comparison.paste(smoothed_img, (original_img.width, 0))

                original_frames.append(np.array(original_img))
                smoothed_frames.append(np.array(smoothed_img))
                comparison_frames.append(np.array(comparison))

            except Exception as frame_error:
                print(f"Skipping frame {frame_idx}: {frame_error}")
                continue

        if not original_frames:
            print("No frames were created for animation.")
            return

        # Try GIF
        try:
            original_gif = os.path.join(vis_dir, 'original_animation.gif')
            smoothed_gif = os.path.join(vis_dir, 'smoothed_animation.gif')
            comparison_gif = os.path.join(vis_dir, 'comparison_animation.gif')

            print("Saving GIF animations...")
            imageio.mimsave(original_gif, original_frames, fps=fps, loop=0)
            imageio.mimsave(smoothed_gif, smoothed_frames, fps=fps, loop=0)
            imageio.mimsave(comparison_gif, comparison_frames, fps=fps, loop=0)

            print("GIF animations saved:")
            print(f"  Original:   {original_gif}")
            print(f"  Smoothed:   {smoothed_gif}")
            print(f"  Comparison: {comparison_gif}")

        except Exception as gif_error:
            print(f"GIF save failed: {gif_error}")

            # Try MP4
            try:
                print("Trying MP4 animations...")
                original_mp4 = os.path.join(vis_dir, 'original_animation.mp4')
                smoothed_mp4 = os.path.join(vis_dir, 'smoothed_animation.mp4')
                comparison_mp4 = os.path.join(vis_dir, 'comparison_animation.mp4')

                imageio.mimsave(original_mp4, original_frames, fps=fps)
                imageio.mimsave(smoothed_mp4, smoothed_frames, fps=fps)
                imageio.mimsave(comparison_mp4, comparison_frames, fps=fps)

                print("MP4 animations saved:")
                print(f"  Original:   {original_mp4}")
                print(f"  Smoothed:   {smoothed_mp4}")
                print(f"  Comparison: {comparison_mp4}")

            except Exception as mp4_error:
                print(f"MP4 save failed: {mp4_error}")

                # Fallback: save individual frames
                print("Fallback: saving individual frames to disk...")
                frames_dir = os.path.join(vis_dir, 'animation_frames')
                os.makedirs(frames_dir, exist_ok=True)

                for i, (orig, smooth, comp) in enumerate(zip(original_frames, smoothed_frames, comparison_frames)):
                    Image.fromarray(orig).save(os.path.join(frames_dir, f'original_{i:03d}.png'))
                    Image.fromarray(smooth).save(os.path.join(frames_dir, f'smoothed_{i:03d}.png'))
                    Image.fromarray(comp).save(os.path.join(frames_dir, f'comparison_{i:03d}.png'))

                print(f"Saved individual frames to: {frames_dir}")
                print(f"You can assemble a video via ffmpeg, e.g.:")
                print(f"  ffmpeg -r {fps} -i {frames_dir}/comparison_%03d.png -y comparison_animation.mp4")

    except ImportError:
        print("Skipping animation creation (imageio is not installed).")
        print("Install with: pip install imageio imageio[ffmpeg]")

        print("Creating additional dense static comparisons...")
        vis_dir = os.path.join(output_dir, 'skeleton_visualization')
        os.makedirs(vis_dir, exist_ok=True)

        total_frames = len(original_seq)
        dense_indices = np.linspace(0, total_frames - 1, 16, dtype=int)

        for frame_idx in dense_indices:
            try:
                original_img = cub15_to_skeleton_image(
                    original_seq[frame_idx],
                    title=f"Original Frame {frame_idx}"
                )
                smoothed_img = cub15_to_skeleton_image(
                    smoothed_seq[frame_idx],
                    title=f"Smoothed Frame {frame_idx}"
                )
                comparison = Image.new('RGB', (original_img.width * 2, original_img.height))
                comparison.paste(original_img, (0, 0))
                comparison.paste(smoothed_img, (original_img.width, 0))

                comparison_path = os.path.join(vis_dir, f'dense_comparison_{frame_idx:03d}.png')
                comparison.save(comparison_path)

            except Exception as frame_error:
                print(f"Skipping static frame {frame_idx}: {frame_error}")
                continue

        print("Dense static comparisons created.")


def analyze_smoothing_effect(original_seq, smoothed_seq):
    """
    Print simple statistics on smoothing effect.

    Args:
        original_seq: np.ndarray (T, 15, 3)
        smoothed_seq: np.ndarray (T, 15, 3)
    """
    print("\nSmoothing effect analysis:")

    original_coords = original_seq[:, :, :2]
    smoothed_coords = smoothed_seq[:, :, :2]
    coord_diff = np.abs(smoothed_coords - original_coords)

    joint_names = [
        "Beak", "Head Center", "Back of Head", "Left Eye", "Right Eye", "Crown",
        "Neck", "Body Front", "Body Back", "Body Bottom",
        "Left Wing", "Right Wing", "Tail", "Left Leg", "Right Leg"
    ]

    print("  Mean absolute change per joint (x,y averaged):")
    for i, name in enumerate(joint_names):
        joint_change = np.mean(coord_diff[:, i, :])
        print(f"    {i:2d}. {name:12s}: {joint_change:.4f}")

    frame_changes = np.mean(coord_diff, axis=(1, 2))

    print("\n  Temporal change stats (average over joints and xy):")
    print(f"    Overall mean change: {np.mean(coord_diff):.4f}")
    print(f"    Max change:          {np.max(coord_diff):.4f}")
    print(f"    Std dev:             {np.std(coord_diff):.4f}")
    print(f"    Frame with max change: {np.argmax(frame_changes)} (value: {np.max(frame_changes):.4f})")
    print(f"    Frame with min change: {np.argmin(frame_changes)} (value: {np.min(frame_changes):.4f})")


def main():
    parser = argparse.ArgumentParser(description='Smooth CUB-15 skeleton sequences and visualize.')
    parser.add_argument('--input', required=True,
                        help='Path to MDM generated pose sequence (e.g., generated_pose_seq_fixed.npy)')
    parser.add_argument('--output', required=True,
                        help='Path to save the smoothed pose (e.g., pose_seq_smooth.npy)')
    parser.add_argument('--window_length', type=int, default=9,
                        help='Savitzky-Golay window length (must be odd)')
    parser.add_argument('--polyorder', type=int, default=3,
                        help='Savitzky-Golay polynomial order')
    parser.add_argument('--smooth_coords_only', action='store_true',
                        help='Only smooth x,y coordinates and keep visibility unchanged')

    # Data cleaning options
    parser.add_argument('--remove_first_frame', action='store_true',
                        help='Remove the first frame if it appears corrupted')
    parser.add_argument('--remove_frames', type=str, default='',
                        help='Comma-separated specific frames to remove, e.g., "0,1,63"')

    # Visualization options
    parser.add_argument('--visualize', action='store_true',
                        help='Create skeleton visualization images')
    parser.add_argument('--vis_samples', type=int, default=8,
                        help='Number of frames to visualize (uniformly sampled)')
    parser.add_argument('--create_animation', action='store_true',
                        help='Create animation comparison (requires imageio)')
    parser.add_argument('--animation_fps', type=int, default=10,
                        help='Animation frames per second')

    args = parser.parse_args()

    # Load sequence
    print(f"Loading MDM sequence: {args.input}")
    data = np.load(args.input)  # expected shape: (T, 15, 3)

    print(f"  Original data shape: {data.shape}")
    print(f"  Data value range: [{data.min():.3f}, {data.max():.3f}]")

    # Shape checks
    if len(data.shape) != 3:
        raise ValueError(f"Expected 3D data (T, J, C), got {data.shape}")

    T, J, C = data.shape
    if J != 15 or C != 3:
        raise ValueError(f"Expected (T, 15, 3) format, got {data.shape}")

    # Frame removal
    frames_to_remove = []

    if args.remove_first_frame:
        frames_to_remove.append(0)
        print("Will remove the first frame (frame 0).")

    if args.remove_frames:
        try:
            specific_frames = [int(x.strip()) for x in args.remove_frames.split(',') if x.strip()]
            frames_to_remove.extend(specific_frames)
            print(f"Will remove specific frames: {specific_frames}")
        except ValueError:
            print(f"Invalid frame indices string: {args.remove_frames}")

    if frames_to_remove:
        frames_to_remove = sorted(set(frames_to_remove))
        valid_frames = [i for i in range(T) if i not in frames_to_remove]

        if len(valid_frames) == 0:
            raise ValueError("No frames left after removal.")

        print(f"Frame removal summary:")
        print(f"  Removed frames: {frames_to_remove}")
        print(f"  Kept frames: {len(valid_frames)}/{T}")

        data = data[valid_frames]
        T = len(valid_frames)
        print(f"  Shape after removal: {data.shape}")

    # Ensure odd window length and valid relative to T
    if args.window_length % 2 == 0:
        args.window_length += 1
        print(f"Adjusted window_length to be odd: {args.window_length}")

    if args.window_length > T:
        args.window_length = min(T, 5) if T >= 5 else 3
        print(f"Adjusted window_length to: {args.window_length}")

    if args.polyorder >= args.window_length:
        args.polyorder = args.window_length - 1
        print(f"Adjusted polyorder to: {args.polyorder}")

    print(f"Smoothing parameters: window_length={args.window_length}, polyorder={args.polyorder}")

    # Smoothing
    if args.smooth_coords_only:
        print("Smoothing only x,y coordinates...")

        coords = data[:, :, :2]       # (T, J, 2)
        visibility = data[:, :, 2:3]  # (T, J, 1)

        print(f"  Coord value range: [{coords.min():.3f}, {coords.max():.3f}]")
        print(f"  Visibility value range: [{visibility.min():.3f}, {visibility.max():.3f}]")

        coords_smooth = savgol_filter(
            coords,
            window_length=args.window_length,
            polyorder=args.polyorder,
            axis=0
        )
        smoothed = np.concatenate([coords_smooth, visibility], axis=2)

        print(f"  Coord value range after smoothing: [{coords_smooth.min():.3f}, {coords_smooth.max():.3f}]")
    else:
        print("Smoothing all channels (x, y, visibility)...")
        smoothed = savgol_filter(
            data,
            window_length=args.window_length,
            polyorder=args.polyorder,
            axis=0
        )
        # Clamp visibility to [0,1]
        smoothed[:, :, 2] = np.clip(smoothed[:, :, 2], 0.0, 1.0)

    print("Smoothing done.")
    print(f"  Smoothed data shape: {smoothed.shape}")
    print(f"  Smoothed data value range: [{smoothed.min():.3f}, {smoothed.max():.3f}]")

    coord_diff = np.mean(np.abs(smoothed[:, :, :2] - data[:, :, :2]))
    print(f"  Mean absolute coordinate change: {coord_diff:.4f}")

    # Analysis
    analyze_smoothing_effect(data, smoothed)

    # Save results
    np.save(args.output, smoothed)
    print(f"Saved smoothed sequence to: {args.output}")

    # Visualization
    if args.visualize:
        output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else '.'

        vis_dir = create_skeleton_comparison(
            data, smoothed, output_dir,
            num_samples=args.vis_samples
        )

        if args.create_animation:
            create_animation_comparison(
                data, smoothed, output_dir,
                fps=args.animation_fps
            )

        print("\nVisualization completed.")
        print(f"  Static comparisons: {vis_dir}")
        if args.create_animation:
            print(f"  Animation files are inside: {vis_dir}")

    # Save processing info JSON
    processing_info = {
        'original_shape': list(np.load(args.input).shape),  # reload to record original shape
        'frames_removed': frames_to_remove if frames_to_remove else [],
        'processed_shape': list(data.shape),
        'smoothed_shape': list(smoothed.shape),
        'original_range': [float(data.min()), float(data.max())],
        'smoothed_range': [float(smoothed.min()), float(smoothed.max())],
        'coord_change': float(coord_diff),
        'window_length': args.window_length,
        'polyorder': args.polyorder,
        'smooth_coords_only': args.smooth_coords_only,
        'visualization_created': args.visualize,
        'remove_first_frame': args.remove_first_frame,
        'remove_frames': args.remove_frames
    }

    info_file = args.output.replace('.npy', '_processing_info.json')
    with open(info_file, 'w') as f:
        json.dump(processing_info, f, indent=2)
    print(f"Saved processing info to: {info_file}")

    # If frames were removed, print a summary
    if frames_to_remove:
        print("\nFrame removal summary:")
        print(f"  Original T: {np.load(args.input).shape[0]}")
        print(f"  Removed:    {len(frames_to_remove)}")
        print(f"  Final T:    {smoothed.shape[0]}")
        print(f"  Removed IDs:{frames_to_remove}")
        if args.remove_first_frame:
            print("Note: consider inspecting the MDM generation step to see why frame 0 looked corrupted.")


if __name__ == '__main__':
    main()
