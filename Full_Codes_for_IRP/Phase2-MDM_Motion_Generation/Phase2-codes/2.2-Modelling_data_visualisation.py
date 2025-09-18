#!/usr/bin/env python
"""
Phase 2-2: CUB-15 Skeleton Animation Generator
Specifically for generating animations of CUB-15 bird skeletons
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

class CUB15AnimationGenerator:
    """CUB-15 Skeleton Animation Generator"""
    
    def __init__(self):
        # CUB-15 joint configuration
        self.joint_names = [
            'beak', 'crown', 'forehead', 'left_eye', 'right_eye',      # 0-4: Head
            'throat', 'nape', 'back', 'breast', 'belly',               # 5-9: Torso
            'left_wing', 'right_wing', 'tail',                         # 10-12: Wings & tail
            'left_leg', 'right_leg'                                    # 13-14: Legs
        ]
        
        # Skeleton connection pairs
        self.connections = [
            # Main spine chain
            (0, 1), (1, 6), (6, 7), (7, 8),    # beak→crown→nape→back→breast
            # Head structure
            (1, 2), (2, 5),                    # crown→forehead→throat
            (1, 3), (1, 4),                    # crown→eyes
            # Torso connections
            (8, 9),                             # breast→belly
            (7, 12),                            # back→tail
            # Wing connections
            (7, 10), (7, 11),                  # back→wings
            # Leg connections
            (9, 13), (9, 14),                  # belly→legs
        ]
        
        # Visualization colors
        self.joint_colors = [
            # Head (0-5)
            'red', 'red', 'orange', 'blue', 'blue', 'orange',
            # Torso (6-9)
            'darkred', 'darkred', 'darkgreen', 'green',
            # Wings & tail (10-12)
            'blue', 'red', 'purple',
            # Legs (13-14)
            'brown', 'brown'
        ]
        
        self.connection_colors = [
            # Spine chain
            'red', 'red', 'red', 'red',
            # Head structure
            'orange', 'orange',
            'lightblue', 'lightblue',
            # Torso
            'green', 'purple',
            # Wings
            'blue', 'red',
            # Legs
            'brown', 'brown'
        ]

    def load_dataset(self, file_path):
        """Load dataset from .npy file"""
        if not Path(file_path).exists():
            print(f"File not found: {file_path}")
            return None
        
        dataset = np.load(file_path)
        print(f"Loaded data: {file_path}")
        print(f"   Shape: {dataset.shape}")
        
        return dataset

    def plot_skeleton_frame(self, pose, ax, title=""):
        """Plot a single skeleton frame"""
        
        ax.clear()
        
        # Draw connections
        for i, (start, end) in enumerate(self.connections):
            if pose[start, 2] > 0.3 and pose[end, 2] > 0.3:
                ax.plot([pose[start, 0], pose[end, 0]],
                        [pose[start, 1], pose[end, 1]], 
                        color=self.connection_colors[i],
                        linewidth=2.5, alpha=0.8)
        
        # Draw keypoints
        for j, (x, y, vis) in enumerate(pose):
            if vis > 0.3:
                size = 100 * vis
                ax.scatter(x, y, c=self.joint_colors[j], s=size,
                           alpha=0.9, edgecolors='black', linewidth=1,
                           zorder=5)
        
        # Add ground reference line
        ax.axhline(y=-0.4, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.0, 2.0)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title)

    def create_skeleton_animation(self, sequence, seq_idx=0, fps=8, max_frames=64):
        """Create skeleton-only animation"""
        
        if len(sequence) > max_frames:
            sequence = sequence[:max_frames]
            print(f"Animation limited to first {max_frames} frames")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def animate(frame):
            self.plot_skeleton_frame(
                sequence[frame], ax,
                title=f'Sequence {seq_idx+1} - Frame {frame+1}/{len(sequence)}'
            )
        
        ani = animation.FuncAnimation(fig, animate, frames=len(sequence),
                                      interval=1000//fps, repeat=True)
        
        gif_path = f"cub15_skeleton_{seq_idx+1}.gif"
        print(f"Saving skeleton animation: {gif_path}...")
        try:
            ani.save(gif_path, writer='pillow', fps=fps)
            print("Animation saved successfully")
        except Exception as e:
            print(f"Failed to save animation: {e}")
        
        plt.show()
        return ani

    def create_trajectory_animation(self, sequence, seq_idx=0, fps=8, max_frames=64):
        """Create animation with skeleton and trajectory views"""
        
        if len(sequence) > max_frames:
            sequence = sequence[:max_frames]
            print(f"Animation limited to first {max_frames} frames")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            # Left: skeleton
            self.plot_skeleton_frame(
                sequence[frame], ax1,
                title=f'Sequence {seq_idx+1} - Frame {frame+1}/{len(sequence)}'
            )
            
            # Right: trajectory
            frames_so_far = frame + 1
            body_x = sequence[:frames_so_far, 9, 0]  # belly x
            body_y = sequence[:frames_so_far, 9, 1]  # belly y
            ax2.plot(body_x, body_y, 'g-', linewidth=3, alpha=0.8, label='Body Path')
            
            # Wing paths
            lw_x = sequence[:frames_so_far, 10, 0]
            lw_y = sequence[:frames_so_far, 10, 1]
            rw_x = sequence[:frames_so_far, 11, 0]
            rw_y = sequence[:frames_so_far, 11, 1]
            ax2.plot(lw_x, lw_y, 'b-', linewidth=2, alpha=0.6, label='Left Wing')
            ax2.plot(rw_x, rw_y, 'r-', linewidth=2, alpha=0.6, label='Right Wing')
            
            # Current positions
            ax2.scatter(body_x[-1], body_y[-1], c='green', s=150, zorder=5, marker='o', edgecolors='black')
            ax2.scatter(lw_x[-1], lw_y[-1], c='blue', s=100, zorder=5, marker='^', edgecolors='black')
            ax2.scatter(rw_x[-1], rw_y[-1], c='red', s=100, zorder=5, marker='^', edgecolors='black')
            
            # Ground line
            ax2.axhline(y=-0.4, color='gray', linestyle='--', alpha=0.5, label='Ground')
            
            ax2.set_xlim(-2.0, 2.0)
            ax2.set_ylim(-1.0, 2.0)
            ax2.set_aspect('equal')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_title('Trajectory')
        
        ani = animation.FuncAnimation(fig, animate, frames=len(sequence),
                                      interval=1000//fps, repeat=True)
        
        gif_path = f"cub15_trajectory_{seq_idx+1}.gif"
        print(f"Saving trajectory animation: {gif_path}...")
        try:
            ani.save(gif_path, writer='pillow', fps=fps)
            print("Animation saved successfully")
        except Exception as e:
            print(f"Failed to save animation: {e}")
        
        plt.show()
        return ani

    def create_multi_view_animation(self, sequence, seq_idx=0, fps=8, max_frames=64):
        """Create multi-view animation"""
        
        if len(sequence) > max_frames:
            sequence = sequence[:max_frames]
            print(f"Animation limited to first {max_frames} frames")
        
        fig = plt.figure(figsize=(20, 10))
        ax1 = plt.subplot(2, 3, 1)  # front view
        ax2 = plt.subplot(2, 3, 2)  # side view
        ax3 = plt.subplot(2, 3, 3)  # top view
        ax4 = plt.subplot(2, 3, (4, 6))  # trajectory
        
        def animate(frame):
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            
            pose = sequence[frame]
            frames_so_far = frame + 1
            
            # Front view (X-Y)
            self._plot_2d_view(pose, ax1, 'x', 'y', 'Front View (X-Y)')
            
            # Side view (Z-Y)
            self._plot_2d_view(pose, ax2, 'z', 'y', 'Side View (Z-Y)', use_z=False)
            
            # Top view (X-Z)
            self._plot_2d_view(pose, ax3, 'x', 'z', 'Top View (X-Z)', use_z=False)
            
            # Body trajectory in ax4
            bx = sequence[:frames_so_far, 9, 0]
            by = sequence[:frames_so_far, 9, 1]
            ax4.plot(bx, by, 'g-', linewidth=3, alpha=0.8, label='Body Path')
            lwx = sequence[:frames_so_far, 10, 0]
            lwy = sequence[:frames_so_far, 10, 1]
            rwx = sequence[:frames_so_far, 11, 0]
            rwy = sequence[:frames_so_far, 11, 1]
            ax4.plot(lwx, lwy, 'b-', linewidth=2, alpha=0.6, label='Left Wing')
            ax4.plot(rwx, rwy, 'r-', linewidth=2, alpha=0.6, label='Right Wing')
            ax4.scatter(bx[-1], by[-1], c='green', s=150, zorder=5)
            
            ax4.set_xlim(-2.0, 2.0)
            ax4.set_ylim(-1.0, 2.0)
            ax4.set_aspect('equal')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            ax4.set_title('Trajectory')
            
            fig.suptitle(f'Sequence {seq_idx+1} Multi-View Animation - Frame {frame+1}/{len(sequence)}', fontsize=16)
        
        ani = animation.FuncAnimation(fig, animate, frames=len(sequence),
                                      interval=1000//fps, repeat=True)
        
        gif_path = f"cub15_multiview_{seq_idx+1}.gif"
        print(f"Saving multi-view animation: {gif_path}...")
        try:
            ani.save(gif_path, writer='pillow', fps=fps)
            print("Animation saved successfully")
        except Exception as e:
            print(f"Failed to save animation: {e}")
        
        plt.show()
        return ani

    def _plot_2d_view(self, pose, ax, axis1, axis2, title, use_z=True):
        """Plot a 2D projection of the skeleton"""
        
        axis_map = {'x': 0, 'y': 1, 'z': 2 if use_z else 0}
        coord1 = axis_map[axis1]
        coord2 = axis_map[axis2]
        
        for i, (start, end) in enumerate(self.connections):
            if pose[start, 2] > 0.3 and pose[end, 2] > 0.3:
                ax.plot([pose[start, coord1], pose[end, coord1]],
                        [pose[start, coord2], pose[end, coord2]],
                        color=self.connection_colors[i],
                        linewidth=2, alpha=0.8)
        
        for j, point in enumerate(pose):
            if point[2] > 0.3:
                ax.scatter(point[coord1], point[coord2],
                           c=self.joint_colors[j], s=80,
                           alpha=0.9, edgecolors='black', linewidth=1,
                           zorder=5)
        
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.0, 2.0)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel(axis1.upper())
        ax.set_ylabel(axis2.upper())

    def batch_create_animations(self, dataset, animation_type='skeleton', fps=8, max_sequences=5):
        """Batch create animations"""
        
        print(f"Batch creating {animation_type} animations...")
        print(f"   Number of sequences: {min(max_sequences, len(dataset))}")
        print(f"   Frame rate: {fps} FPS")
        
        created = []
        for i in range(min(max_sequences, len(dataset))):
            print(f"\nProcessing sequence {i+1}/{min(max_sequences, len(dataset))}...")
            seq = dataset[i]
            if animation_type == 'skeleton':
                ani = self.create_skeleton_animation(seq, i, fps)
            elif animation_type == 'trajectory':
                ani = self.create_trajectory_animation(seq, i, fps)
            elif animation_type == 'multiview':
                ani = self.create_multi_view_animation(seq, i, fps)
            else:
                print(f"Unknown animation type: {animation_type}")
                continue
            created.append(ani)
            plt.close('all')
        
        print(f"\nBatch animation creation complete!")
        return created


def main():
    """Main function"""
    
    print("CUB-15 Skeleton Animation Generator")
    print("=" * 40)
    
    animator = CUB15AnimationGenerator()
    
    data_file = input("Enter dataset file path (e.g., cub15_complete_train.npy): ").strip()
    if not data_file:
        data_file = "cub15_complete_train.npy"
    
    dataset = animator.load_dataset(data_file)
    if dataset is None:
        return
    
    print(f"\nDataset info:")
    print(f"  Number of sequences: {len(dataset)}")
    print(f"  Sequence length: {dataset.shape[1]} frames")
    print(f"  Number of joints: {dataset.shape[2]}")
    
    print("\nAvailable animation types:")
    print("  1. skeleton - skeleton-only animation")
    print("  2. trajectory - animation with trajectory display")
    print("  3. multiview - multi-view animation")
    print("  4. batch - batch generate animations")
    
    choice = input("\nSelect animation type (1-4, default 1): ").strip() or "1"
    fps_input = input("Set frame rate (default 8): ").strip()
    fps = int(fps_input) if fps_input.isdigit() else 8
    
    if choice == "1":
        idx_input = input(f"Select sequence index (0-{len(dataset)-1}, default 0): ").strip()
        seq_idx = int(idx_input) if idx_input.isdigit() else 0
        seq_idx = max(0, min(seq_idx, len(dataset)-1))
        print(f"\nCreating skeleton animation for sequence {seq_idx+1}...")
        animator.create_skeleton_animation(dataset[seq_idx], seq_idx, fps)
    
    elif choice == "2":
        idx_input = input(f"Select sequence index (0-{len(dataset)-1}, default 0): ").strip()
        seq_idx = int(idx_input) if idx_input.isdigit() else 0
        seq_idx = max(0, min(seq_idx, len(dataset)-1))
        print(f"\nCreating trajectory animation for sequence {seq_idx+1}...")
        animator.create_trajectory_animation(dataset[seq_idx], seq_idx, fps)
    
    elif choice == "3":
        idx_input = input(f"Select sequence index (0-{len(dataset)-1}, default 0): ").strip()
        seq_idx = int(idx_input) if idx_input.isdigit() else 0
        seq_idx = max(0, min(seq_idx, len(dataset)-1))
        print(f"\nCreating multi-view animation for sequence {seq_idx+1}...")
        animator.create_multi_view_animation(dataset[seq_idx], seq_idx, fps)
    
    elif choice == "4":
        max_input = input("Batch count (default 3): ").strip()
        max_seq = int(max_input) if max_input.isdigit() else 3
        anim_type = input("Batch animation type (skeleton/trajectory/multiview, default skeleton): ").strip()
        if anim_type not in ['skeleton', 'trajectory', 'multiview']:
            anim_type = 'skeleton'
        print(f"\nBatch creating {anim_type} animations...")
        animator.batch_create_animations(dataset, anim_type, fps, max_seq)
    
    else:
        print("Invalid choice")
        return
    
    print("\nAnimation generation complete!")
    print("Generated files:")
    print("  cub15_*.gif - animation files")

if __name__ == "__main__":
    main()
