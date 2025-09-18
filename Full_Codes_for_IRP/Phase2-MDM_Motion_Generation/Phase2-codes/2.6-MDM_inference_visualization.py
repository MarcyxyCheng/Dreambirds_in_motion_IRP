#!/usr/bin/env python
"""
Phase 2-6: conditional_visualizer
---------------------------------------------------------
Server-friendly tool: generate conditional animation GIFs directly.

Features:
1) Batch-generate GIFs for all actions
2) 15-joint skeleton visualization
3) Action comparison figure
4) Save static key frames
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # force non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import sys

# Add MDM module path
sys.path.append('.')

try:
    from mdm.model.mdm import MDM
except ImportError:
    print("ERROR: Failed to import MDM. Please ensure the module path is correct.")
    sys.exit(1)

# === Action label definitions ===
ACTION_LABELS = {
    0: "flying_straight", 1: "flying_turning", 2: "landing", 3: "takeoff",
    4: "hovering", 5: "diving", 6: "soaring", 7: "flapping_fast",
}

ACTION_DESCRIPTIONS = {
    0: "The bird flies in a straight line with steady wing beats",
    1: "The bird turns while flying, adjusting wing angles",
    2: "The bird prepares to land, slowing down and adjusting posture",
    3: "The bird takes off from ground, powerful wing strokes",
    4: "The bird hovers in place with rapid wing movements",
    5: "The bird dives down quickly with wings folded",
    6: "The bird soars with minimal wing movement, riding air currents",
    7: "The bird flaps wings rapidly for acceleration or maneuvering",
}

# === CUB-15 skeleton connections ===
CUB_CONNECTIONS = [
    # Head (0–5)
    (0, 1), (1, 2), (1, 3), (1, 4), (2, 5),
    # Body spine
    (1, 6), (6, 7), (7, 8), (8, 9),
    # Wings
    (7, 10), (7, 11),
    # Tail
    (8, 12),
    # Legs
    (9, 13), (9, 14),
]

# Joint color definitions
JOINT_COLORS = {
    'head': '#FF6B6B',      # head in red
    'body': '#4ECDC4',      # body in teal
    'wings': '#45B7D1',     # wings in blue
    'tail': '#FFA07A',      # tail in orange
    'legs': '#98D8C8',      # legs in green
}

def get_joint_color(joint_idx):
    """Get color for a joint index."""
    if joint_idx <= 5:
        return JOINT_COLORS['head']
    elif 6 <= joint_idx <= 9:
        return JOINT_COLORS['body']
    elif joint_idx in [10, 11]:
        return JOINT_COLORS['wings']
    elif joint_idx == 12:
        return JOINT_COLORS['tail']
    else:
        return JOINT_COLORS['legs']

class HeadlessConditionalGenerator:
    """Headless conditional generator."""
    
    def __init__(self, model_path="mdm_cub15_conditional.ckpt", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading conditional model (device: {self.device})...")
        
        try:
            self.model, self.normalization_params = self._load_model_and_params(model_path)
            self.diffusion_steps = 1000
            self.noise_schedule = self._create_noise_schedule()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            sys.exit(1)
    
    def _load_model_and_params(self, model_path):
        """Load model and normalization parameters."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        normalization_params = checkpoint.get('normalization_params', None)
        if normalization_params is None:
            raise FileNotFoundError("Normalization parameters not found in checkpoint.")
        
        model = MDM(
            modeltype='trans_enc', njoints=15, nfeats=3, num_actions=8,
            translation=True, pose_rep='xyz', glob=True, glob_rot=True,
            device=self.device, cond_mode='text', clip_version='ViT-B/32',
            latent_dim=256, ff_size=1024, num_layers=8, num_heads=4,
            dropout=0.1, activation="gelu", data_rep='xyz', dataset='cub15_conditional',
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model, normalization_params
    
    def _create_noise_schedule(self):
        """Create cosine noise schedule."""
        steps = self.diffusion_steps
        s = 0.008
        x = torch.linspace(0, steps, steps + 1)
        alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0, 0.999)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        return {
            'betas': betas.to(self.device),
            'alphas_cumprod': alphas_cumprod.to(self.device),
            'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod).to(self.device),
            'sqrt_one_minus_alphas_cumprod': torch.sqrt(1 - alphas_cumprod).to(self.device),
        }
    
    def generate_sequence(self, init_pose, action="flying_straight", num_inference_steps=30, seed=42):
        """Generate a 64-frame sequence for a given action from an initial pose."""
        # Resolve action to id
        if isinstance(action, str):
            action_id = None
            for aid, aname in ACTION_LABELS.items():
                if aname == action:
                    action_id = aid
                    break
            if action_id is None:
                raise ValueError(f"Unknown action: {action}")
        else:
            action_id = action
        
        print(f"Generating action: {ACTION_LABELS[action_id]}")
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Normalize using min/max stored in checkpoint
        coords = init_pose[:, :2]
        data_min = self.normalization_params['data_min']
        data_max = self.normalization_params['data_max']
        data_range = data_max - data_min
        
        coords_norm = (coords - data_min[None, :]) / (data_range[None, :] + 1e-8)
        coords_norm = coords_norm * 3 - 1.5
        coords_norm = np.clip(coords_norm, -2.0, 2.0)
        
        visibility_norm = np.ones((15, 1))
        normalized_pose = np.concatenate([coords_norm, visibility_norm], axis=1)
        normalized_pose = torch.tensor(normalized_pose, dtype=torch.float32, device=self.device)
        
        # Build condition
        text_description = ACTION_DESCRIPTIONS[action_id]
        condition = {
            'mask': torch.ones(1, 1, 1, 64, device=self.device),
            'lengths': torch.full((1,), 64, device=self.device),
            'text': [text_description],
        }
        
        # DDIM sampling
        x = torch.randn(1, 15, 3, 64, device=self.device) * 0.3
        timesteps = torch.linspace(self.diffusion_steps - 1, 0, num_inference_steps, dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                if i % 10 == 0:
                    print(f"   Sampling step: {i+1}/{num_inference_steps}")
                
                t_batch = t.repeat(1)
                x = torch.clamp(x, -3.0, 3.0)
                
                noise_pred = self.model(x, t_batch, condition)
                noise_pred = torch.clamp(noise_pred, -3.0, 3.0)
                
                alpha_t = self.noise_schedule['alphas_cumprod'][t]
                x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                x0_pred = torch.clamp(x0_pred, -2.0, 2.0)
                
                if i < len(timesteps) - 1:
                    t_next = timesteps[i + 1]
                    alpha_next = self.noise_schedule['alphas_cumprod'][t_next]
                    x = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * noise_pred
                else:
                    x = x0_pred
                
                x = torch.clamp(x, -2.5, 2.5)
        
        # Convert and denormalize
        result_normalized = x[0].permute(2, 0, 1).cpu().numpy()
        coords_norm = result_normalized[:, :, :2]
        coords_norm = np.clip(coords_norm, -3.0, 3.0)
        
        coords_denorm = (coords_norm - (-1.5)) / 3.0
        coords_denorm = coords_denorm * data_range[None, None, :] + data_min[None, None, :]
        
        visibility_fixed = np.ones((64, 15, 1))
        denormalized_sequence = np.concatenate([coords_denorm, visibility_fixed], axis=2)
        
        # Ensure first frame matches the input pose
        denormalized_sequence[0] = init_pose
        
        coord_range = [coords_denorm.min(), coords_denorm.max()]
        print(f"   Done. Coord range: [{coord_range[0]:.2f}, {coord_range[1]:.2f}]")
        
        return denormalized_sequence

class HeadlessVisualizer:
    """Headless visualizer."""
    
    def __init__(self):
        print("Initializing headless visualizer...")
        
        # Initialize generator
        self.generator = HeadlessConditionalGenerator()
        
        # Create a test initial pose
        self.init_pose = self._create_test_pose()
        
        # Output directory
        self.output_dir = Path("conditional_animations")
        self.output_dir.mkdir(exist_ok=True)
        
        # Style
        plt.style.use('dark_background')
        
        print("Headless visualizer ready.")
    
    def _create_test_pose(self):
        """Create a test initial pose."""
        pose = np.zeros((15, 3))
        
        typical_ranges = {
            'head': (-0.5, 0.5), 'body': (-1.0, 1.0), 'wings': (-2.0, 2.0),
            'tail': (-0.8, 0.8), 'legs': (-0.6, 0.6),
        }
        
        for i in range(6):  # head
            pose[i, :2] = np.random.uniform(*typical_ranges['head'], 2)
        for i in range(6, 10):  # body
            pose[i, :2] = np.random.uniform(*typical_ranges['body'], 2)
        for i in range(10, 12):  # wings
            pose[i, :2] = np.random.uniform(*typical_ranges['wings'], 2)
        pose[12, :2] = np.random.uniform(*typical_ranges['tail'], 2)  # tail
        for i in range(13, 15):  # legs
            pose[i, :2] = np.random.uniform(*typical_ranges['legs'], 2)
        
        pose[:, 2] = 1.0
        return pose
    
    def draw_frame(self, sequence, frame_idx, ax, title=""):
        """Draw a single frame."""
        ax.clear()
        
        points = sequence[frame_idx]  # (15, 3)
        
        # Axes range
        coord_range = max(4.0, np.abs(points[:, :2]).max() * 1.2)
        ax.set_xlim(-coord_range, coord_range)
        ax.set_ylim(-coord_range, coord_range)
        ax.set_aspect('equal')
        
        # Joints
        for i in range(15):
            if points[i, 2] > 0.5:
                color = get_joint_color(i)
                ax.scatter(points[i, 0], points[i, 1], c=color, s=120, 
                           alpha=0.9, edgecolors='white', linewidths=1)
                ax.text(points[i, 0] + 0.15, points[i, 1] + 0.15, str(i), 
                        fontsize=8, color='white', weight='bold')
        
        # Bones
        for start, end in CUB_CONNECTIONS:
            if points[start, 2] > 0.5 and points[end, 2] > 0.5:
                # Wings highlighted
                if start in [7, 10, 11] or end in [7, 10, 11]:
                    linewidth = 4
                    alpha = 0.9
                    color = '#FFD700'  # gold to emphasize wings
                else:
                    linewidth = 2
                    alpha = 0.7
                    color = 'white'
                
                ax.plot([points[start, 0], points[end, 0]], 
                        [points[start, 1], points[end, 1]], 
                        color=color, linewidth=linewidth, alpha=alpha)
        
        # Title and styling
        ax.set_title(f'{title} - Frame {frame_idx+1}/64', fontsize=14, color='white', pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X', color='white', fontsize=10)
        ax.set_ylabel('Y', color='white', fontsize=10)
        ax.set_facecolor('#1a1a1a')
    
    def generate_single_gif(self, action, fps=10):
        """Generate a GIF for a single action."""
        print(f"\nGenerating GIF for action: {action} ...")
        
        # Generate sequence
        sequence = self.generator.generate_sequence(
            self.init_pose, action=action, num_inference_steps=30,
            seed=42 + hash(action) % 1000
        )
        
        # Animation
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a1a')
        fig.suptitle(f'CUB-15 Skeleton Flight Animation - {action}', fontsize=16, color='white')
        
        def animate(frame):
            self.draw_frame(sequence, frame % 64, ax, action)
            return []
        
        anim = animation.FuncAnimation(fig, animate, frames=64, interval=1000//fps, blit=False, repeat=True)
        
        # Save GIF
        gif_path = self.output_dir / f"{action}_animation.gif"
        print(f"   Saving GIF: {gif_path}")
        anim.save(gif_path, writer='pillow', fps=fps, dpi=100)
        
        # Save first & last frames as PNG
        first_frame_path = self.output_dir / f"{action}_frame_001.png"
        last_frame_path = self.output_dir / f"{action}_frame_064.png"
        
        fig_static, ax_static = plt.subplots(figsize=(8, 6), facecolor='#1a1a1a')
        self.draw_frame(sequence, 0, ax_static, f"{action} - Frame 1")
        plt.tight_layout()
        plt.savefig(first_frame_path, dpi=150, facecolor='#1a1a1a')
        plt.close(fig_static)
        
        fig_static, ax_static = plt.subplots(figsize=(8, 6), facecolor='#1a1a1a')
        self.draw_frame(sequence, 63, ax_static, f"{action} - Frame 64")
        plt.tight_layout()
        plt.savefig(last_frame_path, dpi=150, facecolor='#1a1a1a')
        plt.close(fig_static)
        
        plt.close(fig)
        
        print(f"   {action} done.")
        return sequence
    
    def generate_comparison_gif(self, actions, fps=8):
        """Generate a comparison GIF for multiple actions."""
        print(f"\nGenerating action comparison GIF...")
        
        # Generate sequences
        sequences = {}
        for action in actions:
            print(f"   Generating sequence: {action} ...")
            sequences[action] = self.generator.generate_sequence(
                self.init_pose, action=action, num_inference_steps=25,
                seed=42  # same seed to compare action differences only
            )
        
        # 2x2 or 2x3 grid depending on number of actions
        n_actions = len(actions)
        if n_actions <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3
            
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12), facecolor='#1a1a1a')
        fig.suptitle('CUB-15 Skeleton Flight Action Comparison', fontsize=18, color='white')
        
        if n_actions == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        def animate_comparison(frame):
            for i, action in enumerate(actions[:len(axes)]):
                if i < len(axes):
                    self.draw_frame(sequences[action], frame % 64, axes[i], action)
            for i in range(len(actions), len(axes)):
                axes[i].set_visible(False)
            return []
        
        anim = animation.FuncAnimation(fig, animate_comparison, frames=64, interval=1000//fps, blit=False, repeat=True)
        
        comparison_path = self.output_dir / "action_comparison.gif"
        print(f"   Saving comparison GIF: {comparison_path}")
        anim.save(comparison_path, writer='pillow', fps=fps, dpi=100)
        
        plt.close(fig)
        print("   Comparison GIF complete.")
        
        return sequences
    
    def analyze_differences(self, sequences):
        """Analyze differences between action sequences."""
        print(f"\nAnalyzing action differences...")
        
        actions = list(sequences.keys())
        n_actions = len(actions)
        
        # Difference matrix
        diff_matrix = np.zeros((n_actions, n_actions))
        
        for i, action1 in enumerate(actions):
            for j, action2 in enumerate(actions):
                if i != j:
                    seq1 = sequences[action1]
                    seq2 = sequences[action2]
                    diff = np.mean(np.abs(seq1[:, :, :2] - seq2[:, :, :2]))
                    diff_matrix[i, j] = diff
        
        # Heatmap
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a1a')
        im = ax.imshow(diff_matrix, cmap='viridis', aspect='auto')
        
        ax.set_xticks(range(n_actions))
        ax.set_yticks(range(n_actions))
        ax.set_xticklabels(actions, rotation=45, ha='right', color='white')
        ax.set_yticklabels(actions, color='white')
        
        for i in range(n_actions):
            for j in range(n_actions):
                ax.text(j, i, f'{diff_matrix[i, j]:.3f}', ha="center", va="center", color="white", fontsize=10)
        
        ax.set_title('Action Difference Heatmap', fontsize=16, color='white', pad=20)
        plt.colorbar(im, ax=ax, label='Mean difference')
        plt.tight_layout()
        
        heatmap_path = self.output_dir / "action_differences_heatmap.png"
        plt.savefig(heatmap_path, dpi=150, facecolor='#1a1a1a')
        plt.close(fig)
        
        print(f"   Saved heatmap: {heatmap_path}")
        
        # Stats
        print(f"\nAction difference stats:")
        for i, action1 in enumerate(actions):
            for j, action2 in enumerate(actions):
                if i < j:
                    diff = diff_matrix[i, j]
                    print(f"   {action1} vs {action2}: {diff:.4f}")
        
        avg_diff = np.mean(diff_matrix[diff_matrix > 0]) if np.any(diff_matrix > 0) else 0.0
        print(f"\nAverage difference: {avg_diff:.4f}")
        
        if avg_diff > 0.5:
            print("   Assessment: Strong conditional effect.")
        elif avg_diff > 0.1:
            print("   Assessment: Moderate conditional effect.")
        else:
            print("   Assessment: Weak conditional effect—consider checking the model.")
    
    def generate_all(self):
        """Generate all visualizations."""
        print("Starting batch generation of conditional animations...")
        
        # Main actions to test
        main_actions = ["takeoff", "landing", "hovering", "flying_straight", "soaring"]
        
        # 1) Individual GIFs
        sequences = {}
        for action in main_actions:
            sequences[action] = self.generate_single_gif(action)
        
        # 2) Comparison GIF
        comparison_sequences = self.generate_comparison_gif(main_actions)
        
        # 3) Difference analysis
        self.analyze_differences(comparison_sequences)
        
        # 4) Summary report
        self.generate_summary_report(sequences)
        
        print("\nAll animations generated.")
        print(f"Output directory: {self.output_dir}")
        print("Files generated:")
        for file in sorted(self.output_dir.glob("*")):
            print(f"   - {file.name}")
    
    def generate_summary_report(self, sequences):
        """Write a summary report to disk."""
        report_path = self.output_dir / "generation_report.txt"
        
        from datetime import datetime
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("CUB-15 Conditional Flight Animation Generation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated at: {datetime.now().isoformat()}\n")
            f.write(f"Total actions: {len(sequences)}\n\n")
            
            f.write("Action list:\n")
            for i, (action, seq) in enumerate(sequences.items(), 1):
                coord_range = [seq[:, :, :2].min(), seq[:, :, :2].max()]
                motion_std = np.std(seq[:, :, :2])
                f.write(f"{i}. {action}\n")
                f.write(f"   - Coord range: [{coord_range[0]:.3f}, {coord_range[1]:.3f}]\n")
                f.write(f"   - Motion amplitude (std): {motion_std:.3f}\n")
                f.write(f"   - File: {action}_animation.gif\n\n")
            
            f.write("File notes:\n")
            f.write("- *_animation.gif: full animation for each action\n")
            f.write("- *_frame_*.png: static key frames\n")
            f.write("- action_comparison.gif: side-by-side comparison\n")
            f.write("- action_differences_heatmap.png: difference heatmap\n")
        
        print(f"   Report saved: {report_path}")

def main():
    """Entry point."""
    print("Headless Conditional CUB-15 Flight Animation Generator")
    print("=" * 60)
    print("Server-friendly: writes GIFs and PNGs without opening any window")
    print("=" * 60)
    
    try:
        # Build visualizer
        visualizer = HeadlessVisualizer()
        
        # Generate all assets
        visualizer.generate_all()
        
        print("\nAll tasks complete.")
        print("Check the 'conditional_animations/' directory.")
        print("Key files include:")
        print("   - takeoff_animation.gif")
        print("   - landing_animation.gif")
        print("   - action_comparison.gif")
        print("   - action_differences_heatmap.png")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
    except Exception as e:
        print(f"\nUnhandled exception: {e}")
        import traceback
        traceback.print_exc()
        print("Please verify the model checkpoint and dependencies.")
    
    print("\nDone.")

if __name__ == "__main__":
    main()
