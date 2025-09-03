#!/usr/bin/env python
"""
Phase 2-4: MDM_inference_CUB15_fixed
----------------------------------------------------------
Complete fixes for numerical explosion:
1. Full-process numeric monitoring
2. Multi-layer numerical stabilization
3. Progressive sampling strategy
4. Data quality validation
"""

import torch
import numpy as np
import sys
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# Add MDM module path
sys.path.append('.')
from mdm.model.mdm import MDM

class CUB15Generator:
    """CUB-15 point sequence generator (deeply fixed version)"""
    
    def __init__(self, model_path="mdm_cub15.ckpt", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and normalization params
        self.model, self.normalization_params = self._load_model_and_params(model_path)
        
        # Diffusion parameters
        self.diffusion_steps = 1000
        self.noise_schedule = self._create_noise_schedule()
        
        # Validate normalization params
        self._validate_normalization_params()
        
        # Numerical stabilization parameters (normalized space)
        self.max_coord_range = 5.0    # max coordinate range
        self.noise_clamp_range = 8.0  # noise clamp range

        # Post-processing smoothing parameters
        self.smoothing_window = 5
        self.enable_smoothing = True
    
    def _load_model_and_params(self, model_path):
        """Load trained model and normalization parameters"""
        print(f"Loading model: {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})
        
        # Load normalization params
        normalization_params = checkpoint.get('normalization_params', None)
        if normalization_params is None:
            # Try loading from a separate file
            norm_file = "outputs/normalization_params.pkl"
            if Path(norm_file).exists():
                with open(norm_file, 'rb') as f:
                    normalization_params = pickle.load(f)
                print(f"Loaded normalization params from file: {norm_file}")
            else:
                raise FileNotFoundError(
                    "Normalization params not found. Make sure training saved them "
                    "(outputs/normalization_params.pkl or inside the checkpoint)."
                )
        else:
            print("Loaded normalization params from checkpoint")
        
        # Initialize unconditional model
        model = MDM(
            modeltype='trans_enc',
            njoints=15,
            nfeats=3,
            num_actions=1,
            translation=True,
            pose_rep='xyz',
            glob=True,
            glob_rot=True,
            device=self.device,
            cond_mode='no_cond',
            latent_dim=256,
            ff_size=1024,
            num_layers=8,
            num_heads=4,
            dropout=0.1,
            activation="gelu",
            data_rep='xyz',
            dataset='cub15',
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
        return model, normalization_params
    
    def _validate_normalization_params(self):
        """Validate the integrity and reasonableness of normalization params"""
        required_keys = ['coords_mean', 'coords_std', 'original_coord_range', 'visibility_value']
        for key in required_keys:
            if key not in self.normalization_params:
                raise ValueError(f"Normalization params missing key: {key}")
        
        mean = self.normalization_params['coords_mean']
        std = self.normalization_params['coords_std']
        
        print("Normalization params check:")
        print(f"  coords_mean: {mean}")
        print(f"  coords_std: {std}")
        print(f"  original_range: {self.normalization_params['original_coord_range']}")
        print(f"  visibility_value: {self.normalization_params['visibility_value']}")
        
        if np.any(std < 0.01):
            print(f"  Warning: std too small, may cause instability: {std}")
        if np.any(np.abs(mean) > 5.0):
            print(f"  Warning: mean too large, may cause issues: {mean}")
    
    def _create_noise_schedule(self):
        """Create a conservative cosine noise schedule"""
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
            'sqrt_recip_alphas_cumprod': torch.sqrt(1.0 / alphas_cumprod).to(self.device),
            'sqrt_recipm1_alphas_cumprod': torch.sqrt(1.0 / alphas_cumprod - 1).to(self.device),
        }
    
    def _create_condition(self, batch_size, sequence_length):
        """Create condition dict (unconditional mode)"""
        return {
            'mask': torch.ones(batch_size, 1, 1, sequence_length, device=self.device),
            'lengths': torch.full((batch_size,), sequence_length, device=self.device),
            'uncond': False,
        }
    
    def _normalize_input(self, init_pose):
        """Normalize input pose as in training"""
        coords = init_pose[:, :2]      # (15, 2)
        visibility = init_pose[:, 2:3] # (15, 1)
        
        mean = self.normalization_params['coords_mean']  # (2,)
        std = self.normalization_params['coords_std']    # (2,)
        
        coords_norm = (coords - mean[None, :]) / std[None, :]
        visibility_norm = np.ones_like(visibility)
        
        normalized_pose = np.concatenate([coords_norm, visibility_norm], axis=1)
        
        print("Input normalization:")
        print(f"  raw coord range: [{coords.min():.3f}, {coords.max():.3f}]")
        print(f"  normalized range: [{coords_norm.min():.3f}, {coords_norm.max():.3f}]")
        
        if np.abs(coords_norm).max() > 5.0:
            print("  Warning: large normalized values, may affect stability")
        
        return normalized_pose
    
    def _denormalize_output(self, sequence):
        """Denormalize from normalized space back to original coordinate space"""
        print("Running denormalization...")
        
        coords_norm = sequence[:, :, :2]  # (T, J, 2)
        visibility = sequence[:, :, 2:3]  # (T, J, 1)
        
        mean = self.normalization_params['coords_mean']  # (2,)
        std = self.normalization_params['coords_std']    # (2,)
        
        print(f"  using params: mean={mean}, std={std}")
        print(f"  normalized coord range: [{coords_norm.min():.3f}, {coords_norm.max():.3f}]")
        
        if np.abs(coords_norm).max() > 10.0:
            print("  Warning: abnormal normalized range; clipping applied")
            coords_norm = np.clip(coords_norm, -10.0, 10.0)
        
        coords_denorm = coords_norm * std[None, None, :] + mean[None, None, :]
        visibility_fixed = np.ones_like(visibility)
        
        denormalized_sequence = np.concatenate([coords_denorm, visibility_fixed], axis=2)
        
        print(f"  denormalized coord range: [{coords_denorm.min():.3f}, {coords_denorm.max():.3f}]")
        print(f"  visibility range: [{visibility_fixed.min():.3f}, {visibility_fixed.max():.3f}]")
        
        return denormalized_sequence
    
    def _stabilized_ddim_step(self, x, noise_pred, t, t_next, alpha_t, alpha_next):
        """Numerically stabilized DDIM single-step update"""
        x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        x0_pred = torch.clamp(x0_pred, -self.max_coord_range, self.max_coord_range)
        
        if t_next >= 0:
            x_next = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * noise_pred
        else:
            x_next = x0_pred
        
        x_next = torch.clamp(x_next, -self.noise_clamp_range, self.noise_clamp_range)
        return x_next
    
    def generate_sequence(self, init_pose, num_inference_steps=50, guidance_scale=1.0, 
                          seed=None, debug=True, enable_smoothing=True, smoothing_window=5):
        """
        Generate a 64-frame sequence from an initial pose (numerically stabilized).
        
        Args:
            init_pose: (15, 3) numpy array, initial pose (original coordinate space)
            num_inference_steps: steps for inference (recommended 50–100)
            guidance_scale: guidance strength (reserved)
            seed: RNG seed
            debug: print debug info
            enable_smoothing: apply temporal smoothing
            smoothing_window: smoothing window size
        
        Returns:
            (64, 15, 3) numpy array in original coordinate space
        """
        self.enable_smoothing = enable_smoothing
        self.smoothing_window = smoothing_window
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        print(f"Start generating sequence (steps: {num_inference_steps})")
        
        if init_pose.shape != (15, 3):
            raise ValueError(f"Bad init pose shape: {init_pose.shape}, expected (15, 3)")
        
        print(f"Input initial pose range: [{init_pose[:, :2].min():.3f}, {init_pose[:, :2].max():.3f}]")
        
        # 1) Normalize input
        normalized_init_pose = self._normalize_input(init_pose)
        normalized_init_pose = torch.tensor(normalized_init_pose, dtype=torch.float32, device=self.device)
        
        batch_size = 1
        sequence_length = 64
        
        # Condition (unconditional mode)
        condition = self._create_condition(batch_size, sequence_length)
        
        # 2) Initialize noise (smaller amplitude)
        x = torch.randn(batch_size, 15, 3, sequence_length, device=self.device) * 0.5
        print(f"Initial noise range: [{x.min():.3f}, {x.max():.3f}]")
        
        # 3) Time-step schedule
        if num_inference_steps <= 50:
            timesteps = torch.linspace(self.diffusion_steps - 1, 0, num_inference_steps, dtype=torch.long, device=self.device)
        else:
            dense_steps = int(num_inference_steps * 0.7)
            sparse_steps = num_inference_steps - dense_steps
            dense_range = torch.linspace(self.diffusion_steps - 1, self.diffusion_steps // 2, dense_steps, dtype=torch.long)
            sparse_range = torch.linspace(self.diffusion_steps // 2 - 1, 0, sparse_steps, dtype=torch.long)
            timesteps = torch.cat([dense_range, sparse_range]).to(self.device)
        
        # Track ranges
        x_ranges = []
        noise_ranges = []
        
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                if debug and i % max(1, len(timesteps) // 10) == 0:
                    print(f"  Step {i+1:3d}/{num_inference_steps} (t={t:3d}) ", end='')
                
                t_batch = t.repeat(batch_size)
                
                # Input clamp stabilization
                x = torch.clamp(x, -self.noise_clamp_range, self.noise_clamp_range)
                
                # Predict noise (in normalized space)
                noise_pred = self.model(x, t_batch, condition)
                noise_pred = torch.clamp(noise_pred, -self.noise_clamp_range, self.noise_clamp_range)
                
                # Record ranges
                x_range = [x.min().item(), x.max().item()]
                noise_range = [noise_pred.min().item(), noise_pred.max().item()]
                x_ranges.append(x_range)
                noise_ranges.append(noise_range)
                
                if debug and i % max(1, len(timesteps) // 10) == 0:
                    print(f"x:[{x_range[0]:6.2f},{x_range[1]:6.2f}] noise:[{noise_range[0]:6.2f},{noise_range[1]:6.2f}]")
                
                # Explosion checks
                if abs(x_range[0]) > 50 or abs(x_range[1]) > 50:
                    print(f"  Warning: value explosion (x range abnormal), hard clipping")
                    x = torch.clamp(x, -10, 10)
                
                if abs(noise_range[0]) > 50 or abs(noise_range[1]) > 50:
                    print(f"  Warning: value explosion (noise range abnormal), hard clipping")
                    noise_pred = torch.clamp(noise_pred, -10, 10)
                
                # DDIM update
                alpha_t = self.noise_schedule['alphas_cumprod'][t]
                
                if i < len(timesteps) - 1:
                    t_next = timesteps[i + 1]
                    alpha_next = self.noise_schedule['alphas_cumprod'][t_next]
                    x = self._stabilized_ddim_step(x, noise_pred, t, t_next, alpha_t, alpha_next)
                else:
                    x = self._stabilized_ddim_step(x, noise_pred, t, -1, alpha_t, None)
        
        print("\nInference complete")
        
        # Final clamp stabilization
        x = torch.clamp(x, -self.max_coord_range, self.max_coord_range)
        final_range = [x.min().item(), x.max().item()]
        print(f"Final model output range (normalized): [{final_range[0]:.3f}, {final_range[1]:.3f}]")
        
        # (1, 15, 3, 64) -> (64, 15, 3)
        result_normalized = x[0].permute(2, 0, 1).cpu().numpy()
        
        # Denormalize to original space
        result_denormalized = self._denormalize_output(result_normalized)
        
        # Align first frame to initial pose
        if np.linalg.norm(result_denormalized[0] - init_pose) > 1.0:
            print("Adjust first frame to match the initial pose")
            result_denormalized[0] = init_pose

        # Temporal smoothing
        result_denormalized = self._apply_temporal_smoothing(result_denormalized)
        
        # Overall range check
        final_coord_range = [result_denormalized[:, :, :2].min(), result_denormalized[:, :, :2].max()]
        print("Sequence generation finished")
        print(f"Final output coord range: [{final_coord_range[0]:.3f}, {final_coord_range[1]:.3f}]")
        
        if abs(final_coord_range[0]) > 20 or abs(final_coord_range[1]) > 20:
            print("Warning: output range abnormal; consider checking model or data quality")
            
        # Optional: save debug info
        if debug:
            self._save_debug_info(x_ranges, noise_ranges, num_inference_steps)
        
        return result_denormalized
    
    def _apply_temporal_smoothing(self, sequence):
        """
        Apply temporal smoothing to reduce frame-to-frame jumps.
        Args:
            sequence: (64, 15, 3)
        Returns:
            (64, 15, 3)
        """
        if not self.enable_smoothing:
            return sequence
            
        print(f"Applying temporal smoothing (window size: {self.smoothing_window})")
        
        smoothed = sequence.copy()
        T, J, C = sequence.shape
        
        # Focused joints: head + wings + body
        important_joints = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        for joint_idx in range(J):
            for coord_idx in range(2):  # only x, y
                if joint_idx in important_joints:
                    window = self.smoothing_window
                else:
                    window = max(3, self.smoothing_window - 2)
                
                coord_sequence = sequence[:, joint_idx, coord_idx]
                smoothed_coord = np.zeros_like(coord_sequence)
                
                for t in range(T):
                    start_idx = max(0, t - window // 2)
                    end_idx = min(T, t + window // 2 + 1)
                    window_data = coord_sequence[start_idx:end_idx]
                    window_size = len(window_data)
                    
                    if window_size == 1:
                        smoothed_coord[t] = window_data[0]
                    else:
                        weights = np.exp(-0.5 * ((np.arange(window_size) - window_size // 2) / (window_size / 4)) ** 2)
                        weights = weights / weights.sum()
                        smoothed_coord[t] = np.average(window_data, weights=weights)
                
                smoothed[:, joint_idx, coord_idx] = smoothed_coord
        
        coord_diff = np.abs(smoothed[:, :, :2] - sequence[:, :, :2])
        mean_change = np.mean(coord_diff)
        max_change = np.max(coord_diff)
        print(f"  mean coord change: {mean_change:.4f}")
        print(f"  max coord change: {max_change:.4f}")
        
        original_jumps = np.mean(np.abs(np.diff(sequence[:, :, :2], axis=0)))
        smoothed_jumps = np.mean(np.abs(np.diff(smoothed[:, :, :2], axis=0)))
        improvement = (original_jumps - smoothed_jumps) / max(original_jumps, 1e-8) * 100
        print(f"  inter-frame jump improvement: {improvement:.1f}%")
        
        return smoothed
    
    def _save_debug_info(self, x_ranges, noise_ranges, num_steps):
        """Save sampling debug info and plots"""
        debug_info = {
            'x_ranges': x_ranges,
            'noise_ranges': noise_ranges,
            'num_steps': num_steps
        }
        
        with open('debug_sampling_info.pkl', 'wb') as f:
            pickle.dump(debug_info, f)
        
        # Simple visualization
        try:
            steps = range(len(x_ranges))
            x_mins = [r[0] for r in x_ranges]
            x_maxs = [r[1] for r in x_ranges]
            
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(steps, x_mins, label='x_min')
            plt.plot(steps, x_maxs, label='x_max')
            plt.xlabel('Sampling Step')
            plt.ylabel('Value Range')
            plt.title('X Value Range During Sampling')
            plt.legend()
            plt.grid(True)
            
            noise_mins = [r[0] for r in noise_ranges]
            noise_maxs = [r[1] for r in noise_ranges]
            
            plt.subplot(1, 2, 2)
            plt.plot(steps, noise_mins, label='noise_min')
            plt.plot(steps, noise_maxs, label='noise_max')
            plt.xlabel('Sampling Step')
            plt.ylabel('Value Range')
            plt.title('Noise Prediction Range During Sampling')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('debug_sampling_ranges.png', dpi=150, bbox_inches='tight')
            print("Saved debug chart: debug_sampling_ranges.png")
            plt.close()
        except:
            print("Failed to save debug chart, but sampling data saved to debug_sampling_info.pkl")
    
    def generate_batch(self, init_poses, **kwargs):
        """Generate multiple sequences in batch"""
        results = []
        for i, init_pose in enumerate(init_poses):
            print(f"Generating sequence {i+1}/{len(init_poses)}")
            result = self.generate_sequence(init_pose, **kwargs)
            results.append(result)
        return np.array(results)
    
    def validate_with_training_data(self, train_data_path="cub15_train_pose.npy", num_samples=5):
        """Quick validation using training data"""
        print("\nValidating inference quality with training data...")
        
        if not Path(train_data_path).exists():
            print(f"Training data file not found: {train_data_path}")
            return
        
        train_data = np.load(train_data_path)
        print(f"Training data shape: {train_data.shape}")
        print(f"Training data range: [{train_data.min():.3f}, {train_data.max():.3f}]")
        
        indices = np.random.choice(len(train_data), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            print(f"\nValidation sample {i+1}/{num_samples} (train index: {idx})")
            init_pose = train_data[idx, 0]   # (15, 3)
            ground_truth = train_data[idx]   # (64, 15, 3)
            print(f"  GT coord range: [{ground_truth[:, :, :2].min():.3f}, {ground_truth[:, :, :2].max():.3f}]")
            
            generated = self.generate_sequence(init_pose, num_inference_steps=30, debug=False)
            
            coord_diff = np.abs(generated[:, :, :2] - ground_truth[:, :, :2])
            mean_diff = np.mean(coord_diff)
            max_diff = np.max(coord_diff)
            
            print(f"  Generated coord range: [{generated[:, :, :2].min():.3f}, {generated[:, :, :2].max():.3f}]")
            print(f"  Mean coord diff: {mean_diff:.3f}")
            print(f"  Max coord diff: {max_diff:.3f}")
            
            if mean_diff < 1.0 and max_diff < 5.0:
                print("  Quality: good")
            elif mean_diff < 3.0:
                print("  Quality: fair")
            else:
                print("  Quality: poor")

def create_test_pose():
    """Create a test pose based on typical ranges"""
    print("Creating a test pose using empirical ranges...")
    
    typical_ranges = {
        'head': (-0.5, 0.5),
        'body': (-1.0, 1.0),
        'wings': (-2.0, 2.0),
        'tail': (-0.8, 0.8),
        'legs': (-0.6, 0.6),
    }
    
    pose = np.zeros((15, 3))
    
    # Head (0–5)
    for i in range(6):
        pose[i, :2] = np.random.uniform(*typical_ranges['head'], 2)
    
    # Body (6–9)
    for i in range(6, 10):
        pose[i, :2] = np.random.uniform(*typical_ranges['body'], 2)
    
    # Wings (10–11)
    for i in range(10, 12):
        pose[i, :2] = np.random.uniform(*typical_ranges['wings'], 2)
    
    # Tail (12)
    pose[12, :2] = np.random.uniform(*typical_ranges['tail'], 2)
    
    # Legs (13–14)
    for i in range(13, 15):
        pose[i, :2] = np.random.uniform(*typical_ranges['legs'], 2)
    
    # Visibility = 1
    pose[:, 2] = 1.0
    
    print(f"Test pose coord range: [{pose[:, :2].min():.3f}, {pose[:, :2].max():.3f}]")
    return pose

def main():
    """Main: usage demo"""
    print("CUB-15 Point MDM Sequence Generator (Deeply Fixed Version)")
    print("=" * 60)
    
    # Initialize generator
    try:
        generator = CUB15Generator("mdm_cub15.ckpt")
    except FileNotFoundError as e:
        print(f"{e}")
        print("Please ensure step 2.3 training completed and normalization params were saved")
        return
    except Exception as e:
        print(f"Initialization failed: {e}")
        return
    
    # Quick validation with training data (optional)
    print("\n" + "="*40)
    print("Step 1: Validation using training data")
    print("="*40)
    generator.validate_with_training_data()
    
    # Example 1: Use an existing test pose file
    print("\n" + "="*40)
    print("Step 2: File-based test pose generation")
    print("="*40)
    
    # init_pose = create_test_pose()
    init_pose_path = "generated_pose_seq_fixed.npy"
    init_pose = np.load(init_pose_path)[0]  # shape: [15, 3]
    print(f"Loaded initial pose: {init_pose_path}")

    # Conservative inference params
    print("\nStarting conservative-parameter inference...")
    sequence = generator.generate_sequence(
        init_pose, 
        num_inference_steps=100,
        seed=42,
        debug=True
    )

    print(f"Generation done, sequence shape: {sequence.shape}")
    
    # Quality check
    coord_range = [sequence[:, :, :2].min(), sequence[:, :, :2].max()]
    vis_range = [sequence[:, :, 2].min(), sequence[:, :, 2].max()]
    
    print("\nQuality check:")
    print(f"  Coord range: [{coord_range[0]:.3f}, {coord_range[1]:.3f}]")
    print(f"  Visibility range: [{vis_range[0]:.3f}, {vis_range[1]:.3f}]")
    
    if abs(coord_range[0]) < 10 and abs(coord_range[1]) < 10:
        print("  Coord range is normal")
    else:
        print("  Warning: coord range abnormal; further debugging may be needed")
    
    # Save single sequence
    output_path = "generated_pose_seq_raw.npy"
    np.save(output_path, sequence)
    print(f"Sequence saved: {output_path}")
    
    # Example 2: Quick inference
    print("\n" + "="*40)
    print("Step 3: Quick inference test")
    print("="*40)
    
    quick_sequence = generator.generate_sequence(
        init_pose, 
        num_inference_steps=20,
        seed=123,
        debug=False
    )
    
    quick_coord_range = [quick_sequence[:, :, :2].min(), quick_sequence[:, :, :2].max()]
    print(f"Quick inference done, coord range: [{quick_coord_range[0]:.3f}, {quick_coord_range[1]:.3f}]")
    
    # Example 3: Batch generation
    print("\n" + "="*40)
    print("Step 4: Small-batch generation")
    print("="*40)
    
    batch_init_poses = []
    for i in range(3):
        pose = create_test_pose()
        batch_init_poses.append(pose)
    
    batch_sequences = generator.generate_batch(
        batch_init_poses,
        num_inference_steps=50,
        seed=456,
        debug=False
    )
    
    print(f"Batch generation done, shape: {batch_sequences.shape}")
    
    batch_coord_range = [batch_sequences[:, :, :, :2].min(), batch_sequences[:, :, :, :2].max()]
    print(f"Batch coord range: [{batch_coord_range[0]:.3f}, {batch_coord_range[1]:.3f}]")
    
    # Save batch results
    batch_output_path = "generated_batch_sequences_fixed.npy"
    np.save(batch_output_path, batch_sequences)
    print(f"Batch sequences saved: {batch_output_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("Deeply fixed inference test complete")
    print("="*60)
    
    print("Output files:")
    print(f"  - {output_path} (single sequence)")
    print(f"  - {batch_output_path} (batch sequences)")
    print("  - debug_sampling_ranges.png (debug chart)")
    print("  - debug_sampling_info.pkl (debug data)")
    
    print("\nFinal quality assessment:")
    all_coords = [coord_range, quick_coord_range, batch_coord_range]
    
    quality_good = 0
    quality_issues = 0
    
    for i, (min_val, max_val) in enumerate(all_coords):
        test_names = ["detailed inference", "quick inference", "batch inference"]
        if abs(min_val) < 5 and abs(max_val) < 5:
            print(f"  {test_names[i]}: excellent (range: [{min_val:.2f}, {max_val:.2f}])")
            quality_good += 1
        elif abs(min_val) < 15 and abs(max_val) < 15:
            print(f"  {test_names[i]}: acceptable (range: [{min_val:.2f}, {max_val:.2f}])")
        else:
            print(f"  {test_names[i]}: problematic (range: [{min_val:.2f}, {max_val:.2f}])")
            quality_issues += 1
    
    print("\nTroubleshooting tips:")
    print("  1. If coord range > 20: likely diffusion sampling explosion")
    print("  2. If coord range < 0.1: likely underfitting or data issues")
    print("  3. If batch results vary greatly: likely seed/init effects")
    print("  4. Inspect debug_sampling_ranges.png for sampling dynamics")

if __name__ == "__main__":
    main()
