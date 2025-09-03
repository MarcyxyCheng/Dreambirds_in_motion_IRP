#!/usr/bin/env python
"""
Phase 2-5: MDM_inference_CUB15_cond_anchor
--------------------------------------
Built on the "deeply fixed" version with additions:
1) Label-conditioned sampling (classifier-free guidance, CFG)
2) "Noised first-frame anchoring" during sampling (optionally anchor the first K frames)
   to eliminate the second-frame jump back to mean pose.
Other: retains numerical monitoring, stabilization, smoothing, and validation logic.
"""

import torch
import numpy as np
import sys
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append('.')
from mdm.model.mdm import MDM

# ===== Label set consistent with training =====
LABELS = ["takeoff", "gliding", "hovering", "soaring", "diving", "landing"]
LABEL_TO_ID = {n:i for i,n in enumerate(LABELS)}

class CUB15GeneratorCondAnchor:
    def __init__(self, model_path="mdm_cub15_conditional.ckpt", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model, self.normalization_params = self._load_model_and_params(model_path)

        self.diffusion_steps = 1000
        self.noise_schedule = self._create_noise_schedule()

        self._validate_normalization_params()

        self.max_coord_range = 5.0
        self.noise_clamp_range = 8.0

        self.smoothing_window = 5
        self.enable_smoothing = True

        # Anchoring config
        self.anchor_first_k = 1      # anchor first K frames (1–4 commonly used)
        self.anchor_soft_lam = 1.0   # 1.0=hard anchor; <1.0=soft anchor (e.g., 0.7–0.9)

    def _load_model_and_params(self, model_path):
        print(f"Loading model: {model_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        normalization_params = ckpt.get('normalization_params', None)
        if normalization_params is None:
            norm_file = "outputs/normalization_params.pkl"
            if Path(norm_file).exists():
                with open(norm_file, 'rb') as f:
                    normalization_params = pickle.load(f)
                print(f"Loaded normalization params from file: {norm_file}")
            else:
                raise FileNotFoundError("Normalization params not found")

        num_actions = len(ckpt.get('labels', LABELS))  # compatibility
        model = MDM(
            modeltype='trans_enc',
            njoints=15, nfeats=3, num_actions=num_actions,
            translation=True, pose_rep='xyz', glob=True, glob_rot=True,
            device=self.device, cond_mode='action',
            latent_dim=256, ff_size=1024, num_layers=8, num_heads=4,
            dropout=0.1, activation="gelu", data_rep='xyz', dataset='cub15'
        )
        # model.load_state_dict(ckpt['model_state_dict'])
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.to(self.device); model.eval()
        print(f"Model loaded (epoch {ckpt.get('epoch','?')}) | num actions: {num_actions}")
        return model, normalization_params

    def _validate_normalization_params(self):
        for k in ['coords_mean','coords_std','original_coord_range','visibility_value']:
            if k not in self.normalization_params:
                raise ValueError(f"Normalization params missing key: {k}")
        print("Normalization params validated")

    def _create_noise_schedule(self):
        steps = self.diffusion_steps
        s = 0.008
        x = torch.linspace(0, steps, steps + 1)
        ac = torch.cos(((x/steps)+s)/(1+s)*torch.pi*0.5)**2
        ac = ac / ac[0]
        betas = torch.clamp(1 - (ac[1:]/ac[:-1]), 0, 0.999)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return {
            'betas': betas.to(self.device),
            'alphas_cumprod': alphas_cumprod.to(self.device),
            'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod).to(self.device),
            'sqrt_one_minus_alphas_cumprod': torch.sqrt(1 - alphas_cumprod).to(self.device),
        }

    def _normalize_input(self, pose):
        coords = pose[:, :2]
        vis = pose[:, 2:3]
        mean = self.normalization_params['coords_mean']; std = self.normalization_params['coords_std']
        coords_norm = (coords - mean[None,:]) / std[None,:]
        vis_norm = np.ones_like(vis)
        out = np.concatenate([coords_norm, vis_norm], axis=1)
        print(f"Input normalization: xy [{coords.min():.3f},{coords.max():.3f}] -> [{coords_norm.min():.3f},{coords_norm.max():.3f}]")
        return out

    def _denormalize_output(self, seq):
        coords = seq[:, :, :2]; vis = seq[:, :, 2:3]
        mean = self.normalization_params['coords_mean']; std = self.normalization_params['coords_std']
        coords = coords * std[None,None,:] + mean[None,None,:]
        vis = np.ones_like(vis)
        return np.concatenate([coords, vis], axis=2)

    def _stabilized_ddim_step(self, x, noise_pred, t, t_next, alpha_t, alpha_next):
        x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        x0_pred = torch.clamp(x0_pred, -self.max_coord_range, self.max_coord_range)
        if t_next >= 0:
            x_next = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * noise_pred
        else:
            x_next = x0_pred
        x_next = torch.clamp(x_next, -self.noise_clamp_range, self.noise_clamp_range)
        return x_next

    def _make_condition(self, B, T, label_id, normalized_init_pose=None, uncond=False):
        cond = {
            'mask': torch.ones(B,1,1,T, device=self.device, dtype=torch.bool),
            'lengths': torch.full((B,), T, device=self.device),
            'uncond': uncond,
            'action': torch.full((B,1), label_id, dtype=torch.long, device=self.device)
        }
        if normalized_init_pose is not None:
            cond['init_pose'] = torch.tensor(normalized_init_pose, dtype=torch.float32, device=self.device).unsqueeze(0)
        return cond

    def _compute_noised_firstK(self, x0_firstK, t_scalar):
        """
        Forward noising the first K frames in normalized space:
        x_t = sqrt(a_t)*x_0 + sqrt(1-a_t)*eps
        Args:
            x0_firstK: (K, J, 3) normalized first K frames
        Returns:
            (K, J, 3) noised first K frames at time t_scalar
        """
        alpha_t = self.noise_schedule['alphas_cumprod'][t_scalar]
        sa = torch.sqrt(alpha_t)
        so = torch.sqrt(1 - alpha_t)
        eps = torch.randn_like(x0_firstK)
        return sa * x0_firstK + so * eps

    def generate_sequence(self, init_pose, 
                          action_label="hovering",  # "takeoff","gliding","hovering","soaring","diving","landing"
                          num_inference_steps=60, guidance_scale=3.0,
                          seed=None, debug=True,
                          enable_smoothing=True, smoothing_window=5,
                          anchor_first_k=1, anchor_soft_lam=1.0):
        """
        Generate a 64-frame sequence from an initial pose + action label
        (conditional + CFG + in-sampling first-frame anchoring).
        """
        self.enable_smoothing = enable_smoothing
        self.smoothing_window = smoothing_window
        self.anchor_first_k = int(anchor_first_k)
        self.anchor_soft_lam = float(anchor_soft_lam)

        if seed is not None:
            torch.manual_seed(seed); np.random.seed(seed)

        assert init_pose.shape == (15,3), f"init_pose shape {init_pose.shape} != (15,3)"
        if action_label not in LABELS:
            raise ValueError(f"Unknown action label {action_label}, options: {LABELS}")
        label_id = LABEL_TO_ID[action_label]

        print(f"Conditional generation: label='{action_label}' id={label_id}, steps={num_inference_steps}, "
              f"CFG={guidance_scale}, anchor first K={self.anchor_first_k}, soft lambda={self.anchor_soft_lam:.2f}")

        # Normalize first frame
        norm_init = self._normalize_input(init_pose)
        norm_init_t = torch.tensor(norm_init, dtype=torch.float32, device=self.device)   # (J,3)

        B, T = 1, 64

        # Initialize noise from duplicated first frame then add noise at t_T
        x0_all = norm_init_t.unsqueeze(-1).repeat(1,1,T)      # (J,3,T)
        x0_all = x0_all.permute(2,0,1)                        # (T,J,3)
        x0_all = x0_all.permute(1,2,0).unsqueeze(0)           # (1,J,3,T)
        t_T = self.diffusion_steps - 1
        sa_T = torch.sqrt(self.noise_schedule['alphas_cumprod'][t_T]).to(self.device)
        so_T = torch.sqrt(1 - self.noise_schedule['alphas_cumprod'][t_T]).to(self.device)
        eps0 = torch.randn_like(x0_all)
        x = sa_T * x0_all + so_T * eps0                        # (1,J,3,T)

        print(f"Initial noise range: [{x.min().item():.3f}, {x.max().item():.3f}]")

        # Timesteps
        if num_inference_steps <= 50:
            timesteps = torch.linspace(self.diffusion_steps-1, 0, num_inference_steps, dtype=torch.long, device=self.device)
        else:
            dense = int(num_inference_steps*0.7)
            sparse = num_inference_steps - dense
            dense_range  = torch.linspace(self.diffusion_steps-1, self.diffusion_steps//2, dense, dtype=torch.long)
            sparse_range = torch.linspace(self.diffusion_steps//2 - 1, 0, sparse, dtype=torch.long)
            timesteps = torch.cat([dense_range, sparse_range]).to(self.device)

        x_ranges, noise_ranges = [], []

        with torch.no_grad():
            for i, t in enumerate(timesteps):
                if debug and i % max(1, len(timesteps)//10) == 0:
                    print(f"  Step {i+1:3d}/{num_inference_steps} (t={int(t)}) ", end='')

                t_batch = t.repeat(B)
                x = torch.clamp(x, -self.noise_clamp_range, self.noise_clamp_range)

                # CFG two-branch prediction
                cond_uncond = self._make_condition(B, T, label_id, normalized_init_pose=norm_init, uncond=True)
                cond_cond   = self._make_condition(B, T, label_id, normalized_init_pose=norm_init, uncond=False)

                noise_uncond = torch.clamp(self.model(x, t_batch, cond_uncond), -self.noise_clamp_range, self.noise_clamp_range)
                noise_cond   = torch.clamp(self.model(x, t_batch, cond_cond),   -self.noise_clamp_range, self.noise_clamp_range)

                g = float(guidance_scale)
                noise_pred = noise_uncond + g * (noise_cond - noise_uncond)

                xr = [x.min().item(), x.max().item()]
                nr = [noise_pred.min().item(), noise_pred.max().item()]
                x_ranges.append(xr); noise_ranges.append(nr)
                if debug and i % max(1, len(timesteps)//10) == 0:
                    print(f"x:[{xr[0]:6.2f},{xr[1]:6.2f}] noise:[{nr[0]:6.2f},{nr[1]:6.2f}]")

                if abs(xr[0])>50 or abs(xr[1])>50: x = torch.clamp(x, -10, 10)
                if abs(nr[0])>50 or abs(nr[1])>50: noise_pred = torch.clamp(noise_pred, -10, 10)

                alpha_t = self.noise_schedule['alphas_cumprod'][t]
                if i < len(timesteps)-1:
                    t_next = timesteps[i+1]
                    alpha_next = self.noise_schedule['alphas_cumprod'][t_next]
                    x = self._stabilized_ddim_step(x, noise_pred, t, t_next, alpha_t, alpha_next)
                else:
                    x = self._stabilized_ddim_step(x, noise_pred, t, -1, alpha_t, None)

                # ===== In-sampling anchoring for the first K frames =====
                K = max(1, self.anchor_first_k)
                t_ref = int(t_next) if i < len(timesteps)-1 else int(t)
                init_firstK = norm_init_t.unsqueeze(0).repeat(K,1,1)   # (K,J,3)
                noised_firstK = self._compute_noised_firstK(init_firstK, t_ref)  # (K,J,3)

                lam = self.anchor_soft_lam
                for k in range(K):
                    if lam >= 0.999:
                        x[..., k] = noised_firstK[k]
                    else:
                        x[..., k] = lam * noised_firstK[k] + (1-lam) * x[..., k]

        print("\nInference finished")
        x = torch.clamp(x, -self.max_coord_range, self.max_coord_range)
        print(f"Final output range (normalized): [{x.min().item():.3f}, {x.max().item():.3f}]")

        seq_norm = x[0].permute(2,0,1).cpu().numpy()   # (T,J,3)
        seq_denorm = self._denormalize_output(seq_norm)

        # Safety: first frame equals init_pose
        if np.linalg.norm(seq_denorm[0] - init_pose) > 1.0:
            print("Adjusting first frame to match initial pose")
            seq_denorm[0] = init_pose

        seq_out = self._apply_temporal_smoothing(seq_denorm)

        cr = [seq_out[:,:,:2].min(), seq_out[:,:,:2].max()]
        print(f"Sequence generation complete, coord range: [{cr[0]:.3f}, {cr[1]:.3f}]")

        # Save debug info
        self._save_debug_info(x_ranges, noise_ranges, num_inference_steps)
        return seq_out

    def _apply_temporal_smoothing(self, sequence):
        if not self.enable_smoothing: return sequence
        print(f"Applying temporal smoothing (window: {self.smoothing_window})")
        sm = sequence.copy()
        T,J,_ = sm.shape
        important = [1,2,3,4,5,6,7,8,9]
        for j in range(J):
            for c in range(2):
                window = self.smoothing_window if j in important else max(3, self.smoothing_window-2)
                s = sequence[:, j, c]
                out = np.zeros_like(s)
                for t in range(T):
                    a = max(0, t-window//2); b = min(T, t+window//2+1)
                    w = np.arange(a,b) - t
                    w = np.exp(-0.5*(w/(max(1,window/4)))**2); w = w/w.sum()
                    out[t] = np.sum(s[a:b]*w)
                sm[:, j, c] = out
        return sm

    def _save_debug_info(self, x_ranges, noise_ranges, num_steps):
        info = {'x_ranges': x_ranges, 'noise_ranges': noise_ranges, 'num_steps': num_steps}
        with open('debug_sampling_info.pkl', 'wb') as f:
            pickle.dump(info, f)
        try:
            steps = range(len(x_ranges))
            xm = [r[0] for r in x_ranges]; xM = [r[1] for r in x_ranges]
            nm = [r[0] for r in noise_ranges]; nM = [r[1] for r in noise_ranges]
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1); plt.plot(steps,xm,label='x_min'); plt.plot(steps,xM,label='x_max')
            plt.xlabel('Sampling Step'); plt.ylabel('Value Range'); plt.title('X Range'); plt.legend(); plt.grid(True)
            plt.subplot(1,2,2); plt.plot(steps,nm,label='noise_min'); plt.plot(steps,nM,label='noise_max')
            plt.xlabel('Sampling Step'); plt.ylabel('Value Range'); plt.title('Noise Range'); plt.legend(); plt.grid(True)
            plt.tight_layout(); plt.savefig('debug_sampling_ranges.png', dpi=150, bbox_inches='tight'); plt.close()
            print("Saved debug chart: debug_sampling_ranges.png")
        except:
            print("Failed to save debug chart, but data is saved to debug_sampling_info.pkl")

# ===== Example entry =====
def main():
    print("CUB-15 Conditional + First-frame Anchoring Inference")
    print("="*60)

    try:
        gen = CUB15GeneratorCondAnchor("mdm_cub15_conditional.ckpt")
    except Exception as e:
        print(f"Init failed: {e}")
        return

    init_path = "generated_pose_seq_fixed.npy"
    if not Path(init_path).exists():
        print(f"Initial pose not found: {init_path}")
        return
    init_pose = np.load(init_path)[0]  # (15,3)
    print(f"Loaded initial pose: {init_path}")

    seq = gen.generate_sequence(
        init_pose,
        action_label="takeoff",
        num_inference_steps=60,
        guidance_scale=3.5,
        seed=42,
        debug=True,
        enable_smoothing=True,
        smoothing_window=5,
        anchor_first_k=2,      # recommend anchoring first 2 frames for smoother transition
        anchor_soft_lam=0.9    # soft anchoring to avoid overly hard constraint
    )

    np.save("generated_pose_seq_cond_anchor.npy", seq)
    print("Saved: generated_pose_seq_cond_anchor.npy")

if __name__ == "__main__":
    main()
