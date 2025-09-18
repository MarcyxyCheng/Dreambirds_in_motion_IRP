#!/usr/bin/env python
"""
Phase 2-3: train_MDM_CUB15
--------------------------------------
Fixed version + conditional training (action label) + first-frame consistency loss
Also adds:
1) Dataset subset ratio (train_ratio / val_ratio)
2) Per-epoch batch loss records (JSON+CSV)
3) Overall training history summary (JSON+CSV)
Save directory: reportoutput/phase2/trainMDM
"""

import torch
import numpy as np
import datetime
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
import pickle
import json
import csv
import random

# add MDM module path
sys.path.append('.')
from mdm.model.mdm import MDM

# ======= labels =======
LABELS = ["takeoff", "gliding", "hovering", "soaring", "diving", "landing"]
LABEL_TO_ID = {n: i for i, n in enumerate(LABELS)}
NUM_ACTIONS = len(LABELS)

# === hyperparameters ===
CONFIG = {
    # data
    'njoints': 15,
    'nfeats': 3,
    'sequence_length': 64,
    'pose_dim': 45,

    # training
    'epochs_stage1': 80,
    'epochs_stage2': 20,
    'batch_size': 16,
    'lr_stage1': 1e-4,
    'lr_stage2': 5e-5,

    # diffusion
    'diffusion_steps': 1000,
    'noise_schedule': 'cosine',
    'beta_start': 0.0001,
    'beta_end': 0.01,

    # loss weights
    'mse_weight': 0.5,
    'skeleton_weight': 0.5,
    'range_weight': 0.7,
    'first_frame_weight': 0.5,

    # CFG / conditional training
    'enable_uncond_prob': False,
    'uncond_prob': 0.0,

    # save / eval
    'save_interval': 10,
    'eval_interval': 5,

    # New: dataset subset ratios (1.0 = full data)
    'train_ratio': 1.00,   # e.g., 0.10 uses 10% of training data; 1.0 for full
    'val_ratio':   1.00,   # usually keep full validation set
    'subset_seed': 42,     # random seed for subset sampling (reproducible)
}

# ========= dataset =========
class CUB15DatasetCond(torch.utils.data.Dataset):
    """
    CUB-15 conditional dataset:
    - data: npy of shape (N, T, J, C)
    - labels: npy/int of shape (N,), values 0..num_actions-1
    """
    def __init__(self, data_path, label_path, augment=False, normalization_params=None):
        self.data = np.load(data_path)  # (N, T, J, C)
        assert self.data.ndim == 4 and self.data.shape[2] == 15 and self.data.shape[3] == 3, \
            f"data shape invalid: {self.data.shape}"

        labels = np.load(label_path) if isinstance(label_path, str) else label_path
        assert len(labels) == len(self.data), f"labels len {len(labels)} != data len {len(self.data)}"
        self.labels = labels.astype(np.int64)
        self.augment = augment

        print(f"Loaded data: {data_path}")
        print(f"   Raw shape: {self.data.shape}")
        print(f"   Label histogram: {np.bincount(self.labels, minlength=NUM_ACTIONS)}")

        # to (N, J, C, T)
        self.data = self.data.transpose(0, 2, 3, 1)
        print(f"   Transposed shape: {self.data.shape}")

        # normalization
        self.data, self.normalization_params = self._normalize_data_with_params(
            self.data, normalization_params
        )
        self._validate_data()

    def _normalize_data_with_params(self, data, existing_params=None):
        coords = data[:, :, :2, :]  # (N, J, 2, T)
        visibility = data[:, :, 2:3, :]

        if existing_params is None:
            coords_flat = coords.reshape(-1, 2)
            mean = np.mean(coords_flat, axis=0)
            std = np.std(coords_flat, axis=0) + 1e-8
            normalization_params = {
                'coords_mean': mean,
                'coords_std': std,
                'original_coord_range': [coords_flat.min(axis=0), coords_flat.max(axis=0)],
                'visibility_value': 1.0
            }
            print("   Computing normalization params")
            print(f"     coords_mean: {mean}")
            print(f"     coords_std : {std}")
        else:
            mean = existing_params['coords_mean']
            std  = existing_params['coords_std']
            normalization_params = existing_params
            print("   Using existing normalization params")

        coords_norm = (coords - mean[None,None,:,None]) / std[None,None,:,None]
        visibility_norm = np.ones_like(visibility)
        normalized_data = np.concatenate([coords_norm, visibility_norm], axis=2)

        cr = [coords_norm.min(), coords_norm.max()]
        print(f"     Normalized coord range: [{cr[0]:.3f}, {cr[1]:.3f}]")
        return normalized_data.astype(np.float32), normalization_params

    def _validate_data(self):
        print("   Data quality check:")
        print(f"     Value range: [{self.data.min():.3f}, {self.data.max():.3f}]")
        nan_count = np.sum(np.isnan(self.data))
        inf_count = np.sum(np.isinf(self.data))
        if nan_count == 0 and inf_count == 0:
            print("     OK: no NaN/Inf")
        visibility = self.data[:, :, 2, :]
        print(f"     Mean visibility: {np.mean(visibility):.3f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)  # (J, C, T)
        label = int(self.labels[idx])
        if self.augment and np.random.random() < 0.3:
            noise = torch.randn_like(sample[:, :2, :]) * 0.02
            sample[:, :2, :] += noise
        return sample, label

# ========= noise schedule =========
def create_noise_schedule(schedule_type, steps, beta_start=0.0001, beta_end=0.02):
    if schedule_type == 'linear':
        betas = torch.linspace(beta_start, beta_end, steps)
    elif schedule_type == 'cosine':
        s = 0.008
        x = torch.linspace(0, steps, steps + 1)
        alphas_cumprod = torch.cos(((x/steps)+s)/(1+s)*torch.pi*0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:]/alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0, 0.999)
    else:
        raise ValueError(schedule_type)

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
    return {
        'betas': betas, 'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1 - alphas_cumprod),
    }

# ========= condition dict =========
def create_condition_dict(batch_size, sequence_length, device, labels=None, uncond_flags=False):
    cond = {
        'mask': torch.ones(batch_size, 1, 1, sequence_length, device=device, dtype=torch.bool),
        'lengths': torch.full((batch_size,), sequence_length, device=device),
        'uncond': bool(uncond_flags),   # pass scalar bool, compatible with mdm.mask_cond
    }
    if labels is not None:
        # aligned with mdm.EmbedAction: forward takes idx = input[:,0]
        cond['action'] = labels.view(-1, 1).long()
    return cond

# ========= losses =========
def compute_skeleton_loss(pred, target):
    cub_connections = [
        (0,1),(1,6),(6,7),(7,8),
        (1,2),(2,5),
        (1,3),(1,4),
        (8,9),
        (7,12),
        (7,10),(7,11),
        (9,13),(9,14)
    ]
    loss = 0.0
    for j1,j2 in cub_connections:
        pd = torch.norm(pred[:, j1, :2, :] - pred[:, j2, :2, :], dim=1)
        gt = torch.norm(target[:, j1, :2, :] - target[:, j2, :2, :], dim=1)
        loss += F.mse_loss(pd, gt)
    return loss / len(cub_connections)

def compute_range_constraint_loss(pred, expected_range=(-5,5)):
    coords = pred[:, :, :2, :]
    lo, hi = expected_range
    below = torch.clamp(lo - coords, min=0)
    above = torch.clamp(coords - hi, min=0)
    return (below.pow(2).mean() + above.pow(2).mean())

# ========= train / val =========
def train_epoch(model, train_loader, optimizer, device, epoch, noise_schedule, report_dir=None):
    model.train()
    total = total_mse = total_skel = total_range = total_first = 0.0
    num_batches = len(train_loader)

    batch_records = []

    for bidx, (x, lbl) in enumerate(train_loader):
        x = x.to(device)                    # (B, J, C, T)
        lbl = lbl.to(device).long()         # (B,)
        B = x.shape[0]

        # random timestep
        t = torch.randint(0, CONFIG['diffusion_steps'], (B,), device=device)

        # CFG disabled here (could be enabled probabilistically; must pass scalar bool)
        uflag = False

        condition = create_condition_dict(B, CONFIG['sequence_length'], device, labels=lbl, uncond_flags=uflag)

        # forward diffusion
        noise = torch.randn_like(x)
        sa = noise_schedule['sqrt_alphas_cumprod'][t].view(B,1,1,1)
        so = noise_schedule['sqrt_one_minus_alphas_cumprod'][t].view(B,1,1,1)
        x_noisy = sa * x + so * noise

        optimizer.zero_grad()

        # predict noise
        pred_noise = model(x_noisy, t, condition)
        mse_loss = F.mse_loss(pred_noise, noise)

        # reconstruct x0 (pred_x0)
        pred_x0 = (x_noisy - so * pred_noise) / sa

        # auxiliary losses
        skeleton_loss = compute_skeleton_loss(pred_x0, x)
        range_loss    = compute_range_constraint_loss(pred_x0, expected_range=(-5, 4))

        # first-frame consistency loss (xy only)
        first_pred = pred_x0[:, :, :2, 0]  # (B, J, 2)
        first_gt   = x[:,      :, :2, 0]
        first_loss = F.mse_loss(first_pred, first_gt)

        loss = (CONFIG['mse_weight'] * mse_loss +
                CONFIG['skeleton_weight'] * skeleton_loss +
                CONFIG['range_weight'] * range_loss +
                CONFIG['first_frame_weight'] * first_loss)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Skip abnormal batch {bidx} (loss={loss.item()})")
            continue

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total += loss.item(); total_mse += mse_loss.item()
        total_skel += skeleton_loss.item(); total_range += range_loss.item()
        total_first += first_loss.item()

        # record batch results
        batch_records.append({
            "epoch": epoch+1,
            "batch": bidx,
            "loss": float(loss.item()),
            "mse": float(mse_loss.item()),
            "skeleton": float(skeleton_loss.item()),
            "range": float(range_loss.item()),
            "first": float(first_loss.item())
        })

        # moderate logging
        if bidx % max(1, num_batches//10) == 0:
            print(f"  Batch [{bidx:3d}/{num_batches}] "
                  f"Loss:{loss.item():.6f} MSE:{mse_loss.item():.6f} "
                  f"Skeleton:{skeleton_loss.item():.6f} Range:{range_loss.item():.6f} "
                  f"First:{first_loss.item():.6f}")

    # === save batch-level logs ===
    if report_dir is not None and batch_records:
        with open(report_dir / f"epoch{epoch+1:03d}_batches.json", "w") as f:
            json.dump(batch_records, f, indent=2)
        with open(report_dir / f"epoch{epoch+1:03d}_batches.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=batch_records[0].keys())
            writer.writeheader()
            writer.writerows(batch_records)

    n = num_batches
    return total/n, total_mse/n, total_skel/n, total_range/n, total_first/n

def validate_model(model, val_loader, device, noise_schedule):
    print(f"[DEBUG] Validation batch count: {len(val_loader)}")
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, lbl in val_loader:
            x = x.to(device)
            lbl = lbl.to(device).long()
            B = x.shape[0]
            t = torch.randint(0, CONFIG['diffusion_steps'], (B,), device=device)
            condition = create_condition_dict(B, CONFIG['sequence_length'], device, labels=lbl, uncond_flags=False)

            noise = torch.randn_like(x)
            sa = noise_schedule['sqrt_alphas_cumprod'][t].view(B,1,1,1)
            so = noise_schedule['sqrt_one_minus_alphas_cumprod'][t].view(B,1,1,1)
            x_noisy = sa * x + so * noise

            pred_noise = model(x_noisy, t, condition)
            mse_loss = F.mse_loss(pred_noise, noise)
            total_loss += mse_loss.item()
    return total_loss / len(val_loader)

# ========= utils =========
def save_checkpoint(model, optimizer, epoch, loss, filepath, config=None, normalization_params=None):
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'config': config or CONFIG,
        'normalization_params': normalization_params,
        'labels': LABELS,  # store labels list for inference alignment
        'timestamp': datetime.datetime.now().isoformat()
    }
    torch.save(ckpt, filepath)
    print(f"Model saved: {filepath}")

def log_training(message, logbook_path):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(logbook_path, "a", encoding='utf-8') as f:
        f.write(f"{ts}: {message}\n")
    print(f"[{ts}] {message}")

# ========= main =========
def main():
    # fix random seeds (including subset sampling)
    np.random.seed(CONFIG['subset_seed'])
    random.seed(CONFIG['subset_seed'])
    torch.manual_seed(CONFIG['subset_seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # output & report dirs
    output_dir = Path("outputs"); output_dir.mkdir(exist_ok=True)
    report_dir = Path("reportoutput/phase2/trainMDM"); report_dir.mkdir(parents=True, exist_ok=True)
    logbook = output_dir / "2.3_cub15_training_log_cond_first.md"

    with open(logbook, "w", encoding='utf-8') as f:
        f.write("# CUB-15 MDM Training Log (Conditional + First-Frame Consistency)\n\n")
        f.write(f"Start time: {datetime.datetime.now()}\n\n")
        f.write("## Config\n")
        for k,v in CONFIG.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")

    # paths
    train_data_path = "cub15_enhanced_train_pose_large_wing.npy"
    val_data_path   = "cub15_enhanced_val_pose_large_wing.npy"
    train_label_path = "cub15_enhanced_train_labels.npy"
    val_label_path   = "cub15_enhanced_val_labels.npy"

    if not os.path.exists(train_data_path) or not os.path.exists(train_label_path):
        print(f"Missing training data or labels: {train_data_path}, {train_label_path}")
        return
    if not os.path.exists(val_data_path) or not os.path.exists(val_label_path):
        print(f"Missing validation data or labels: {val_data_path}, {val_label_path}")
        return

    log_training("Loading CUB-15 conditional dataset...", logbook)
    full_train = CUB15DatasetCond(train_data_path, train_label_path, augment=True)
    # validation normalization must match training
    full_val   = CUB15DatasetCond(val_data_path, val_label_path, augment=False,
                                  normalization_params=full_train.normalization_params)

    # save normalization params
    normalization_params = full_train.normalization_params
    with open(output_dir / "normalization_params.pkl", 'wb') as f:
        pickle.dump(normalization_params, f)
    log_training(f"Normalization params saved to: {output_dir}/normalization_params.pkl", logbook)

    # ===== subset sampling =====
    def make_subset(ds, ratio):
        if ratio >= 1.0:
            return ds
        n = len(ds)
        k = max(1, int(n * ratio))
        idx = list(range(n))
        random.Random(CONFIG['subset_seed']).shuffle(idx)
        sub_idx = idx[:k]
        return Subset(ds, sub_idx)

    train_dataset = make_subset(full_train, CONFIG['train_ratio'])
    val_dataset   = make_subset(full_val,   CONFIG['val_ratio'])

    log_training(f"Train samples: {len(train_dataset)} / total {len(full_train)} "
                 f"(ratio={CONFIG['train_ratio']})", logbook)
    log_training(f"Val samples: {len(val_dataset)} / total {len(full_val)} "
                 f"(ratio={CONFIG['val_ratio']})", logbook)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=2, pin_memory=False)

    # noise schedule to device
    log_training("Creating noise schedule...", logbook)
    noise_schedule = create_noise_schedule(CONFIG['noise_schedule'],
                                           CONFIG['diffusion_steps'],
                                           CONFIG['beta_start'],
                                           CONFIG['beta_end'])
    for k in list(noise_schedule.keys()):
        noise_schedule[k] = noise_schedule[k].to(device)

    # ======= initialize conditional MDM model =======
    log_training("Initializing conditional MDM model...", logbook)
    model = MDM(
        modeltype='trans_enc',
        njoints=CONFIG['njoints'],
        nfeats=CONFIG['nfeats'],
        num_actions=NUM_ACTIONS,
        translation=True,
        pose_rep='xyz',
        glob=True,
        glob_rot=True,
        device=device,
        cond_mode='action',   # must match your mdm.py
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        data_rep='xyz',
        dataset='cub15',
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_training(f"Model params: total={total_params:,}, trainable={trainable_params:,}", logbook)

    # === stage 1 ===
    log_training("Start Stage 1: base training (conditional + first-frame consistency)", logbook)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr_stage1'],
                            weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG['epochs_stage1'])

    best_val = float('inf')
    history = []

    for epoch in range(CONFIG['epochs_stage1']):
        log_training(f"\nEpoch [{epoch+1}/{CONFIG['epochs_stage1']}]", logbook)
        train_loss, train_mse, train_skel, train_range, train_first = train_epoch(
            model, train_loader, optimizer, device, epoch, noise_schedule, report_dir
        )
        if (epoch % CONFIG['eval_interval']) == 0:
            val_loss = validate_model(model, val_loader, device, noise_schedule)
        else:
            val_loss = 0.0

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        history.append({
            'epoch': epoch+1,
            'train_loss': float(train_loss),
            'train_mse': float(train_mse),
            'train_skeleton': float(train_skel),
            'train_range': float(train_range),
            'train_first': float(train_first),
            'val_loss': float(val_loss),
            'lr': float(lr)
        })

        msg = (f"Train:{train_loss:.6f} (MSE:{train_mse:.6f} Skel:{train_skel:.6f} "
               f"Range:{train_range:.6f} First:{train_first:.6f}) | "
               f"Val:{val_loss:.6f} | LR:{lr:.2e}")
        log_training(msg, logbook)

        if val_loss > 0 and val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            output_dir/"mdm_cub15_cond_best.ckpt",
                            CONFIG, normalization_params)
            log_training(f"New best model! Val Loss: {val_loss:.6f}", logbook)

        if (epoch+1) % CONFIG['save_interval'] == 0:
            save_checkpoint(model, optimizer, epoch, train_loss,
                            output_dir/f"mdm_cub15_cond_epoch_{epoch+1}.ckpt",
                            CONFIG, normalization_params)

    # === stage 2 (finetune) ===
    log_training("\nStart Stage 2: finetuning", logbook)
    for pg in optimizer.param_groups:
        pg['lr'] = CONFIG['lr_stage2']

    for ep in range(CONFIG['epochs_stage2']):
        total_ep = CONFIG['epochs_stage1'] + ep + 1
        log_training(f"\nFinetune Epoch [{ep+1}/{CONFIG['epochs_stage2']}] (total {total_ep})", logbook)
        train_loss, train_mse, train_skel, train_range, train_first = train_epoch(
            model, train_loader, optimizer, device, total_ep, noise_schedule, report_dir
        )
        val_loss = validate_model(model, val_loader, device, noise_schedule)

        msg = (f"Train:{train_loss:.6f} (MSE:{train_mse:.6f} Range:{train_range:.6f} First:{train_first:.6f}) | "
               f"Val:{val_loss:.6f}")
        log_training(msg, logbook)

        history.append({
            'epoch': total_ep,
            'train_loss': float(train_loss),
            'train_mse': float(train_mse),
            'train_skeleton': float(train_skel),
            'train_range': float(train_range),
            'train_first': float(train_first),
            'val_loss': float(val_loss),
            'lr': float(optimizer.param_groups[0]['lr'])
        })

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, total_ep, val_loss,
                            output_dir/"mdm_cub15_cond_best.ckpt",
                            CONFIG, normalization_params)
            log_training(f"New best in finetune! Val Loss: {val_loss:.6f}", logbook)

    # save final
    save_checkpoint(model, optimizer,
                    CONFIG['epochs_stage1']+CONFIG['epochs_stage2'],
                    best_val, output_dir/"mdm_cub15_cond_final.ckpt",
                    CONFIG, normalization_params)

    # duplicate best to project root
    import shutil
    if os.path.exists(output_dir/"mdm_cub15_cond_best.ckpt"):
        shutil.copy(output_dir/"mdm_cub15_cond_best.ckpt", "mdm_cub15_conditional.ckpt")

    # save epoch-level history (report dir + outputs)
    with open(report_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(report_dir / "training_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    with open(output_dir/"training_history_cond.json", "w") as f:
        json.dump(history, f, indent=2)

    log_training("\nTraining complete (conditional + first-frame consistency).", logbook)
    log_training(f"Best model: mdm_cub15_conditional.ckpt", logbook)
    log_training(f"Training history: {report_dir}/training_history.json (CSV with same name)", logbook)
    log_training(f"Normalization params: {output_dir}/normalization_params.pkl", logbook)

if __name__ == "__main__":
    main()
