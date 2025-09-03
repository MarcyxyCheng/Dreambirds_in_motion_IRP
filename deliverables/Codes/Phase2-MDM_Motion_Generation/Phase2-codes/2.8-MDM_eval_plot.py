#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2-8: Make four plots from training/eval artifacts:

A1: Train MSE vs Val MSE (first K epochs)
A2: Train loss components (Skeleton / Range) (first K epochs)
B1: PCK vs threshold τ curve for motion generation
B2: Per-keypoint bars: PCK@0.1 (left y-axis) + MPJPE (right y-axis)

Usage (loss only):
    python make_plots_A1_A2.py \
        --history outputs/training_history.json \
        --outdir figs_eval \
        --max_epoch 50

Usage (PCK/MPJPE):
    python make_plots_A1_A2.py \
        --gen_npy generated_pose_seq_fixed2222.npy \
        --gt_npy  cub15_enhanced_val_pose_large_wing.npy \
        --outdir figs_eval \
        --norm bbox \
        --pck_thresholds 0.05,0.10,0.15,0.20 \
        --key_names "back,beak,belly,breast,crown,forehead,left_eye,left_leg,left_wing,nape,right_eye,right_leg,right_wing,tail,throat"

Options:
    --pair_index i  
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional

mpl.rcParams.update({
    "figure.figsize": (9, 6),
    "savefig.dpi": 300,
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "lines.linewidth": 2.6,
    "lines.markersize": 6,
})

# ---------------------- Loss plots ----------------------
def load_history(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        h = json.load(f)
    return h["history"] if isinstance(h, dict) and "history" in h else h

def filter_first_k(epochs, *series, k=50):
    idx = [i for i, e in enumerate(epochs) if e is not None and e <= k]
    out = [[arr[i] for i in idx] for arr in (epochs, *series)]
    return out[0], out[1:]

def plot_A1_train_val_mse(epochs, train_mse, val_mse, out_path, max_epoch):
    if len(epochs) == 0:
        return
    EPS = 1e-12
    epochs, (train_mse, val_mse) = filter_first_k(epochs, train_mse, val_mse, k=max_epoch)
    train_y = [max(v if v is not None else EPS, EPS) for v in train_mse]
    val_pairs = [(e, v) for e, v in zip(epochs, val_mse)
                 if (v is not None) and np.isfinite(v) and (v > 0)]
    val_epochs = [e for e, _ in val_pairs]
    val_values = [max(v, EPS) for _, v in val_pairs]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, train_y, label="Train MSE", linewidth=2.2)
    if val_epochs:
        ax.plot(val_epochs, val_values, "o-", label="Val MSE", linewidth=2.1, markersize=4, alpha=0.95)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.set_title(f"Train vs Val MSE (first {max_epoch} epochs)")
    ax.grid(True, linestyle="--", alpha=0.35); ax.legend(loc="upper right")
    fig.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300); plt.close(fig)

def plot_A2_train_components(epochs, train_skeleton, train_range, out_path, max_epoch):
    if len(epochs) == 0:
        return
    EPS = 1e-12
    epochs, (skel, rng) = filter_first_k(epochs, train_skeleton, train_range, k=max_epoch)
    skel = [max(v if v is not None else EPS, EPS) for v in skel]
    rng  = [max(v if v is not None else EPS, EPS) for v in rng]
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, skel, label="Skeleton", linewidth=2.1)
    plt.plot(epochs, rng,  label="Range",    linewidth=2.1)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Loss Components (Train, first {max_epoch} epochs)")
    plt.grid(True, linestyle="--", alpha=0.35); plt.legend()
    plt.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300); plt.close()

# ---------------------- Eval data loading ----------------------
def _ensure_4d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:         # [T,K,C]
        arr = arr[np.newaxis, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected [N,T,K,C], got {arr.shape}")
    return arr

def load_eval_data(gen_npy: Optional[Path], gt_npy: Optional[Path]) -> Tuple[np.ndarray, np.ndarray]:
    if gen_npy and gt_npy:
        gen_all = _ensure_4d(np.load(str(gen_npy)))
        gt_all  = _ensure_4d(np.load(str(gt_npy)))
        return gen_all, gt_all
    return np.empty((0,)), np.empty((0,))

# ---------------------- logic align----------------------
def align_gen_gt(gen: np.ndarray, gt: np.ndarray, pair_index: int = None):
    Ng, Tg, Kg, Cg = gen.shape
    Nt, Tt, Kt, Ct = gt.shape
    # cut T/K/C
    T = min(Tg, Tt); K = min(Kg, Kt); C = min(Cg, Ct)
    gen = gen[:, :T, :K, :C]; gt = gt[:, :T, :K, :C]

    if pair_index is not None:
        gen = gen[pair_index:pair_index+1]
        gt  = gt [pair_index:pair_index+1]
        return gen, gt

    if Ng == Nt:
        return gen, gt
    if Ng == 1 and Nt > 1:
        gen = np.tile(gen, (Nt,1,1,1))
        return gen, gt
    if Nt == 1 and Ng > 1:
        gt = np.tile(gt, (Ng,1,1,1))
        return gen, gt
    N = min(Ng, Nt)
    return gen[:N], gt[:N]

# ---------------------- Metrics ----------------------
def _bbox_diag(gt: np.ndarray) -> np.ndarray:
    x = gt[..., 0]; y = gt[..., 1]
    xmin = np.nanmin(x, axis=1); xmax = np.nanmax(x, axis=1)
    ymin = np.nanmin(y, axis=1); ymax = np.nanmax(y, axis=1)
    diag = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
    diag[diag <= 1e-9] = 1.0
    return diag

def compute_metrics(gen: np.ndarray, gt: np.ndarray, norm="bbox", img_wh=(256,256)):
    N, T, K, _ = gt.shape
    gen_xy, gt_xy = gen[...,:2], gt[...,:2]

    if norm == "image":
        W,H = img_wh; diag = np.sqrt(W**2+H**2); norm_per_frame = np.full((N,T), diag)
    else:
        norm_per_frame = np.stack([_bbox_diag(gt[n]) for n in range(N)], axis=0)

    err = np.linalg.norm(gen_xy - gt_xy, axis=-1)

    def pck_over_thresholds(thrs: np.ndarray):
        norm_err = err / norm_per_frame[...,None]
        norm_err = norm_err.reshape(-1)
        valid = np.isfinite(norm_err)
        out = []
        for t in thrs:
            out.append(np.nanmean((norm_err[valid] <= t).astype(float)) if valid.sum()>0 else np.nan)
        return np.array(out)

    tau_key=0.10
    mpjpe_per_key=[]; pck_key=[]
    for k in range(K):
        ek = err[...,k]/norm_per_frame
        v=np.isfinite(ek)
        if v.sum()==0:
            mpjpe_per_key.append(np.nan); pck_key.append(np.nan)
        else:
            raw_err_k = np.linalg.norm(gen_xy[...,k,:]-gt_xy[...,k,:],axis=-1)
            mpjpe_per_key.append(np.nanmean(np.where(v,raw_err_k,np.nan)))
            pck_key.append(np.nanmean(np.where(v,(ek<=tau_key).astype(float),np.nan)))

    return {"pck_func":pck_over_thresholds,"pck_per_key":np.array(pck_key),"mpjpe_per_key":np.array(mpjpe_per_key)}

# ---------------------- Plots ----------------------
def plot_B1_pck_curve(pck_gen, taus, out_path, label_gen="MDM (gen)"):
    fig, ax = plt.subplots(figsize=(9,6))
    ax.plot(taus, pck_gen,"o-",label=label_gen)
    ax.set_xlabel("Threshold τ"); ax.set_ylabel("PCK"); ax.set_title("PCK vs Threshold τ")
    ax.set_xticks(list(taus)); ax.set_ylim(0.0,1.05)
    ax.grid(True,linestyle="--",alpha=0.35); ax.legend()
    fig.tight_layout(); out_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out_path,dpi=300); plt.close(fig)

def plot_B2_per_keypoint(pck_key, mpjpe_key, out_path, key_names=None):
    K=len(pck_key); idx=np.arange(K)
    if key_names is None: key_names=[f"k{k}" for k in range(K)]
    fig, ax1 = plt.subplots(figsize=(11,6))
    ax1.bar(idx-0.2,pck_key,width=0.4,label="PCK@0.1")
    ax1.set_ylabel("PCK@0.1"); ax1.set_ylim(0,1.05)
    ax2=ax1.twinx()
    ax2.bar(idx+0.2,mpjpe_key,width=0.4,label="MPJPE (px)",alpha=0.75)
    ax2.set_ylabel("MPJPE (px)")
    ax1.set_title("Per-Keypoint Accuracy (MDM generation)")
    ax1.set_xticks(idx); ax1.set_xticklabels(key_names,rotation=40,ha="right")
    ax1.grid(True,axis="y",linestyle="--",alpha=0.3)
    fig.tight_layout(); out_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out_path,dpi=300); plt.close(fig)

# ---------------------- Main ----------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--history",type=str,default="outputs/training_history.json")
    ap.add_argument("--max_epoch",type=int,default=50)
    ap.add_argument("--gen_npy",type=str,default=None)
    ap.add_argument("--gt_npy", type=str,default=None)
    ap.add_argument("--pair_index",type=int,default=None,help="只比较第 i 条序列，两边都取 index=i；默认自动对齐/广播")
    ap.add_argument("--norm",type=str,choices=["bbox","image"],default="bbox")
    ap.add_argument("--img_wh",nargs=2,type=int,default=[256,256])
    ap.add_argument("--pck_thresholds",type=str,default="0.05,0.10,0.15,0.20")
    ap.add_argument("--key_names",type=str,default="")
    ap.add_argument("--outdir",type=str,default="figs_eval")
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)

    # Loss
    hist=load_history(Path(args.history))
    if hist:
        def getv(d,k,default=None): return d[k] if k in d else default
        epochs=[getv(h,"epoch") for h in hist]
        train_mse=[getv(h,"train_mse") for h in hist]
        val_mse=[getv(h,"val_mse",None) for h in hist]
        train_skeleton=[getv(h,"train_skeleton") for h in hist]
        train_range=[getv(h,"train_range") for h in hist]
        plot_A1_train_val_mse(epochs,train_mse,val_mse,outdir/f"A1_train_val_mse.png",args.max_epoch)
        plot_A2_train_components(epochs,train_skeleton,train_range,outdir/f"A2_train_loss_components.png",args.max_epoch)

    # Eval
    gen_all,gt_all=load_eval_data(Path(args.gen_npy) if args.gen_npy else None,
                                  Path(args.gt_npy) if args.gt_npy else None)
    if gen_all.size!=0 and gt_all.size!=0:
        gen_all,gt_all=align_gen_gt(gen_all,gt_all,pair_index=args.pair_index)
        metrics=compute_metrics(gen_all,gt_all,norm=args.norm,img_wh=(args.img_wh[0],args.img_wh[1]))
        taus=np.array([float(t) for t in args.pck_thresholds.split(",")])
        pck_curve=metrics["pck_func"](taus)
        plot_B1_pck_curve(pck_curve,taus,outdir/"B1_pck_curve.png")
        key_names=[s.strip() for s in args.key_names.split(",")] if args.key_names else None
        plot_B2_per_keypoint(metrics["pck_per_key"],metrics["mpjpe_per_key"],outdir/"B2_per_keypoint.png",key_names=key_names)
        print(f"Saved PCK/MPJPE figures to {outdir.resolve()}")
    else:
        print("No gen/gt npy provided; skip B1/B2.")

    print(f"Done. Figures in {outdir.resolve()}")

if __name__=="__main__":
    main()
