#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1-6: visualize evaluation results
-----------------------
Render figures from the outputs of 1.5_eval_hrnet.py.

Inputs (produced by the evaluator):
  - {WORK_DIR}/evaluation_results/evaluation_results.json
  - {WORK_DIR}/evaluation_results/keypoint_performance.csv

Outputs:
  - {WORK_DIR}/evaluation_results/figs/*.png

Optional:
  - If `evaluation_results.json` contains `coco.ap_per_kpt`, draws a per-keypoint OKS-AP heatmap.
  - If `evaluation_results.json` contains `lr_confusion`, draws left/right confusion pie charts.

Usage:
  python 1.2.4_visualize_eval.py --workdir workdir/hrnet15_cub
"""

import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----- Groups for aggregation -----
RIGID_KEYS = {'beak', 'crown', 'back', 'tail'}
FLEX_KEYS  = {'left_wing', 'right_wing', 'left_leg', 'right_leg'}

def _norm_key(s: str) -> str:
    """Normalize keypoint names to lowercase snake_case for grouping."""
    return str(s).strip().lower().replace('-', '_').replace(' ', '_')

def load_eval(workdir: Path):
    eval_dir = workdir / "evaluation_results"
    json_path = eval_dir / "evaluation_results.json"
    csv_path  = eval_dir / "keypoint_performance.csv"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    with open(json_path, "r") as f:
        E = json.load(f)

    df = pd.read_csv(csv_path)
    df['Keypoint_norm'] = df['Keypoint'].apply(_norm_key)
    return E, df, eval_dir

def ensure_out(eval_dir: Path):
    out = eval_dir / "figs"
    out.mkdir(parents=True, exist_ok=True)
    return out

# ---------- Fig 1: Overall PCK vs tau ----------
def plot_pck_curve(E, out_dir: Path):
    pck = E.get('pck', {})
    if not pck:
        return
    taus, vals = [], []
    # Keys like "PCK@0.05", "PCK@0.1", ...
    for k in sorted(pck.keys(), key=lambda x: float(x.split('@')[-1])):
        taus.append(float(k.split('@')[-1]))
        vals.append(float(pck[k]['overall']))
    if not taus:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(taus, vals, marker='o')
    plt.xlabel('τ (fraction of bbox diagonal)')
    plt.ylabel('PCK')
    plt.title('Overall PCK vs τ (CUB-15)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'pck_overall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

# ---------- Fig 2: Rigid vs Flexible at PCK@0.1 ----------
def plot_rigid_flex(E, out_dir: Path):
    pck01 = E.get('pck', {}).get('PCK@0.1', {})
    per_point = pck01.get('per_point', {})
    if not per_point:
        return

    rigid_vals, flex_vals = [], []
    for name, v in per_point.items():
        n = _norm_key(name)
        if n in RIGID_KEYS:
            rigid_vals.append(float(v))
        if n in FLEX_KEYS:
            flex_vals.append(float(v))

    if not rigid_vals and not flex_vals:
        return

    means = [
        np.mean(rigid_vals) if len(rigid_vals) else np.nan,
        np.mean(flex_vals)  if len(flex_vals)  else np.nan
    ]

    plt.figure(figsize=(5, 4))
    bars = plt.bar(['Rigid', 'Flexible'], means, alpha=0.85)
    for i, m in enumerate(means):
        if not np.isnan(m):
            plt.text(i, m + 0.01, f'{m:.3f}', ha='center', fontweight='bold')
    ymax = max([x for x in means if not np.isnan(x)], default=1.0)
    plt.ylim(0, ymax * 1.2)
    plt.ylabel('PCK@0.1')
    plt.title('PCK@0.1 by Group (Rigid vs Flexible)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'pck01_rigid_vs_flexible.png', dpi=300, bbox_inches='tight')
    plt.close()

# ---------- Fig 3: Per-keypoint PCK@0.1 (sorted) ----------
def plot_per_keypoint_pck_sorted(E, out_dir: Path):
    pck01 = E.get('pck', {}).get('PCK@0.1', {})
    per_point = pck01.get('per_point', {})
    if not per_point:
        return
    items = sorted(per_point.items(), key=lambda x: x[1])  # easiest to hardest
    names = [k for k, _ in items]
    vals  = [float(v) for _, v in items]
    plt.figure(figsize=(10, 0.4 * len(names) + 2))
    plt.barh(names, vals, alpha=0.85)
    for i, v in enumerate(vals):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    plt.xlim(0, max(vals) * 1.2)
    plt.xlabel('PCK@0.1')
    plt.title('Per-Keypoint PCK@0.1 (sorted)')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'pck01_per_keypoint_sorted.png', dpi=300, bbox_inches='tight')
    plt.close()

# ---------- Fig 4: Per-keypoint normalized error (mean ± std) ----------
def plot_error_bar(df, out_dir: Path):
    needed = {'Mean_Error', 'Std_Error', 'Keypoint'}
    if not needed.issubset(df.columns):
        return
    d = df[['Keypoint', 'Mean_Error', 'Std_Error']].copy().sort_values('Mean_Error', ascending=True)
    plt.figure(figsize=(10, 0.4 * len(d) + 2))
    plt.barh(d['Keypoint'], d['Mean_Error'], xerr=d['Std_Error'], alpha=0.85)
    for i, (m, s) in enumerate(zip(d['Mean_Error'], d['Std_Error'])):
        label_x = m + (s if s > 0 else 0.005)
        plt.text(label_x, i, f'{m:.3f}±{s:.3f}', va='center', fontsize=9)
    plt.xlabel('Normalized Error (by bbox diagonal)')
    plt.title('Per-Keypoint Error (mean ± std)')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'error_mean_std.png', dpi=300, bbox_inches='tight')
    plt.close()

# ---------- Fig 5: Left/Right confusion (if present) ----------
def plot_lr_confusion(E, out_dir: Path):
    lr = E.get('lr_confusion', None)
    if not lr:
        return

    def _pie(stats: dict, title: str, fname: str):
        total = stats.get('total_cases', 0)
        confused = stats.get('confused_cases', 0)
        normal = max(total - confused, 0)
        if total <= 0:
            return
        plt.figure(figsize=(5, 5))
        plt.pie([normal, confused],
                labels=['Normal', 'Confused'],
                autopct='%1.1f%%',
                startangle=90)
        rate = confused / total if total else 0.0
        plt.title(f"{title}\nTotal: {total}  (Confusion {rate:.1%})")
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=300, bbox_inches='tight')
        plt.close()

    if 'wing_confusion' in lr:
        _pie(lr['wing_confusion'], 'Left/Right Wing Confusion', 'lr_confusion_wing.png')
    if 'leg_confusion' in lr:
        _pie(lr['leg_confusion'], 'Left/Right Leg Confusion', 'lr_confusion_leg.png')

# ---------- Fig 6: Per-keypoint OKS-AP heatmap (if present) ----------
def plot_ap_heatmap(E, out_dir: Path):
    coco = E.get('coco', {})
    apk = coco.get('ap_per_kpt', None)
    names = coco.get('keypoint_names', None)
    if apk is None:
        return
    vals = np.array(apk, dtype=float).reshape(-1, 1)
    if names is None:
        # fallback: use PCK key order or kp_i
        per_point = E.get('pck', {}).get('PCK@0.1', {}).get('per_point', {})
        names = list(per_point.keys()) if per_point else [f'kp_{i}' for i in range(len(apk))]
    plt.figure(figsize=(6, 0.35 * len(names) + 1))
    im = plt.imshow(vals, aspect='auto')
    plt.yticks(range(len(names)), names)
    plt.xticks([0], ['AP@[.50:.95]'])
    for i, v in enumerate(vals[:, 0]):
        plt.text(0, i, f'{v:.2f}', ha='center', va='center',
                 color=('white' if v < 0.5 else 'black'), fontsize=9)
    plt.title('Per-Keypoint OKS AP')
    plt.colorbar(im, fraction=0.046, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_dir / 'ap_per_keypoint_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# ---------- Small README with a quick summary ----------
def write_readme(E, out_dir: Path):
    readme = out_dir / "README.md"
    lines = [
        "# Evaluation Figures",
        "",
        "- pck_overall_curve.png",
        "- pck01_rigid_vs_flexible.png",
        "- pck01_per_keypoint_sorted.png",
        "- error_mean_std.png",
        "- lr_confusion_wing.png / lr_confusion_leg.png (if present)",
        "- ap_per_keypoint_heatmap.png (if present)",
        "",
    ]
    pck = E.get('pck', {})
    if 'PCK@0.1' in pck:
        overall = float(pck['PCK@0.1']['overall'])
        lines.append(f"**PCK@0.1 overall**: {overall:.3f} ({overall*100:.1f}%)")
    readme.write_text("\n".join(lines), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, default="workdir/hrnet15_cub",
                        help="Training/evaluation working directory")
    args = parser.parse_args()
    workdir = Path(args.workdir)

    E, df, eval_dir = load_eval(workdir)
    out_dir = ensure_out(eval_dir)

    plot_pck_curve(E, out_dir)
    plot_rigid_flex(E, out_dir)
    plot_per_keypoint_pck_sorted(E, out_dir)
    plot_error_bar(df, out_dir)
    plot_lr_confusion(E, out_dir)
    plot_ap_heatmap(E, out_dir)
    write_readme(E, out_dir)

    print(f"Done. Figures saved to: {out_dir}")

if __name__ == "__main__":
    main()
