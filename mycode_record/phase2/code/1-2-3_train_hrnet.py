#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_train_hrnet_simple.py
------------------------
简化版训练脚本，避免验证集COCO格式问题
专注于训练，每5个epoch保存一次模型
"""

from pathlib import Path
import json, yaml, textwrap, shutil
import numpy as np
from mmengine import Config
from mmengine.runner import Runner
from mmpose.utils import register_all_modules

# 路径常量
IMG_ROOT = Path("data")
JSON_DIR = IMG_ROOT / "merged17"
WORK_DIR = Path("workdir/hrnet17_simple")  # 使用新的工作目录
CONFIG_OUT = WORK_DIR / "bird17_hrnet_simple.py"
CKPT_BEST = WORK_DIR / "bird17_detector.pth"
NUM_EPOCH = 10  # 直接训练20个epoch

def make_simple_cfg():
    """生成简化的训练配置，无验证集评估"""
    num_joints = yaml.safe_load(Path("skeleton.yaml").read_text())["num_joints"]
    
    cfg_text = textwrap.dedent(f"""
    _base_ = [
        'mmpose::animal_2d_keypoint/topdown_heatmap/ap10k/td-hm_hrnet-w32_8xb64-210e_ap10k-256x256.py',
    ]

    # ---- 基本配置 ----
    data_root = 'data'

    # ---- 只使用训练数据加载器 ----
    train_dataloader = dict(
        batch_size=16,
        dataset=dict(
            type='CocoDataset',
            data_root=data_root,
            ann_file='merged17/train.json',
            data_prefix=dict(img=''),
        )
    )

    # ---- 禁用验证集 ----
    val_dataloader = None
    val_cfg = None
    val_evaluator = None

    # ---- 模型配置 ----
    model = dict(
        head=dict(
            out_channels={num_joints},
        )
    )

    # ---- 训练配置 ----
    train_cfg = dict(
        max_epochs={NUM_EPOCH},
        val_interval=999999  # 禁用验证
    )
    
    # ---- 优化器配置 ----
    optim_wrapper = dict(
        optimizer=dict(
            type='AdamW', 
            lr=1e-4,
            weight_decay=0.01
        )
    )
    
    # ---- 学习率调度 ----
    param_scheduler = [
        dict(
            type='LinearLR',
            start_factor=0.1,
            by_epoch=False,
            begin=0,
            end=500),
        dict(
            type='MultiStepLR',
            by_epoch=True,
            milestones=[10, 16],
            gamma=0.1)
    ]
    
    # ---- 简化的Hooks配置 ----
    default_hooks = dict(
        runtime_info=dict(type='RuntimeInfoHook'),
        timer=dict(type='IterTimerHook'),
        logger=dict(
            type='LoggerHook', 
            interval=100
        ),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(
            type='CheckpointHook',
            interval=5,  # 每5个epoch保存
            max_keep_ckpts=4,
            save_last=True,
            save_best=None  # 不使用验证指标
        ),
        sampler_seed=dict(type='DistSamplerSeedHook'),
    )

    # ---- 工作目录 ----
    work_dir = '{WORK_DIR}'
    
    # ---- 环境配置 ----
    env_cfg = dict(
        cudnn_benchmark=False,
        mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
        dist_cfg=dict(backend='nccl'),
    )
    
    # ---- 日志配置 ----
    log_processor = dict(
        type='LogProcessor', 
        window_size=50, 
        by_epoch=True
    )
    
    log_level = 'INFO'
    load_from = None
    resume = False
    """)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_OUT.write_text(cfg_text)
    print("✅ 简化训练配置已生成:", CONFIG_OUT)

def train():
    """启动简化训练"""
    print("🚀 开始简化训练（无验证集）...")
    
    try:
        cfg = Config.fromfile(CONFIG_OUT)
        register_all_modules(init_default_scope=False)
        runner = Runner.from_cfg(cfg)
        runner.train()
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return False

def copy_best():
    """复制最新的模型文件"""
    
    # 查找最新的checkpoint
    possible_files = []
    possible_files.extend(WORK_DIR.glob("epoch_*.pth"))
    possible_files.extend(WORK_DIR.glob("latest.pth"))
    
    if not possible_files:
        print("⚠️  没找到模型文件")
        return
    
    # 选择最新的文件
    latest = max(possible_files, key=lambda p: p.stat().st_mtime)
    
    try:
        shutil.copy(latest, CKPT_BEST)
        print("✅ 模型已复制到:", CKPT_BEST)
        print("   源文件:", latest)
    except Exception as e:
        print(f"⚠️  复制模型文件时出错: {e}")

def main():
    print("🦜 简化版鸟类关键点检测器训练")
    print("=" * 60)
    print(f"训练轮数: {NUM_EPOCH}")
    print("策略: 无验证集，避免COCO格式问题")
    print()
    
    # 生成配置
    make_simple_cfg()
    
    # 开始训练
    success = train()
    
    if success:
        copy_best()
        print(f"\n🎉 训练完成!")
        print(f"模型保存在: {CKPT_BEST}")
        print("现在可以用这个模型进行评估和演示")
    else:
        print(f"\n❌ 训练失败")
        print("请检查错误信息和日志")

if __name__ == "__main__":
    main()