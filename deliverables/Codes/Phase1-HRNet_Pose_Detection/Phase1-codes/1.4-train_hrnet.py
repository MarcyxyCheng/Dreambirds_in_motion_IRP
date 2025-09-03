#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1-4: train the hrnet model
-------------------
Train an HRNet detector for CUB-15 keypoints based on MMPose.
Assumes data has been prepared by 1.2.1; focus on the training process.
Enhance error handling, logging, and debugging.
"""

from pathlib import Path
import json, yaml, textwrap, shutil, traceback, datetime, time
import numpy as np
from mmengine import Config
from mmengine.runner import Runner
from mmpose.utils import register_all_modules

# Paths
WORK_DIR = Path("workdir/hrnet15_cub")
COCO_DIR = WORK_DIR / "coco_format"
CONFIG_OUT = WORK_DIR / "cub15_hrnet.py"
CKPT_BEST = WORK_DIR / "bird15_detector.pth"
LOGBOOK_FILE = WORK_DIR / "training_logbook.md"

# Training parameters
NUM_EPOCH = 60
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# Debug settings
DEBUG_MODE = False  # Use test subset for quick debugging
TEST_SUBSET = False  # Use _test.json files

class TrainingLogger:
    def __init__(self, logbook_path):
        self.logbook_path = Path(logbook_path)
        self.start_time = None
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logbook_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_session()
    
    def _init_session(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        header = f"""
## Training Session - {self.session_id}

**Start Time**: {timestamp}
**Num Epochs**: {NUM_EPOCH}
**Batch Size**: {BATCH_SIZE}
**Learning Rate**: {LEARNING_RATE}
**Debug Mode**: {'ON' if DEBUG_MODE else 'OFF'}
**Use Test Subset**: {'YES' if TEST_SUBSET else 'NO'}

### Training Process

"""
        
        with open(self.logbook_path, 'a', encoding='utf-8') as f:
            f.write(header)
    
    def log_step(self, step_name, message, status="IN-PROGRESS"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"{timestamp} {status} **{step_name}**: {message}\n\n"
        
        with open(self.logbook_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        print(f"{status} {step_name}: {message}")
    
    def log_error(self, step_name, error, full_traceback=None):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"{timestamp} ERROR **{step_name}**: {error}\n\n"
        
        if full_traceback:
            log_entry += f"```\n{full_traceback}\n```\n\n"
        
        with open(self.logbook_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        print(f"ERROR {step_name}: {error}")
    
    def log_training_start(self):
        self.start_time = time.time()
        self.log_step("Training Start", f"Start training CUB-15 HRNet (session ID: {self.session_id})", "START")
    
    def log_training_complete(self, success=True, final_model_path=None):
        if self.start_time:
            duration = time.time() - self.start_time
            duration_str = f"{duration//3600:.0f}h {(duration%3600)//60:.0f}m {duration%60:.0f}s"
        else:
            duration_str = "UNKNOWN"
        
        if success:
            self.log_step("Training Complete", f"Training finished successfully, duration {duration_str}", "DONE")
            if final_model_path:
                self.log_step("Model Saved", f"Final model: {final_model_path}", "SAVED")
        else:
            self.log_step("Training Failed", f"Training failed, duration {duration_str}", "FAILED")

logger = TrainingLogger(LOGBOOK_FILE)

def check_data_ready():
    """Check if data is ready"""
    logger.log_step("Data Check", "Checking COCO-format data availability")
    
    # Determine filenames
    train_file = "train_test.json" if (DEBUG_MODE and TEST_SUBSET) else "train.json"
    val_file = "val_test.json" if (DEBUG_MODE and TEST_SUBSET) else "val.json"
    
    train_path = COCO_DIR / train_file
    val_path = COCO_DIR / val_file
    
    missing_files = []
    if not train_path.exists():
        missing_files.append(f"Train data: {train_path}")
    if not val_path.exists():
        missing_files.append(f"Val data: {val_path}")
    
    if missing_files:
        logger.log_step("Data Check", f"Missing data files: {missing_files}", "ERROR")
        logger.log_step("Hint", "Please run 1.2.1_prepare_data.py to generate data", "HINT")
        return False, None, None
    
    # Validate content
    try:
        with open(train_path) as f:
            train_data = json.load(f)
        with open(val_path) as f:
            val_data = json.load(f)
        
        train_stats = {
            "Num Train Images": len(train_data['images']),
            "Num Train Annotations": len(train_data['annotations']),
            "Avg Visible Keypoints": f"{np.mean([ann['num_keypoints'] for ann in train_data['annotations']]):.1f}"
        }
        
        val_stats = {
            "Num Val Images": len(val_data['images']),
            "Num Val Annotations": len(val_data['annotations'])
        }
        
        logger.log_step("Data Stats", f"Train: {train_stats}, Val: {val_stats}", "STATS")
        logger.log_step("Data Check", "Data files validated", "OK")
        
        return True, train_file, val_file
        
    except Exception as e:
        logger.log_error("Data Check", f"Data validation failed: {e}")
        return False, None, None

def create_mmpose_config(train_file, val_file):
    """Generate MMPose training config"""
    logger.log_step("Config Generation", "Generating MMPose training config")
    
    try:
        # Load skeleton config
        with open("skeleton.yaml") as f:
            skeleton_config = yaml.safe_load(f)
    except Exception as e:
        logger.log_error("Config Generation", f"Failed to load skeleton.yaml: {e}")
        return False
    
    # Build keypoint_info dict in Python to ensure it is a dict (not a list)
    kp_info = {
        i: dict(
            name=kp['name'],
            id=i,
            color=[255 - i*15, i*10, i*20],
            type='',
            swap=kp.get('swap', '')
        )
        for i, kp in enumerate(skeleton_config['keypoints'])
    }
    
    num_keypoints = len(kp_info)
    keypoint_names = [kp_info[i]['name'] for i in range(num_keypoints)]
    logger.log_step("Skeleton Check", f"Keypoints: {keypoint_names}")
    
    # Hard-coded raw connections using indices
    raw_connections = [
        [1, 4], [4, 9], [9, 0],
        [4, 5], [5, 14],
        [4, 6], [4, 10],
        [0, 3], [3, 2],
        [0, 8], [0, 12],
        [0, 13],
        [2, 7], [2, 11],
    ]
    
    # Filter out-of-range connections and convert to names
    valid_connections = []
    invalid_connections = []
    
    for connection in raw_connections:
        if all(0 <= idx < num_keypoints for idx in connection):
            name_connection = (keypoint_names[connection[0]], keypoint_names[connection[1]])
            valid_connections.append(name_connection)
        else:
            invalid_connections.append(connection)
    
    if invalid_connections:
        logger.log_step("Skeleton Fix", f"Removed invalid connections: {invalid_connections}")
    
    logger.log_step("Skeleton Verify", f"Valid connections: {len(valid_connections)}/{len(raw_connections)}")
    
    skeleton_info = {
        i: dict(
            link=connection,  # tuple with keypoint names
            id=i,
            color=[255 - i*10, i*15, i*25]
        )
        for i, connection in enumerate(valid_connections)
    }
    
    dataset_info = dict(
        dataset_name='cub15',
        paper_info=dict(
            author='CUB-200-2011',
            title='CUB-200-2011 Bird Dataset',
            container='Caltech-UCSD Birds 200',
            year='2011',
        ),
        keypoint_info=kp_info,
        skeleton_info=skeleton_info,
        joint_weights=[1.0] * len(kp_info),
        num_keypoints=len(kp_info),
        sigmas=[0.025] * len(kp_info),
    )
    
    cfg_text = textwrap.dedent(f"""
    # CUB-15 HRNet config - adapted from AP-10K
    _base_ = [
        'mmpose::animal_2d_keypoint/topdown_heatmap/ap10k/td-hm_hrnet-w32_8xb64-210e_ap10k-256x256.py',
    ]

    auto_scale_lr = dict(base_batch_size=512)
    backend_args = dict(backend='local')
    codec = dict(
        _scope_='mmpose',
        heatmap_size=(64, 64),
        input_size=(256, 256),
        sigma=2,
        type='MSRAHeatmap'
    )
    custom_hooks = [dict(_scope_='mmpose', type='SyncBuffersHook')]
    data_mode = 'topdown'

    dataset_info = {repr(dataset_info)}

    dataset_type = 'CocoDataset'

    default_hooks = dict(
        runtime_info=dict(type='RuntimeInfoHook'),
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=20),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(
            type='CheckpointHook',
            interval=3,
            max_keep_ckpts=5,
            save_last=True,
            save_best='coco/AP',
            rule='greater'
        ),
        sampler_seed=dict(type='DistSamplerSeedHook'),
    )

    env_cfg = dict(
        cudnn_benchmark=False,
        mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
        dist_cfg=dict(backend='nccl'),
    )

    log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
    log_level = 'INFO'
    load_from = None
    resume = False

    model = dict(
        head=dict(out_channels={len(kp_info)}),
    )

    optim_wrapper = dict(
        optimizer=dict(type='AdamW', lr={LEARNING_RATE}, weight_decay=0.01)
    )

    param_scheduler = [
        dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
        dict(type='MultiStepLR', by_epoch=True, milestones=[12, 16], gamma=0.1)
    ]

    train_cfg = dict(max_epochs={NUM_EPOCH}, val_interval=3)
    val_cfg = dict()

    train_dataloader = dict(
        batch_size={BATCH_SIZE},
        dataset=dict(
            type='CocoDataset',
            data_root='.',
            ann_file='workdir/hrnet15_cub/coco_format/{train_file}',
            data_prefix=dict(img=''),
            metainfo=dataset_info,
            pipeline=[
                dict(type='LoadImage'),
                dict(type='GetBBoxCenterScale'),
                dict(type='RandomFlip', prob=0.5, direction='horizontal'),
                dict(type='RandomBBoxTransform', scale_factor=[0.8, 1.2], rotate_factor=20),
                dict(type='TopdownAffine', input_size=(256, 256)),
                dict(type='GenerateTarget', encoder=codec),
                dict(type='PackPoseInputs')
            ]
        )
    )

    # Note: default_hooks is already defined above.

    val_dataloader = dict(
        batch_size={BATCH_SIZE},
        dataset=dict(
            type='CocoDataset',
            data_root='.',
            ann_file='workdir/hrnet15_cub/coco_format/{val_file}',
            data_prefix=dict(img=''),
            metainfo=dataset_info,
        )
    )

    val_evaluator = dict(
        type='CocoMetric',
        ann_file='workdir/hrnet15_cub/coco_format/{val_file}'
    )

    work_dir = '{WORK_DIR}'
    """)

    try:
        CONFIG_OUT.write_text(cfg_text)
        logger.log_step("Config Generation", f"Config written: {CONFIG_OUT}", "OK")
        return True
    except Exception as e:
        logger.log_error("Config Generation", f"Failed to write config: {e}")
        return False

def test_data_loading():
    """Simplified data loading test"""
    logger.log_step("Data Load Test", "Skip manual data loading test and go directly to training")
    return True

def train_model():
    """Start training"""
    logger.log_training_start()
    
    try:
        cfg = Config.fromfile(CONFIG_OUT)
        try:
            register_all_modules(init_default_scope=False)
        except KeyError as e:
            if "already registered" in str(e):
                print(f"Modules already registered, skip: {e}")
            else:
                raise e
        # Debug prints to ensure dataset_info types are dict
        print("-> keypoint_info type:", type(cfg.dataset_info['keypoint_info']))
        print("-> skeleton_info type:", type(cfg.dataset_info['skeleton_info']))
        
        runner = Runner.from_cfg(cfg)
        runner.train()
        logger.log_step("Training Run", "Training loop finished normally", "OK")
        return True
        
    except KeyboardInterrupt:
        logger.log_step("Training Interrupted", "User interrupted training", "INTERRUPTED")
        return False
        
    except Exception as e:
        logger.log_error("Training Run", f"Error during training: {e}", traceback.format_exc())
        return False

def find_and_copy_best_model():
    """Find and copy the best model"""
    logger.log_step("Model Collection", "Searching for model files")
    possible_files = []
    if WORK_DIR.exists():
        possible_files.extend(WORK_DIR.glob("best_*.pth"))
        possible_files.extend(WORK_DIR.glob("epoch_*.pth"))
        possible_files.extend(WORK_DIR.glob("latest.pth"))
    
    if not possible_files:
        logger.log_step("Model Collection", "No model files found", "WARN")
        return False
    
    best_files = [f for f in possible_files if 'best_' in f.name]
    if best_files:
        latest_model = max(best_files, key=lambda p: p.stat().st_mtime)
        model_type = "Best model"
    else:
        latest_model = max(possible_files, key=lambda p: p.stat().st_mtime)
        model_type = "Latest model"
    
    try:
        shutil.copy(latest_model, CKPT_BEST)
        logger.log_step("Model Collection", f"{model_type} copied: {latest_model.name} → {CKPT_BEST}", "OK")
        return True
    except Exception as e:
        logger.log_error("Model Collection", f"Failed to copy model: {e}")
        return False

def create_training_summary():
    """Create training summary"""
    summary_file = WORK_DIR / "training_summary.json"
    summary = {
        "session_id": logger.session_id,
        "completion_time": datetime.datetime.now().isoformat(),
        "parameters": {
            "epochs": NUM_EPOCH,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "debug_mode": DEBUG_MODE,
            "test_subset": TEST_SUBSET
        },
        "output_files": {
            "config": str(CONFIG_OUT),
            "final_model": str(CKPT_BEST),
            "logbook": str(LOGBOOK_FILE)
        }
    }
    
    try:
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.log_step("Training Summary", f"Summary saved: {summary_file}", "OK")
    except Exception as e:
        logger.log_error("Training Summary", f"Failed to save summary: {e}")

def main():
    print("CUB-15 HRNet Training Script")
    print("=" * 60)
    
    data_ready, train_file, val_file = check_data_ready()
    if not data_ready:
        logger.log_step("Abort", "Data check failed", "ERROR")
        return
    
    if not create_mmpose_config(train_file, val_file):
        logger.log_step("Abort", "Config generation failed", "ERROR")
        return
    
    if DEBUG_MODE and not test_data_loading():
        logger.log_step("Abort", "Data loading test failed", "ERROR")
        return
    
    training_success = train_model()
    
    if training_success:
        model_copied = find_and_copy_best_model()
        logger.log_training_complete(success=True, final_model_path=CKPT_BEST if model_copied else None)
    else:
        logger.log_training_complete(success=False)
    
    create_training_summary()
    
    print(f"\nDetailed training log: {LOGBOOK_FILE}")
    print(f"Working directory: {WORK_DIR}")
    
    if training_success:
        print("Training finished.")
        if CKPT_BEST.exists():
            print(f"Final model: {CKPT_BEST}")
            print("You can now run the evaluation script to test the model.")
        else:
            print("Training finished but no model file found.")
    else:
        print("Training failed. Please check logs for details.")
        
    if DEBUG_MODE:
        print("\nDebug mode is ON")
        print("For full training, set DEBUG_MODE=False and re-run 1.2.1 to prepare full data")

if __name__ == "__main__":
    main()
