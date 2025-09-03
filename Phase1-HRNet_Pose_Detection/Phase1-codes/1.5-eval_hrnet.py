#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1-5: evaluate hrnet output
------------------
Evaluation script for the CUB-15 HRNet keypoint detector.
Includes PCK metrics, visualizations, error analysis, and logging.
"""

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import yaml
from tqdm import tqdm
import datetime
import warnings
warnings.filterwarnings('ignore')

# MMPose imports
from mmengine import Config
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

# Paths
WORK_DIR = Path("workdir/hrnet15_cub")
MODEL_PATH = WORK_DIR / "bird15_detector.pth"
CONFIG_PATH = WORK_DIR / "cub15_hrnet.py"
COCO_DIR = WORK_DIR / "coco_format"
EVAL_DIR = WORK_DIR / "evaluation_results"
LOGBOOK_FILE = WORK_DIR / "evaluation_logbook.md"

# Create output directory
EVAL_DIR.mkdir(parents=True, exist_ok=True)

class CUBEvaluator:
    def __init__(self):
        self.model = None
        self.keypoint_names = []
        self.results = {}
        self.ground_truth = {}
        self.predictions = {}
        self.failed_cases = []
        self.lr_confusion_stats = {}  # left/right confusion stats
        
        # Initialize logbook
        self.init_logbook()
        
    def init_logbook(self):
        """Initialize the evaluation logbook markdown file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        header = f"""
# CUB-15 Model Evaluation Report

**Evaluation Time**: {timestamp}  
**Session ID**: {session_id}  
**Model Path**: {MODEL_PATH}  
**Config File**: {CONFIG_PATH}

## Evaluation Steps

"""
        with open(LOGBOOK_FILE, 'w', encoding='utf-8') as f:
            f.write(header)
    
    def log_step(self, step_name, message, status="IN-PROGRESS"):
        """Append a step entry to the logbook and print to console."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"{timestamp} {status} **{step_name}**: {message}\n\n"
        
        with open(LOGBOOK_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        print(f"{status} {step_name}: {message}")
    
    def load_model(self):
        """Load the trained HRNet model and read keypoint names."""
        self.log_step("Model Loading", "Start loading HRNet model")
        
        try:
            print("Step 1: Register MMPose modules...")
            register_all_modules(init_default_scope=False)
            print("Modules registered")
            
            print("Step 2: Initialize model...")
            print(f"   Config path: {CONFIG_PATH} (type: {type(CONFIG_PATH)})")
            print(f"   Model path: {MODEL_PATH} (type: {type(MODEL_PATH)})")
            
            # Ensure using string paths
            self.model = init_model(str(CONFIG_PATH), str(MODEL_PATH), device='cuda:0')
            print("Model initialized")
            
            print("Step 3: Load skeleton config...")
            # Read keypoint names
            with open("skeleton.yaml") as f:
                skeleton = yaml.safe_load(f)
            self.keypoint_names = [kp['name'] for kp in skeleton['keypoints']]
            print(f"Skeleton loaded, keypoints: {len(self.keypoint_names)}")
            
            self.log_step("Model Loading", f"Model loaded, {len(self.keypoint_names)} keypoints supported", "OK")
            return True
            
        except Exception as e:
            print(f"Exception: {e}")
            print(f"Type: {type(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            self.log_step("Model Loading", f"Failed to load model: {e}", "ERROR")
            return False
    
    def load_ground_truth(self, split='val'):
        """Load COCO-style annotations for a given split."""
        self.log_step("Data Loading", f"Load annotations for split: {split}")
        
        json_file = COCO_DIR / f"{split}.json"
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Build image_id → metadata map
            for img_info in data['images']:
                img_id = img_info['id']
                self.ground_truth[img_id] = {
                    'file_name': img_info['file_name'],
                    'width': img_info['width'],
                    'height': img_info['height']
                }
            
            # Attach per-image annotations
            for ann in data['annotations']:
                img_id = ann['image_id']
                if img_id in self.ground_truth:
                    keypoints = np.array(ann['keypoints']).reshape(-1, 3)
                    self.ground_truth[img_id]['keypoints'] = keypoints
                    self.ground_truth[img_id]['bbox'] = ann['bbox']
                    self.ground_truth[img_id]['num_keypoints'] = ann['num_keypoints']
            
            self.log_step("Data Loading", f"Loaded annotations for {len(self.ground_truth)} images", "OK")
            return True
            
        except Exception as e:
            self.log_step("Data Loading", f"Failed to load annotations: {e}", "ERROR")
            return False
    
    def run_inference(self):
        """Run inference on a small subset (first 5 images) for evaluation."""
        self.log_step("Inference", "Start batch inference")
        
        success_count = 0
        total_count = len(self.ground_truth)
        
        # Process only first 5 images for quick debug
        debug_limit = 5
        processed_count = 0
        
        for img_id, gt_data in self.ground_truth.items():
            if processed_count >= debug_limit:
                break
            processed_count += 1
            
            img_path = Path(gt_data['file_name'])
            print(f"\nProcessing image #{processed_count}: {img_path}")
            
            if not img_path.exists():
                print(f"Image file not found: {img_path}")
                self.failed_cases.append({
                    'img_id': img_id,
                    'reason': 'Image file not found',
                    'file_name': str(img_path)
                })
                continue
            
            try:
                print(f"   Path: {img_path}")
                print(f"   Size: {gt_data['width']}x{gt_data['height']}")
                
                # Check readability
                test_img = cv2.imread(str(img_path))
                if test_img is None:
                    print(f"Unable to read image")
                    self.failed_cases.append({
                        'img_id': img_id,
                        'reason': 'Image unreadable',
                        'file_name': str(img_path)
                    })
                    continue
                
                print(f"   Image loaded: {test_img.shape}")
                
                # Run inference
                print(f"   Inference...")
                results = inference_topdown(self.model, str(img_path))
                print(f"   Result type: {type(results)}")
                print(f"   Result length: {len(results) if results else 0}")
                
                if results and len(results) > 0:
                    print(f"   Got predictions")
                    
                    result = results[0]
                    print(f"   First result type: {type(result)}")
                    
                    if hasattr(result, 'pred_instances'):
                        pred_instances = result.pred_instances
                        print(f"   pred_instances type: {type(pred_instances)}")
                        
                        if hasattr(pred_instances, 'keypoints'):
                            keypoints = pred_instances.keypoints
                            print(f"   keypoints shape: {keypoints.shape if hasattr(keypoints, 'shape') else 'N/A'}")
                            print(f"   keypoints type: {type(keypoints)}")
                            
                            if len(keypoints) > 0:
                                # Convert to numpy if needed
                                if hasattr(keypoints, 'cpu'):
                                    kp_data = keypoints[0].cpu().numpy()
                                elif isinstance(keypoints, np.ndarray):
                                    kp_data = keypoints[0]
                                else:
                                    kp_data = keypoints[0]
                                
                                print(f"   extracted keypoints shape: {kp_data.shape}")
                                
                                # Scores
                                pred_scores = None
                                if hasattr(pred_instances, 'keypoint_scores'):
                                    scores = pred_instances.keypoint_scores
                                    print(f"   scores type: {type(scores)}")
                                    
                                    if hasattr(scores, 'cpu'):
                                        pred_scores = scores[0].cpu().numpy()
                                    elif isinstance(scores, np.ndarray):
                                        pred_scores = scores[0]
                                    else:
                                        pred_scores = scores[0]
                                    
                                    print(f"   scores shape: {pred_scores.shape}")
                                    print(f"   score range: [{pred_scores.min():.3f}, {pred_scores.max():.3f}]")
                                else:
                                    print(f"   No keypoint_scores; using default scores")
                                    pred_scores = np.ones(15) * 0.5  # default confidences
                                
                                # Ensure expected shape
                                if kp_data.shape == (15, 2):
                                    pred_keypoints = kp_data
                                else:
                                    print(f"   Unexpected keypoint shape: {kp_data.shape}")
                                    pred_keypoints = kp_data.reshape(15, 2) if kp_data.size >= 30 else None
                                
                                if pred_keypoints is not None:
                                    # Convert to COCO (15,3): [x, y, visibility/conf]
                                    pred_kpts_coco = np.zeros((15, 3))
                                    pred_kpts_coco[:, :2] = pred_keypoints
                                    pred_kpts_coco[:, 2] = pred_scores
                                    
                                    # Predicted bbox (if any)
                                    bbox = None
                                    if hasattr(pred_instances, 'bboxes') and len(pred_instances.bboxes) > 0:
                                        bbox_data = pred_instances.bboxes[0]
                                        if hasattr(bbox_data, 'cpu'):
                                            bbox = bbox_data.cpu().numpy()
                                        else:
                                            bbox = bbox_data
                                    
                                    self.predictions[img_id] = {
                                        'keypoints': pred_kpts_coco,
                                        'scores': pred_scores,
                                        'bbox': bbox
                                    }
                                    success_count += 1
                                    print(f"   Done: {img_path.name}")
                                else:
                                    print(f"   Failed to process keypoints")
                                    self.failed_cases.append({
                                        'img_id': img_id,
                                        'reason': 'Keypoint postprocess failed',
                                        'file_name': str(img_path)
                                    })
                            else:
                                print(f"   No keypoints detected")
                                self.failed_cases.append({
                                    'img_id': img_id,
                                    'reason': 'No keypoints detected',
                                    'file_name': str(img_path)
                                })
                        else:
                            print(f"   pred_instances has no `keypoints`")
                            self.failed_cases.append({
                                'img_id': img_id,
                                'reason': 'Missing keypoints field',
                                'file_name': str(img_path)
                            })
                    else:
                        print(f"   Result has no `pred_instances`")
                        self.failed_cases.append({
                            'img_id': img_id,
                            'reason': 'Missing pred_instances',
                            'file_name': str(img_path)
                        })
                else:
                    print(f"   Empty inference result")
                    self.failed_cases.append({
                        'img_id': img_id,
                        'reason': 'Empty result',
                        'file_name': str(img_path)
                    })
                    
            except Exception as e:
                print(f"   Inference error: {e}")
                import traceback
                print(f"   Traceback:\n{traceback.format_exc()}")
                self.failed_cases.append({
                    'img_id': img_id,
                    'reason': f'Inference exception: {e}',
                    'file_name': str(img_path)
                })
        
        print(f"\nInference summary:")
        print(f"   Success: {success_count}")
        print(f"   Failed: {len(self.failed_cases)}")
        print(f"   Processed: {processed_count}")
        
        self.log_step("Inference", f"Done: {success_count}/{processed_count} success, {len(self.failed_cases)} failed", 
                     "OK" if success_count > 0 else "WARN")
        return success_count > 0
    
    def calculate_pck(self, threshold=0.1):
        """Compute PCK (Percentage of Correct Keypoints) at a given threshold."""
        pck_per_point = np.zeros(15)
        pck_counts = np.zeros(15)
        total_pck = 0
        total_valid = 0
        
        for img_id in self.predictions:
            if img_id not in self.ground_truth:
                continue
                
            gt_kpts = self.ground_truth[img_id]['keypoints']
            pred_kpts = self.predictions[img_id]['keypoints']
            
            # Normalization factor: bbox diagonal
            bbox = self.ground_truth[img_id]['bbox']
            norm = np.sqrt(bbox[2]**2 + bbox[3]**2)
            
            for i in range(15):
                if gt_kpts[i, 2] > 0:  # only visible keypoints
                    dist = np.linalg.norm(pred_kpts[i, :2] - gt_kpts[i, :2])
                    normalized_dist = dist / norm
                    
                    if normalized_dist <= threshold:
                        pck_per_point[i] += 1
                        total_pck += 1
                    
                    pck_counts[i] += 1
                    total_valid += 1
        
        # Per-keypoint PCK
        pck_results = {}
        for i in range(15):
            if pck_counts[i] > 0:
                pck_results[self.keypoint_names[i]] = pck_per_point[i] / pck_counts[i]
            else:
                pck_results[self.keypoint_names[i]] = 0.0
        
        # Overall PCK
        overall_pck = total_pck / total_valid if total_valid > 0 else 0.0
        
        return overall_pck, pck_results
    
    def evaluate_all_metrics(self):
        """Compute all metrics (PCK at multiple thresholds) and L/R confusion."""
        self.log_step("Metric Computation", "Compute PCK at multiple thresholds")
        
        thresholds = [0.05, 0.1, 0.15, 0.2]
        pck_results = {}
        
        for thresh in thresholds:
            overall_pck, per_point_pck = self.calculate_pck(thresh)
            pck_results[f'PCK@{thresh}'] = {
                'overall': overall_pck,
                'per_point': per_point_pck
            }
        
        self.results['pck'] = pck_results

        # Left/right confusion analysis
        self.analyze_lr_confusion()
        
        # Log summary
        for thresh in thresholds:
            overall = pck_results[f'PCK@{thresh}']['overall']
            self.log_step("PCK Summary", f"PCK@{thresh}: {overall:.3f} ({overall*100:.1f}%)", "STATS")
        
        return pck_results
    
    def analyze_errors(self):
        """Aggregate normalized error statistics and worst cases."""
        self.log_step("Error Analysis", "Analyze error distributions")
        
        error_stats = {
            'per_keypoint_error': {},
            'distance_stats': {},
            'worst_cases': []
        }
        
        all_errors = []
        keypoint_errors = {name: [] for name in self.keypoint_names}
        
        for img_id in self.predictions:
            if img_id not in self.ground_truth:
                continue
                
            gt_kpts = self.ground_truth[img_id]['keypoints']
            pred_kpts = self.predictions[img_id]['keypoints']
            bbox = self.ground_truth[img_id]['bbox']
            norm = np.sqrt(bbox[2]**2 + bbox[3]**2)
            
            img_errors = []
            for i in range(15):
                if gt_kpts[i, 2] > 0:  # only visible keypoints
                    error = np.linalg.norm(pred_kpts[i, :2] - gt_kpts[i, :2])
                    normalized_error = error / norm
                    
                    all_errors.append(normalized_error)
                    keypoint_errors[self.keypoint_names[i]].append(normalized_error)
                    img_errors.append(normalized_error)
            
            if img_errors:
                avg_error = np.mean(img_errors)
                error_stats['worst_cases'].append({
                    'img_id': img_id,
                    'file_name': self.ground_truth[img_id]['file_name'],
                    'avg_error': avg_error,
                    'max_error': np.max(img_errors)
                })
        
        # Per-keypoint stats
        for kp_name, errors in keypoint_errors.items():
            if errors:
                error_stats['per_keypoint_error'][kp_name] = {
                    'mean': np.mean(errors),
                    'std': np.std(errors),
                    'median': np.median(errors),
                    'count': len(errors)
                }
        
        # Global stats
        if all_errors:
            error_stats['distance_stats'] = {
                'mean_error': np.mean(all_errors),
                'std_error': np.std(all_errors),
                'median_error': np.median(all_errors),
                'max_error': np.max(all_errors),
                'min_error': np.min(all_errors)
            }
        
        # Sort worst cases descending by avg error
        error_stats['worst_cases'].sort(key=lambda x: x['avg_error'], reverse=True)
        
        self.results['errors'] = error_stats
        
        if all_errors:
            mean_err = np.mean(all_errors)
            self.log_step("Error Summary", f"Mean normalized error: {mean_err:.3f}", "STATS")
        
        return error_stats
    
    def analyze_lr_confusion(self):
        """Analyze left/right confusion for wings and legs."""
        self.log_step("Left/Right Confusion", "Analyze L/R confusion for wings and legs")
        
        # Keypoint indices (based on your skeleton definition)
        left_wing_idx = 8    # left_wing
        right_wing_idx = 12  # right_wing  
        left_leg_idx = 7     # left_leg
        right_leg_idx = 11   # right_leg
        back_idx = 0         # back (midline reference)
        
        wing_distances = []
        leg_distances = []
        wing_confusion_cases = []
        leg_confusion_cases = []
        
        for img_id in self.predictions:
            if img_id not in self.ground_truth:
                continue
                
            gt_kpts = self.ground_truth[img_id]['keypoints']
            pred_kpts = self.predictions[img_id]['keypoints']
            
            # Wings
            if (gt_kpts[left_wing_idx, 2] > 0 and gt_kpts[right_wing_idx, 2] > 0 and 
                pred_kpts[left_wing_idx, 2] > 0.1 and pred_kpts[right_wing_idx, 2] > 0.1):
                
                wing_dist = np.linalg.norm(pred_kpts[left_wing_idx, :2] - pred_kpts[right_wing_idx, :2])
                wing_distances.append(wing_dist)
                
                bbox = self.ground_truth[img_id]['bbox']
                bbox_diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
                normalized_wing_dist = wing_dist / bbox_diag
                
                # Consider "confused" if too close
                if normalized_wing_dist < 0.1:
                    wing_confusion_cases.append({
                        'img_id': img_id,
                        'distance': wing_dist,
                        'normalized_distance': normalized_wing_dist,
                        'file_name': self.ground_truth[img_id]['file_name']
                    })
            
            # Legs
            if (gt_kpts[left_leg_idx, 2] > 0 and gt_kpts[right_leg_idx, 2] > 0 and 
                pred_kpts[left_leg_idx, 2] > 0.1 and pred_kpts[right_leg_idx, 2] > 0.1):
                
                leg_dist = np.linalg.norm(pred_kpts[left_leg_idx, :2] - pred_kpts[right_leg_idx, :2])
                leg_distances.append(leg_dist)
                
                bbox = self.ground_truth[img_id]['bbox']
                bbox_diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
                normalized_leg_dist = leg_dist / bbox_diag
                
                if normalized_leg_dist < 0.1:
                    leg_confusion_cases.append({
                        'img_id': img_id,
                        'distance': leg_dist,
                        'normalized_distance': normalized_leg_dist,
                        'file_name': self.ground_truth[img_id]['file_name']
                    })
        
        # Aggregate stats
        self.lr_confusion_stats = {
            'wing_confusion': {
                'total_cases': len(wing_distances),
                'confused_cases': len(wing_confusion_cases),
                'confusion_rate': len(wing_confusion_cases) / len(wing_distances) if wing_distances else 0,
                'avg_distance': np.mean(wing_distances) if wing_distances else 0,
                'confused_files': wing_confusion_cases
            },
            'leg_confusion': {
                'total_cases': len(leg_distances),
                'confused_cases': len(leg_confusion_cases),
                'confusion_rate': len(leg_confusion_cases) / len(leg_distances) if leg_distances else 0,
                'avg_distance': np.mean(leg_distances) if leg_distances else 0,
                'confused_files': leg_confusion_cases
            }
        }
        
        # Log confusion rates
        wing_rate = self.lr_confusion_stats['wing_confusion']['confusion_rate']
        leg_rate = self.lr_confusion_stats['leg_confusion']['confusion_rate']
        self.log_step("L/R Confusion Summary", f"Wing confusion: {wing_rate:.1%}, Leg confusion: {leg_rate:.1%}", "STATS")
        
        return self.lr_confusion_stats
    
    def create_visualizations(self):
        """Generate visualization figures."""
        self.log_step("Visualization", "Generate plots")
        
        # 1) Overall PCK at different thresholds
        self.plot_pck_comparison()
        
        # 2) Per-keypoint PCK@0.1
        self.plot_per_keypoint_pck()
        
        # 3) Per-keypoint normalized error (mean)
        self.plot_error_distribution()
        
        # 4) Worst qualitative cases
        self.visualize_worst_cases()

        # 5) Left/Right confusion
        self.plot_lr_confusion()        
        
        self.log_step("Visualization", f"Figures saved to: {EVAL_DIR}", "OK")
    
    def plot_pck_comparison(self):
        """Bar chart: overall PCK at different thresholds."""
        thresholds = [0.05, 0.1, 0.15, 0.2]
        pck_values = [self.results['pck'][f'PCK@{t}']['overall'] for t in thresholds]
        
        plt.figure(figsize=(10, 6))
        plt.bar([f'PCK@{t}' for t in thresholds], pck_values, color='skyblue', alpha=0.7)
        plt.title('CUB-15 Keypoint Detection — PCK vs Threshold', fontsize=14, fontweight='bold')
        plt.ylabel('PCK', fontsize=12)
        plt.xlabel('Threshold (τ of bbox diagonal)', fontsize=12)
        
        for i, v in enumerate(pck_values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.ylim(0, max(pck_values) * 1.2 if pck_values else 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(EVAL_DIR / 'pck_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_per_keypoint_pck(self):
        """Horizontal bars: PCK@0.1 for each keypoint."""
        pck_01 = self.results['pck']['PCK@0.1']['per_point']
        
        keypoints = list(pck_01.keys())
        values = list(pck_01.values())
        
        plt.figure(figsize=(12, 8))
        plt.barh(keypoints, values, color='lightcoral', alpha=0.7)
        plt.title('Per-Keypoint PCK@0.1', fontsize=14, fontweight='bold')
        plt.xlabel('PCK@0.1', fontsize=12)
        plt.ylabel('Keypoint', fontsize=12)
        
        for i, v in enumerate(values):
            plt.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')
        
        plt.xlim(0, max(values) * 1.2 if values else 1)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(EVAL_DIR / 'per_keypoint_pck.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_distribution(self):
        """Horizontal bars: mean normalized error per keypoint."""
        if 'errors' not in self.results:
            return
            
        error_data = []
        keypoint_labels = []
        
        for kp_name, error_info in self.results['errors']['per_keypoint_error'].items():
            error_data.append(error_info['mean'])
            keypoint_labels.append(kp_name)
        
        if not error_data:
            return
            
        plt.figure(figsize=(12, 8))
        plt.barh(keypoint_labels, error_data, color='orange', alpha=0.7)
        plt.title('Per-Keypoint Normalized Error (mean)', fontsize=14, fontweight='bold')
        plt.xlabel('Normalized Error (by bbox diagonal)', fontsize=12)
        plt.ylabel('Keypoint', fontsize=12)
        
        for i, v in enumerate(error_data):
            plt.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(EVAL_DIR / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_worst_cases(self, num_cases=5):
        """Show a few worst qualitative cases with GT/pred keypoints."""
        if 'errors' not in self.results or not self.results['errors']['worst_cases']:
            return
            
        worst_cases = self.results['errors']['worst_cases'][:num_cases]
        
        fig, axes = plt.subplots(1, min(num_cases, len(worst_cases)),
                                 figsize=(4*min(num_cases, len(worst_cases)), 4))
        if num_cases == 1:
            axes = [axes]
        
        for i, case in enumerate(worst_cases):
            if i >= num_cases:
                break
                
            img_id = case['img_id']
            img_path = Path(case['file_name'])
            if not img_path.exists():
                continue
                
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            gt_kpts = self.ground_truth[img_id]['keypoints']
            pred_kpts = self.predictions[img_id]['keypoints']
            
            axes[i].imshow(img_rgb)
            
            # Ground truth (green)
            visible_gt = gt_kpts[gt_kpts[:, 2] > 0]
            if len(visible_gt) > 0:
                axes[i].scatter(visible_gt[:, 0], visible_gt[:, 1], 
                                c='green', s=50, alpha=0.8, label='Ground Truth')
            
            # Prediction (red)
            visible_pred = pred_kpts[pred_kpts[:, 2] > 0.1]
            if len(visible_pred) > 0:
                axes[i].scatter(visible_pred[:, 0], visible_pred[:, 1], 
                                c='red', s=50, alpha=0.8, label='Prediction')
            
            axes[i].set_title(f'Case {i+1}\nAvg. Error: {case["avg_error"]:.3f}', fontsize=10)
            axes[i].axis('off')
            if i == 0:
                axes[i].legend(fontsize=8)
        
        plt.suptitle('Worst-Case Predictions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(EVAL_DIR / 'worst_cases.png', dpi=300, bbox_inches='tight')
        plt.close()


    def plot_lr_confusion(self):
        """Pie charts for left/right confusion of wings and legs."""
        if not hasattr(self, 'lr_confusion_stats'):
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Wings
        wing_stats = self.lr_confusion_stats['wing_confusion']
        wing_labels = ['Normal', 'Confused']
        wing_values = [wing_stats['total_cases'] - wing_stats['confused_cases'], 
                       wing_stats['confused_cases']]
        
        axes[0].pie(wing_values, labels=wing_labels, autopct='%1.1f%%',
                    colors=['lightgreen', 'lightcoral'])
        axes[0].set_title(f"Left/Right Wing Confusion\nTotal: {wing_stats['total_cases']}")
        
        # Legs
        leg_stats = self.lr_confusion_stats['leg_confusion']
        leg_labels = ['Normal', 'Confused']
        leg_values = [leg_stats['total_cases'] - leg_stats['confused_cases'],
                      leg_stats['confused_cases']]
        
        axes[1].pie(leg_values, labels=leg_labels, autopct='%1.1f%%',
                    colors=['lightgreen', 'lightcoral'])
        axes[1].set_title(f"Left/Right Leg Confusion\nTotal: {leg_stats['total_cases']}")
        
        plt.suptitle('Left/Right Limb Confusion Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(EVAL_DIR / 'lr_confusion_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save results to JSON and CSV files."""
        self.log_step("Save Results", "Serialize evaluation artifacts")
        
        # JSON dump
        results_file = EVAL_DIR / 'evaluation_results.json'
        
        json_results = {}
        for key, value in self.results.items():
            if key == 'pck':
                json_results[key] = {}
                for thresh, pck_data in value.items():
                    json_results[key][thresh] = {
                        'overall': float(pck_data['overall']),
                        'per_point': {k: float(v) for k, v in pck_data['per_point'].items()}
                    }
            else:
                json_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # CSV table
        pck_01 = self.results['pck']['PCK@0.1']['per_point']
        error_stats = self.results.get('errors', {}).get('per_keypoint_error', {})
        
        report_data = []
        for kp_name in self.keypoint_names:
            row = {
                'Keypoint': kp_name,
                'PCK@0.1': pck_01.get(kp_name, 0.0),
                'Mean_Error': error_stats.get(kp_name, {}).get('mean', 0.0),
                'Std_Error': error_stats.get(kp_name, {}).get('std', 0.0),
                'Count': error_stats.get(kp_name, {}).get('count', 0)
            }
            report_data.append(row)
        
        df = pd.DataFrame(report_data)
        df.to_csv(EVAL_DIR / 'keypoint_performance.csv', index=False)
        
        self.log_step("Save Results", f"Artifacts saved to: {EVAL_DIR}", "OK")
    
    def generate_summary_report(self):
        """Append a human-readable summary to the logbook."""
        pck_01 = self.results['pck']['PCK@0.1']['overall']
        pck_05 = self.results['pck']['PCK@0.05']['overall']
        
        # Best/worst keypoints by PCK@0.1
        per_point_pck = self.results['pck']['PCK@0.1']['per_point']
        best_kp = max(per_point_pck.items(), key=lambda x: x[1])
        worst_kp = min(per_point_pck.items(), key=lambda x: x[1])
        
        summary = f"""
        ## Evaluation Summary

        ### Overall
        - PCK@0.05: {pck_05:.3f} ({pck_05*100:.1f}%)
        - PCK@0.1: {pck_01:.3f} ({pck_01*100:.1f}%)
        - #Predicted Images: {len(self.predictions)}
        - #Failed Cases: {len(self.failed_cases)}

        ### Keypoint Breakdown
        - Best: {best_kp[0]} (PCK@0.1: {best_kp[1]:.3f})
        - Worst: {worst_kp[0]} (PCK@0.1: {worst_kp[1]:.3f})

        ### Files
        - PCK bars: `{EVAL_DIR}/pck_comparison.png`
        - Per-keypoint PCK: `{EVAL_DIR}/per_keypoint_pck.png`  
        - Error distribution: `{EVAL_DIR}/error_distribution.png`
        - Worst cases: `{EVAL_DIR}/worst_cases.png`
        - JSON: `{EVAL_DIR}/evaluation_results.json`
        - CSV: `{EVAL_DIR}/keypoint_performance.csv`

        ### Suggestions
        """
        
        if pck_01 < 0.1:
            summary += "- Overall performance is low; consider more data or adjusting the model.\n"
        elif pck_01 < 0.3:
            summary += "- Room for improvement; try more epochs or stronger augmentation.\n"
        else:
            summary += "- Performance is reasonable; proceed to downstream testing.\n"
        
        if len(self.failed_cases) > len(self.predictions) * 0.1:
            summary += "- Failure rate is high; inspect data quality and model stability.\n"
        
        with open(LOGBOOK_FILE, 'a', encoding='utf-8') as f:
            f.write(summary)
        
        print(summary)

def main():
    print("CUB-15 HRNet Model Evaluation")
    print("="*60)
    
    evaluator = CUBEvaluator()
    
    # Check required files
    if not MODEL_PATH.exists():
        print(f"Missing model file: {MODEL_PATH}")
        print("Please run the training script first.")
        return
    
    if not CONFIG_PATH.exists():
        print(f"Missing config file: {CONFIG_PATH}")
        return
    
    # Pipeline
    success = True
    
    # 1) Load model
    if not evaluator.load_model():
        success = False
    
    # 2) Load ground-truth
    if success and not evaluator.load_ground_truth('val'):
        success = False
    
    # 3) Inference
    if success and not evaluator.run_inference():
        success = False
    
    # 4) Metrics
    if success:
        evaluator.evaluate_all_metrics()
        evaluator.analyze_errors()
    
    # 5) Visualizations
    if success:
        evaluator.create_visualizations()
    
    # 6) Save results and summary
    if success:
        evaluator.save_results()
        evaluator.generate_summary_report()
    
    if success:
        print(f"\nEvaluation complete.")
        print(f"Logbook: {LOGBOOK_FILE}")
        print(f"Results dir: {EVAL_DIR}")
    else:
        print(f"\nEvaluation failed — please check the logbook and console output.")

if __name__ == "__main__":
    main()
