#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1-7: inferencedemo
----------------------
CUB-15 HRNet inference demo script.
Supports single-image inference, batch processing, and visualization.
python 1.7-inference_demo.py -i data/CUB/images/091.Mockingbird --batch -m 15 -c 0.1
python 1.7-inference_demo.py -i mybird.jpg
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
from pathlib import Path
import json
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
DEMO_DIR = WORK_DIR / "demo_results"

# Create output directory
DEMO_DIR.mkdir(parents=True, exist_ok=True)

class CUBInferenceDemo:
    def __init__(self):
        self.model = None
        self.keypoint_names = []
        self.skeleton_connections = []
        self.colors = []
        
    def get_correct_skeleton_connections(self):
        """Get corrected skeleton connections (based on debug-verified result)."""
        
        # name -> index
        name_to_idx = {name: i for i, name in enumerate(self.keypoint_names)}
        
        # Correct connections by name
        correct_connections_by_name = [
            # Spine main chain: beak→crown→nape→back
            ('beak', 'crown'),
            ('crown', 'nape'), 
            ('nape', 'back'),
            
            # Head structure
            ('crown', 'forehead'),
            ('crown', 'throat'),
            
            # Eye symmetry  
            ('crown', 'left_eye'),
            ('crown', 'right_eye'),
            
            # Torso
            ('back', 'breast'),
            ('breast', 'belly'),
            
            # Wings
            ('back', 'left_wing'),
            ('back', 'right_wing'),
            
            # Tail
            ('back', 'tail'),
            
            # Legs  
            ('belly', 'left_leg'),
            ('belly', 'right_leg'),
        ]
        
        # Convert to index pairs
        idx_connections = []
        
        for name1, name2 in correct_connections_by_name:
            if name1 in name_to_idx and name2 in name_to_idx:
                idx1, idx2 = name_to_idx[name1], name_to_idx[name2]
                idx_connections.append([idx1, idx2])
        
        return idx_connections
    
    def load_model(self):
        """Load the trained model."""
        print("Loading HRNet model...")
        
        try:
            register_all_modules(init_default_scope=False)
            
            # Ensure string paths
            self.model = init_model(str(CONFIG_PATH), str(MODEL_PATH), device='cuda:0')
            
            # Load skeleton config
            with open("skeleton.yaml") as f:
                skeleton = yaml.safe_load(f)
            
            self.keypoint_names = [kp['name'] for kp in skeleton['keypoints']]
            
            # Use corrected connections instead of YAML
            self.skeleton_connections = self.get_correct_skeleton_connections()
            
            # Colors
            self.colors = plt.cm.Set1(np.linspace(0, 1, len(self.keypoint_names)))
            
            print(f"Model loaded. {len(self.keypoint_names)} keypoints supported.")
            print(f"Loaded {len(self.skeleton_connections)} skeleton connections.")
            return True
            
        except Exception as e:
            print(f"Model load failed: {e}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            return False
    
    def predict_single_image(self, image_path, confidence_threshold=0.3):
        """Predict keypoints for a single image."""
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"Image file does not exist: {image_path}")
            return None
        
        try:
            # Run inference
            results = inference_topdown(self.model, str(image_path))
            
            if not results or len(results) == 0:
                print(f"No objects detected: {image_path.name}")
                return None
            
            # Extract prediction
            pred_instances = results[0].pred_instances
            
            # Handle keypoints data
            keypoints = pred_instances.keypoints
            if hasattr(keypoints, 'cpu'):
                keypoints = keypoints[0].cpu().numpy()
            elif isinstance(keypoints, np.ndarray):
                keypoints = keypoints[0]
            else:
                keypoints = keypoints[0]
            
            # Handle scores
            scores = pred_instances.keypoint_scores
            if hasattr(scores, 'cpu'):
                scores = scores[0].cpu().numpy()
            elif isinstance(scores, np.ndarray):
                scores = scores[0]
            else:
                scores = scores[0]
            
            # Handle bbox
            bbox = None
            if hasattr(pred_instances, 'bboxes') and len(pred_instances.bboxes) > 0:
                bbox_data = pred_instances.bboxes[0]
                if hasattr(bbox_data, 'cpu'):
                    bbox = bbox_data.cpu().numpy()
                else:
                    bbox = bbox_data
            
            # Filter by confidence
            valid_points = scores >= confidence_threshold
            
            result = {
                'image_path': str(image_path),
                'keypoints': keypoints,
                'scores': scores,
                'bbox': bbox,
                'valid_points': valid_points,
                'num_valid_points': np.sum(valid_points)
            }
            
            print(f"Prediction done: {image_path.name} - detected {np.sum(valid_points)}/{len(self.keypoint_names)} keypoints")
            return result
            
        except Exception as e:
            print(f"Prediction failed for {image_path.name}: {e}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            return None
    
    def visualize_result(self, result, save_path=None, show_connections=True, show_labels=True):
        """Visualize prediction results."""
        if result is None:
            return None
        
        # Read image
        img = cv2.imread(result['image_path'])
        if img is None:
            print(f"Cannot read image: {result['image_path']}")
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Figure
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        
        keypoints = result['keypoints']
        scores = result['scores']
        valid_points = result['valid_points']
        
        # Keypoints
        for i, (kp, score, is_valid) in enumerate(zip(keypoints, scores, valid_points)):
            if is_valid:
                color = self.colors[i]
                plt.scatter(kp[0], kp[1], c=[color], s=80, alpha=0.8, edgecolors='white', linewidth=2)
                
                # Labels
                if show_labels:
                    plt.annotate(f'{self.keypoint_names[i]}\n{score:.2f}', 
                               (kp[0], kp[1]), 
                               xytext=(5, 5), 
                               textcoords='offset points',
                               fontsize=8, 
                               color='white',
                               bbox=dict(boxstyle='round,pad=0.3', fc=color, alpha=0.7),
                               fontweight='bold')
        
        # Skeleton
        if show_connections:
            for connection in self.skeleton_connections:
                if len(connection) >= 2:
                    pt1_idx, pt2_idx = connection[0], connection[1]
                    if (0 <= pt1_idx < len(valid_points) and 
                        0 <= pt2_idx < len(valid_points) and
                        valid_points[pt1_idx] and valid_points[pt2_idx]):
                        
                        pt1 = keypoints[pt1_idx]
                        pt2 = keypoints[pt2_idx]
                        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 
                               'b-', linewidth=2, alpha=0.6)
        
        # Bbox
        if result['bbox'] is not None:
            bbox = result['bbox']
            if len(bbox) >= 4:
                rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                   fill=False, color='green', linewidth=2, linestyle='--')
                plt.gca().add_patch(rect)
        
        # Title
        img_name = Path(result['image_path']).name
        valid_count = result['num_valid_points']
        total_count = len(self.keypoint_names)
        
        plt.title(
            f'CUB-15 Keypoint Detection Result\n'
            f'Image: {img_name}\n'
            f'Detected: {valid_count}/{total_count} keypoints', 
            fontsize=14, fontweight='bold'
        )
        plt.axis('off')
        
        # Save
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Visualization saved: {save_path}")
        
        return plt.gcf()
    
    def process_single_image(self, image_path, confidence_threshold=0.3, save_result=True):
        """Full pipeline for a single image."""
        print(f"\nProcessing image: {Path(image_path).name}")
        print("-" * 50)
        
        # Predict
        result = self.predict_single_image(image_path, confidence_threshold)
        
        if result is None:
            return None
        
        # Visualize
        if save_result:
            output_name = f"{Path(image_path).stem}_result.png"
            save_path = DEMO_DIR / output_name
            fig = self.visualize_result(result, save_path)
            plt.close(fig)
        else:
            fig = self.visualize_result(result)
            plt.show()
        
        # Details
        self.print_detailed_results(result)
        
        return result
    
    def print_detailed_results(self, result):
        """Print detailed prediction results."""
        print(f"\nDetails:")
        print(f"   Image: {Path(result['image_path']).name}")
        print(f"   Valid keypoints: {result['num_valid_points']}/{len(self.keypoint_names)}")
        
        print(f"\nKeypoint details:")
        for i, (name, kp, score, is_valid) in enumerate(zip(
            self.keypoint_names, result['keypoints'], result['scores'], result['valid_points']
        )):
            status = "OK" if is_valid else "MISS"
            print(f"   {status} {i:2d}. {name:12s}: ({kp[0]:6.1f}, {kp[1]:6.1f}) confidence: {score:.3f}")
    
    def process_batch(self, input_dir, confidence_threshold=0.3, max_images=None):
        """Batch process an image directory."""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            print(f"Input directory does not exist: {input_dir}")
            return []
        
        # Supported extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No image files found in: {input_dir}")
            return []
        
        # Limit
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"Start batch processing: {len(image_files)} images")
        print("=" * 60)
        
        results = []
        success_count = 0
        
        for i, img_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {img_file.name}")
            
            result = self.predict_single_image(img_file, confidence_threshold)
            
            if result is not None:
                # Visualize and save
                output_name = f"{img_file.stem}_result.png"
                save_path = DEMO_DIR / output_name
                fig = self.visualize_result(result, save_path)
                plt.close(fig)
                
                results.append(result)
                success_count += 1
            
        # Batch report
        self.generate_batch_report(results, input_dir)
        
        print(f"\nBatch processing complete.")
        print(f"   Success: {success_count}/{len(image_files)} images")
        print(f"   Results saved to: {DEMO_DIR}")
        
        return results
    
    def generate_batch_report(self, results, input_dir):
        """Generate a batch processing report."""
        if not results:
            return
        
        # Stats
        total_images = len(results)
        avg_keypoints = np.mean([r['num_valid_points'] for r in results])
        avg_confidence = np.mean([np.mean(r['scores'][r['valid_points']]) for r in results if np.any(r['valid_points'])])
        
        # Detection rate per keypoint
        keypoint_detection_rate = {}
        for i, kp_name in enumerate(self.keypoint_names):
            detected_count = sum(1 for r in results if r['valid_points'][i])
            keypoint_detection_rate[kp_name] = detected_count / total_images
        
        # Report
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'input_directory': str(input_dir),
            'total_images': total_images,
            'average_keypoints_per_image': float(avg_keypoints),
            'average_confidence': float(avg_confidence),
            'keypoint_detection_rates': keypoint_detection_rate,
            'results': [
                {
                    'image': Path(r['image_path']).name,
                    'num_keypoints': int(r['num_valid_points']),
                    'avg_confidence': float(np.mean(r['scores'][r['valid_points']])) if np.any(r['valid_points']) else 0.0
                }
                for r in results
            ]
        }
        
        # Save
        report_path = DEMO_DIR / 'batch_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nBatch report:")
        print(f"   Avg keypoints: {avg_keypoints:.1f}/{len(self.keypoint_names)}")
        print(f"   Avg confidence: {avg_confidence:.3f}")
        print(f"   Report: {report_path}")
    
    def create_comparison_grid(self, results, grid_size=(2, 3)):
        """Create a comparison grid figure."""
        if not results:
            return
        
        num_images = min(len(results), grid_size[0] * grid_size[1])
        selected_results = results[:num_images]
        
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 10))
        axes = axes.flatten() if num_images > 1 else [axes]
        
        for i, result in enumerate(selected_results):
            if i >= len(axes):
                break
                
            # Read image
            img = cv2.imread(result['image_path'])
            if img is None:
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img_rgb)
            
            # Draw keypoints
            keypoints = result['keypoints']
            valid_points = result['valid_points']
            
            for j, (kp, is_valid) in enumerate(zip(keypoints, valid_points)):
                if is_valid:
                    axes[i].scatter(kp[0], kp[1], c=[self.colors[j]], s=30, alpha=0.8)
            
            # Connections
            for connection in self.skeleton_connections:
                if len(connection) >= 2:
                    pt1_idx, pt2_idx = connection[0], connection[1]
                    if (0 <= pt1_idx < len(valid_points) and 
                        0 <= pt2_idx < len(valid_points) and
                        valid_points[pt1_idx] and valid_points[pt2_idx]):
                        
                        pt1 = keypoints[pt1_idx]
                        pt2 = keypoints[pt2_idx]
                        axes[i].plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=1, alpha=0.6)
            
            img_name = Path(result['image_path']).name
            axes[i].set_title(f'{img_name}\n{result["num_valid_points"]}/{len(self.keypoint_names)} points', fontsize=10)
            axes[i].axis('off')
        
        # Hide empty subplots
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('CUB-15 Keypoint Detection Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save grid
        comparison_path = DEMO_DIR / 'results_comparison.png'
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison grid saved: {comparison_path}")

def main():
    parser = argparse.ArgumentParser(description='CUB-15 keypoint detection demo')
    parser.add_argument('--input', '-i', required=True, help='Path to input image file or directory')
    parser.add_argument('--confidence', '-c', type=float, default=0.3, help='Confidence threshold (default: 0.3)')
    parser.add_argument('--batch', '-b', action='store_true', help='Batch processing mode')
    parser.add_argument('--max_images', '-m', type=int, help='Maximum number of images to process')
    parser.add_argument('--no_save', action='store_true', help='Do not save result images')
    parser.add_argument('--grid', action='store_true', help='Generate comparison grid')
    
    args = parser.parse_args()
    
    print("CUB-15 keypoint detection demo")
    print("=" * 60)
    
    # Check required files
    if not MODEL_PATH.exists():
        print(f"Missing model file: {MODEL_PATH}")
        print("Please run the training script first.")
        return
    
    if not CONFIG_PATH.exists():
        print(f"Missing config file: {CONFIG_PATH}")
        return
    
    # Initialize demo
    demo = CUBInferenceDemo()
    if not demo.load_model():
        return
    
    # Handle input
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        # Batch mode
        results = demo.process_batch(
            input_path, 
            confidence_threshold=args.confidence,
            max_images=args.max_images
        )
        
        # Comparison grid
        if args.grid and results:
            demo.create_comparison_grid(results)
            
    else:
        # Single-image mode
        demo.process_single_image(
            input_path,
            confidence_threshold=args.confidence,
            save_result=not args.no_save
        )
    
    print(f"\nDemo complete. Results saved to: {DEMO_DIR}")

if __name__ == "__main__":
    main()
