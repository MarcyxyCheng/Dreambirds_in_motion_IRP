#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1-3: prepare data to coco format
--------------------
Convert CUB-200-2011 data to COCO format usable by MMPose.
Focus on the CUB-15 keypoints with enhanced debugging and validation.
"""

import json, random, cv2, datetime, time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# Paths
CUB_ROOT = Path("data/CUB")
WORK_DIR = Path("workdir/hrnet15_cub")
COCO_DIR = WORK_DIR / "coco_format"
LOGBOOK_FILE = WORK_DIR / "data_preparation_logbook.md"
SEED = 42

# Debug mode: process a small subset for quick testing
DEBUG_MODE = True  # Set to False to process all data
DEBUG_LIMIT = 5000   # Max images per split in debug mode

class DataLogger:
    def __init__(self, logbook_path):
        self.logbook_path = Path(logbook_path)
        self.start_time = None
        self.logbook_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_session()
    
    def _init_session(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        header = f"""
## Data Preparation Session - {session_id}

**Start Time**: {timestamp}
**Debug Mode**: {'ON (limit {} images)'.format(DEBUG_LIMIT) if DEBUG_MODE else 'OFF (process all data)'}
**CUB Data Path**: {CUB_ROOT}
**Output Path**: {COCO_DIR}

### Steps
"""
        
        # Append to logbook
        with open(self.logbook_path, 'a', encoding='utf-8') as f:
            f.write(header)
    
    def log_step(self, step_name, message, status="IN-PROGRESS"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"{timestamp} {status} **{step_name}**: {message}\n\n"
        
        with open(self.logbook_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        print(f"{status} {step_name}: {message}")
    
    def log_stats(self, stats_dict):
        stats_text = "**Statistics**:\n"
        for key, value in stats_dict.items():
            stats_text += f"- {key}: {value}\n"
        stats_text += "\n"
        
        with open(self.logbook_path, 'a', encoding='utf-8') as f:
            f.write(stats_text)

logger = DataLogger(LOGBOOK_FILE)

def load_cub_data():
    """Load raw CUB data"""
    logger.log_step("Load CUB Data", "Start reading raw files")
    
    # Check required files
    required_files = [
        CUB_ROOT / "images.txt",
        CUB_ROOT / "parts" / "part_locs.txt", 
        CUB_ROOT / "train_test_split.txt"
    ]
    
    for file_path in required_files:
        if not file_path.exists():
            logger.log_step("File Check", f"Missing required file: {file_path}", "ERROR")
            return None, None, None
    
    # Load image info
    images_df = pd.read_csv(CUB_ROOT / "images.txt", 
                           sep=' ', header=None,
                           names=['image_id', 'image_path'])
    
    # Load keypoint data
    parts_df = pd.read_csv(CUB_ROOT / "parts" / "part_locs.txt", 
                          sep=' ', header=None,
                          names=['image_id', 'part_id', 'x', 'y', 'visible'])
    
    # Load train/test split
    split_df = pd.read_csv(CUB_ROOT / "train_test_split.txt",
                          sep=' ', header=None, 
                          names=['image_id', 'is_train'])
    
    logger.log_stats({
        "Total Images": len(images_df),
        "Keypoint Records": len(parts_df),
        "Training Images": len(split_df[split_df['is_train']==1]),
        "Test Images": len(split_df[split_df['is_train']==0])
    })
    
    logger.log_step("Load CUB Data", "Raw data loaded", "OK")
    return images_df, parts_df, split_df

def calculate_bbox_and_area(keypoints, img_width, img_height):
    """Compute a valid bbox from keypoints"""
    visible_kpts = keypoints[keypoints[:, 2] > 0]
    
    if len(visible_kpts) < 2:
        return None, None
    
    x_coords, y_coords = visible_kpts[:, 0], visible_kpts[:, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Add margins
    margin = 20
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(img_width, x_max + margin)
    y_max = min(img_height, y_max + margin)
    
    bbox_w, bbox_h = x_max - x_min, y_max - y_min
    
    # Ensure bbox is valid
    if bbox_w > 10 and bbox_h > 10:
        bbox = [float(x_min), float(y_min), float(bbox_w), float(bbox_h)]
        area = float(bbox_w * bbox_h)
        return bbox, area
    
    return None, None

def validate_sample_quality(keypoints, bbox, img_width, img_height):
    """Validate a single sample's data quality"""
    issues = []
    
    # Check keypoint coordinate ranges
    visible_kpts = keypoints[keypoints[:, 2] > 0]
    if len(visible_kpts) > 0:
        x_coords, y_coords = visible_kpts[:, 0], visible_kpts[:, 1]
        if x_coords.min() < 0 or x_coords.max() >= img_width:
            issues.append("Keypoint X out of image bounds")
        if y_coords.min() < 0 or y_coords.max() >= img_height:
            issues.append("Keypoint Y out of image bounds")
    
    # Check bbox
    if bbox is None:
        issues.append("Failed to compute a valid bbox")
    else:
        x, y, w, h = bbox
        if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
            issues.append("BBox out of image bounds")
    
    # Check number of visible keypoints
    visible_count = np.sum(keypoints[:, 2] > 0)
    if visible_count < 3:
        issues.append(f"Too few visible keypoints ({visible_count})")
    
    return issues

def convert_cub_to_coco():
    """Convert CUB data to COCO format while keeping 15 keypoints"""
    logger.log_step("Conversion", "Start CUB→COCO conversion")
    
    # Load raw data
    images_df, parts_df, split_df = load_cub_data()
    if images_df is None:
        return False
    
    # Load skeleton config
    try:
        with open("skeleton.yaml") as f:
            skeleton_config = yaml.safe_load(f)
        logger.log_step("Skeleton Config", "skeleton.yaml loaded", "OK")
    except Exception as e:
        logger.log_step("Skeleton Config", f"Failed to load skeleton.yaml: {e}", "ERROR")
        return False
    
    # Create output directory
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    
    def process_split(is_train_split):
        """Process a single data split"""
        split_name = "train" if is_train_split else "val"
        logger.log_step(f"Process {split_name}", "Start")
        
        # Get image IDs for this split
        target_image_ids = split_df[split_df['is_train'] == (1 if is_train_split else 0)]['image_id'].tolist()
        
        # Limit in debug mode
        if DEBUG_MODE:
            target_image_ids = target_image_ids[:DEBUG_LIMIT]
            logger.log_step(f"{split_name} Debug", f"Limit to {len(target_image_ids)} images")
        
        images = []
        annotations = []
        ann_id = 1
        processed = 0
        skipped = 0
        quality_issues = []
        
        for img_id in target_image_ids:
            # Image info
            img_info = images_df[images_df['image_id'] == img_id]
            if len(img_info) == 0:
                skipped += 1
                continue
                
            img_path = CUB_ROOT / "images" / img_info.iloc[0]['image_path']
            
            # Check existence
            if not img_path.exists():
                quality_issues.append(f"Image file not found: {img_path}")
                skipped += 1
                continue
            
            # Read and get size
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    quality_issues.append(f"Failed to read image: {img_path}")
                    skipped += 1
                    continue
                h, w = img.shape[:2]
            except Exception as e:
                quality_issues.append(f"Image read error {img_path}: {e}")
                skipped += 1
                continue
            
            # Keypoints for this image
            img_parts = parts_df[parts_df['image_id'] == img_id]
            if len(img_parts) == 0:
                quality_issues.append(f"Image {img_id} has no keypoint data")
                skipped += 1
                continue
            
            # Build 15x3 keypoints
            keypoints = np.zeros((15, 3))
            for _, row in img_parts.iterrows():
                part_idx = int(row['part_id']) - 1  # 0-14
                if 0 <= part_idx < 15:
                    keypoints[part_idx] = [row['x'], row['y'], row['visible']]
            
            # BBox
            bbox, area = calculate_bbox_and_area(keypoints, w, h)
            
            # Sample quality checks
            issues = validate_sample_quality(keypoints, bbox, w, h)
            if issues:
                quality_issues.extend([f"Image {img_id}: {issue}" for issue in issues])
                skipped += 1
                continue
            
            # Save image entry
            images.append({
                'id': int(img_id),
                'file_name': str(img_path.relative_to(Path("data"))),
                'width': int(w),
                'height': int(h)
            })
            
            # Save annotation entry
            visible_count = int(np.sum(keypoints[:, 2] > 0))
            annotations.append({
                'id': ann_id,
                'image_id': int(img_id),
                'category_id': 1,
                'keypoints': keypoints.reshape(-1).tolist(),  # 45-dim
                'num_keypoints': visible_count,
                'bbox': bbox,
                'area': area,
                'iscrowd': 0,
                'segmentation': []
            })
            ann_id += 1
            processed += 1
        
        # Compose COCO dict
        coco_data = {
            'images': images,
            'annotations': annotations,
            'categories': [{
                'id': 1,
                'name': 'bird',
                'keypoints': [kp['name'] for kp in skeleton_config['keypoints']],
                'skeleton': skeleton_config['connections']['rigid'] + skeleton_config['connections']['flexible']
            }]
        }
        
        # Save file
        output_file = COCO_DIR / f"{split_name}.json"
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        # Stats
        logger.log_stats({
            f"{split_name} Results": f"{processed} succeeded, {skipped} skipped",
            f"{split_name} Output File": str(output_file),
            f"Average Visible Keypoints": f"{np.mean([ann['num_keypoints'] for ann in annotations]):.1f}" if annotations else "N/A"
        })
        
        # Quality issues (first 10)
        if quality_issues:
            logger.log_step(f"{split_name} Quality Issues", f"Found {len(quality_issues)} issues, examples:", "WARN")
            for issue in quality_issues[:10]:
                logger.log_step("Issue Detail", issue, "-")
        
        return processed > 0, len(images), len(annotations)
    
    # Process train and val
    train_success, train_imgs, train_anns = process_split(True)
    val_success, val_imgs, val_anns = process_split(False)
    
    if train_success and val_success:
        logger.log_step("Conversion", f"Done! Train {train_imgs} imgs/{train_anns} anns, Val {val_imgs} imgs/{val_anns} anns", "DONE")
        return True
    else:
        logger.log_step("Conversion", "Failed", "ERROR")
        return False

def verify_coco_format():
    """Validate generated COCO format"""
    logger.log_step("Format Check", "Start verifying COCO format")
    
    for split in ["train", "val"]:
        json_file = COCO_DIR / f"{split}.json"
        if not json_file.exists():
            logger.log_step("Format Check", f"{split}.json not found", "ERROR")
            continue
        
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Required keys
            required_keys = ['images', 'annotations', 'categories']
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                logger.log_step("Format Check", f"{split}.json missing keys: {missing_keys}", "ERROR")
                continue
            
            # Consistency
            image_ids = set(img['id'] for img in data['images'])
            ann_image_ids = set(ann['image_id'] for ann in data['annotations'])
            
            orphan_anns = ann_image_ids - image_ids
            if orphan_anns:
                logger.log_step("Format Check", f"{split}.json has orphan annotations: {len(orphan_anns)}", "WARN")
            
            # Keypoint shape
            sample_ann = data['annotations'][0] if data['annotations'] else None
            if sample_ann:
                keypoints = sample_ann['keypoints']
                if len(keypoints) != 45:  # 15 * 3
                    logger.log_step("Format Check", f"{split}.json keypoint dimension error: {len(keypoints)}", "ERROR")
                    continue
            
            logger.log_step("Format Check", f"{split}.json OK", "OK")
            
        except Exception as e:
            logger.log_step("Format Check", f"{split}.json validation failed: {e}", "ERROR")

def create_test_subset():
    """Create a small test subset for quick debugging"""
    if not DEBUG_MODE:
        return
        
    logger.log_step("Test Subset", "Create small test datasets")
    
    for split in ["train", "val"]:
        json_file = COCO_DIR / f"{split}.json"
        test_file = COCO_DIR / f"{split}_test.json"
        
        if not json_file.exists():
            continue
            
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Keep only first 5 images
            test_images = data['images'][:5]
            test_image_ids = set(img['id'] for img in test_images)
            test_annotations = [ann for ann in data['annotations'] if ann['image_id'] in test_image_ids]
            
            test_data = {
                'images': test_images,
                'annotations': test_annotations,
                'categories': data['categories']
            }
            
            with open(test_file, 'w') as f:
                json.dump(test_data, f, indent=2)
                
            logger.log_step("Test Subset", f"Created {split}_test.json: {len(test_images)} imgs/{len(test_annotations)} anns", "OK")
            
        except Exception as e:
            logger.log_step("Test Subset", f"{split} test subset creation failed: {e}", "ERROR")

def main():
    logger.log_step("Start", "CUB-15 data preparation pipeline started")
    
    # Seeds
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Check CUB dataset
    if not CUB_ROOT.exists():
        logger.log_step("Env Check", f"CUB dataset not found: {CUB_ROOT}", "ERROR")
        return
    
    # Check skeleton.yaml
    if not Path("skeleton.yaml").exists():
        logger.log_step("Env Check", "skeleton.yaml missing; please run Phase 1 first", "ERROR")
        return
    
    logger.log_step("Env Check", "Environment ready", "OK")
    
    # Step 1: Convert
    if not convert_cub_to_coco():
        logger.log_step("Stop", "Conversion failed", "ERROR")
        return
    
    # Step 2: Verify
    verify_coco_format()
    
    # Step 3: Test subset (debug mode)
    create_test_subset()
    
    # Done
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.log_step("Finish", f"Data preparation completed ({end_time})", "DONE")
    
    print(f"\nDetailed log saved to: {LOGBOOK_FILE}")
    print(f"COCO-format data saved to: {COCO_DIR}")
    
    if DEBUG_MODE:
        print(f"Debug mode is ON with limit {DEBUG_LIMIT} images")
        print("Set DEBUG_MODE to False to process all data")

if __name__ == "__main__":
    main()
