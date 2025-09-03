#!/usr/bin/env python3
"""
Phase 1-2: CUB Skeleton Visualization
Function: Overlay the 15-point skeleton on real CUB images to verify connection validity.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import random

class SimpleCUBVisualizer:
    def __init__(self, cub_root):
        self.cub_root = Path(cub_root)
        self.load_cub_data()
        
        # Hard-coded 14 connections (based on CUB part_id: 1-15)
        self.skeleton_connections = [
            # Rigid connections - spine main chain
            (2, 5, 'rigid'),   # beak-crown
            (5, 10, 'rigid'),  # crown-nape  
            (10, 1, 'rigid'),  # nape-back
            
            # Rigid connections - head structure
            (5, 6, 'rigid'),   # crown-forehead
            (6, 15, 'rigid'),  # forehead-throat
            (5, 7, 'rigid'),   # crown-left_eye
            (5, 11, 'rigid'),  # crown-right_eye
            
            # Flexible connections - torso and appendages
            (1, 4, 'flexible'),   # back-breast
            (4, 3, 'flexible'),   # breast-belly
            (1, 9, 'flexible'),   # back-left_wing
            (1, 13, 'flexible'),  # back-right_wing
            (1, 14, 'flexible'),  # back-tail
            (3, 8, 'flexible'),   # belly-left_leg
            (3, 12, 'flexible'),  # belly-right_leg
        ]
        
        # CUB keypoint name mapping
        self.keypoint_names = {
            1: "back", 2: "beak", 3: "belly", 4: "breast", 5: "crown",
            6: "forehead", 7: "left_eye", 8: "left_leg", 9: "left_wing",
            10: "nape", 11: "right_eye", 12: "right_leg", 13: "right_wing",
            14: "tail", 15: "throat"
        }
        
    def load_cub_data(self):
        """Load basic CUB data"""
        # Keypoint data
        parts_file = self.cub_root / "parts" / "part_locs.txt"
        self.parts_df = pd.read_csv(parts_file, sep=' ', header=None,
                                   names=['image_id', 'part_id', 'x', 'y', 'visible'])
        
        # Image info
        images_file = self.cub_root / "images.txt"
        self.images_df = pd.read_csv(images_file, sep=' ', header=None,
                                    names=['image_id', 'image_path'])
        
        # Class labels
        labels_file = self.cub_root / "image_class_labels.txt"
        self.labels_df = pd.read_csv(labels_file, sep=' ', header=None,
                                    names=['image_id', 'class_id'])
        
        classes_file = self.cub_root / "classes.txt"
        self.classes_df = pd.read_csv(classes_file, sep=' ', header=None,
                                     names=['class_id', 'class_name'])
         
        print(f"Loaded CUB data: {len(self.images_df)} images")
        
    def get_keypoints_for_image(self, image_id):
        """Get keypoints for a single image"""
        image_parts = self.parts_df[self.parts_df['image_id'] == image_id]
        
        keypoints = {}
        for _, row in image_parts.iterrows():
            part_id = int(row['part_id'])
            keypoints[part_id] = {
                'x': float(row['x']),
                'y': float(row['y']),
                'visible': int(row['visible'])
            }
        
        return keypoints
    
    def get_label_offset(self, point_id, x, y):
        """Compute label offset based on keypoint ID and position to avoid overlap"""
        # Define different label offsets for different keypoints
        offset_map = {
            # Head region - dispersed to different directions
            2: (15, -15),   # beak - top-right
            5: (-25, -15),  # crown - top-left  
            6: (0, -25),    # forehead - top
            7: (-30, 0),    # left_eye - left
            11: (20, 0),    # right_eye - right
            15: (0, 20),    # throat - bottom
            10: (-15, -25), # nape - top-left
            
            # Torso region
            1: (-20, 0),    # back - left
            4: (15, 0),     # breast - right
            3: (0, 20),     # belly - bottom
            
            # Wings and tail
            9: (-35, -10),  # left_wing - top-left
            13: (20, -10),  # right_wing - top-right
            14: (-15, 15),  # tail - bottom-left
            
            # Legs
            8: (-25, 10),   # left_leg - bottom-left
            12: (15, 10),   # right_leg - bottom-right
        }
        
        return offset_map.get(point_id, (10, -10))  # default offset
    
    def draw_skeleton_on_image(self, image_path, keypoints, save_path=None):
        """Draw skeleton on the image - improved version"""
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Cannot read: {image_path}")
            return None
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Color definitions - make colors more visible
        colors = {
            'visible_point': (255, 50, 50),        # bright red
            'invisible_point': (150, 150, 150),    # light gray
            'rigid_line': (255, 0, 0),             # red
            'flexible_line': (30, 144, 255)        # dodger blue
        }
        
        # Draw connections - draw lines before points to avoid covering points
        for p1_id, p2_id, conn_type in self.skeleton_connections:
            if p1_id in keypoints and p2_id in keypoints:
                p1 = keypoints[p1_id]
                p2 = keypoints[p2_id]
                
                # Draw line only if both points are visible and coordinates are valid
                if (p1['visible'] and p2['visible'] and 
                    p1['x'] > 0 and p1['y'] > 0 and 
                    p2['x'] > 0 and p2['y'] > 0):
                    
                    color = colors['rigid_line'] if conn_type == 'rigid' else colors['flexible_line']
                    thickness = 2 if conn_type == 'rigid' else 1  # increase line thickness
                    
                    cv2.line(image, 
                            (int(p1['x']), int(p1['y'])), 
                            (int(p2['x']), int(p2['y'])), 
                            color, thickness)
        
        # Draw keypoints - increase size to be more visible
        for point_id, point in keypoints.items():
            x, y = int(point['x']), int(point['y'])
            
            # Skip invalid coordinates
            if x <= 0 or y <= 0:
                continue
                
            if point['visible']:
                # Visible point: bright red filled circle, larger size
                cv2.circle(image, (x, y), 3, colors['visible_point'], -1)
                cv2.circle(image, (x, y), 2, (255, 255, 255), 2)  # white border
            else:
                # Invisible point: light gray hollow circle
                cv2.circle(image, (x, y), 3, colors['invisible_point'], 3)
            
            # Add point label - optimized layout to avoid overlap
            name = self.keypoint_names.get(point_id, str(point_id))
            label = f"{point_id}"  # show ID only, simplified label
            
            # Adjust label offset based on point position to avoid overlap
            offset_x, offset_y = self.get_label_offset(point_id, x, y)
            
            # Add text background to improve readability
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            bg_x1, bg_y1 = x + offset_x - 2, y + offset_y - label_size[1] - 2
            bg_x2, bg_y2 = x + offset_x + label_size[0] + 2, y + offset_y + 2
            
            cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)  # black background
            cv2.putText(image, label, (x+offset_x, y+offset_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save image
        if save_path:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), image_bgr)
            
        return image
    
    def visualize_samples(self, num_samples=12, output_dir="cub_skeleton_samples"):
        """Visualize sample images"""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Randomly select images (ensure diversity)
        all_image_ids = self.images_df['image_id'].tolist()
        selected_ids = random.sample(all_image_ids, min(num_samples, len(all_image_ids)))
        
        print(f"Start processing {len(selected_ids)} images...")
        
        success_count = 0
        for i, image_id in enumerate(selected_ids):
            print(f"Processing {i+1}/{len(selected_ids)}: image ID {image_id}")
            
            # Get image path and class info
            image_row = self.images_df[self.images_df['image_id'] == image_id].iloc[0]
            image_path = self.cub_root / "images" / image_row['image_path']
            
            class_id = self.labels_df[self.labels_df['image_id'] == image_id]['class_id'].iloc[0]
            class_name = self.classes_df[self.classes_df['class_id'] == class_id]['class_name'].iloc[0]
            
            # Get keypoints
            keypoints = self.get_keypoints_for_image(image_id)
            
            # Generate visualization
            output_file = output_path / f"{class_name}_{image_id}.jpg"
            result = self.draw_skeleton_on_image(image_path, keypoints, output_file)
            
            if result is not None:
                success_count += 1
        
        print(f"Done! Successfully processed {success_count}/{len(selected_ids)} images")
        print(f"Output directory: {output_path}")
        
        return output_path

def main():
    """Main function"""
    print("Phase 1.5: Simplified CUB Skeleton Visualization\n")
    
    # Check CUB data path
    cub_root_path = "./CUB"
    if not os.path.exists(cub_root_path):
        print(f"CUB dataset path does not exist: {cub_root_path}")
        print("Please ensure the CUB folder is in the current directory")
        return
    
    # Create visualizer
    print("1. Initializing visualizer...")
    visualizer = SimpleCUBVisualizer(cub_root_path)
    
    # Generate sample visualizations
    print("\n2. Generating skeleton visualization samples...")
    output_dir = visualizer.visualize_samples(
        num_samples=15,  # generate 15 sample images
        output_dir="cub_skeleton_samples"
    )
    
    print(f"\nPhase 1.5 completed!")
    print("Generated content:")
    print(f"  - 15 optimized skeleton overlay sample images")
    print(f"  - Red thick line: rigid connections (spine main chain)")
    print(f"  - Blue thick line: flexible connections (wings, legs)")
    print(f"  - Red filled circle: visible keypoints")
    print(f"  - Gray hollow circle: invisible keypoints")
    print(f"  - Black background with white text labels: non-overlapping point ID markers")
    print("\nPlease check whether the skeleton connections are reasonable, then proceed to Phase 2 training")

if __name__ == "__main__":
    main()
