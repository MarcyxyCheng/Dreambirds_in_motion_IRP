#!/usr/bin/env python3
"""
Phase 1-1: CUB-200 Skeleton Definition (based on dataset structure)
Functions:
1. Define skeleton connections for the standard CUB 15 keypoints
2. Generate skeleton.yaml configuration file
3. Create dataset loader
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import matplotlib.pyplot as plt

def create_cub_skeleton_config():
    """Create CUB-15 skeleton configuration"""
    
    # CUB-200 standard 15 keypoints (part_id 1-15)
    keypoints_info = [
        {"id": 1, "name": "back", "visibility_weight": 1.0},
        {"id": 2, "name": "beak", "visibility_weight": 1.0}, 
        {"id": 3, "name": "belly", "visibility_weight": 0.8},
        {"id": 4, "name": "breast", "visibility_weight": 0.9},
        {"id": 5, "name": "crown", "visibility_weight": 1.0},
        {"id": 6, "name": "forehead", "visibility_weight": 0.9},
        {"id": 7, "name": "left_eye", "visibility_weight": 0.7},
        {"id": 8, "name": "left_leg", "visibility_weight": 0.6},
        {"id": 9, "name": "left_wing", "visibility_weight": 0.8},
        {"id": 10, "name": "nape", "visibility_weight": 0.9},
        {"id": 11, "name": "right_eye", "visibility_weight": 0.7},
        {"id": 12, "name": "right_leg", "visibility_weight": 0.6},
        {"id": 13, "name": "right_wing", "visibility_weight": 0.8},
        {"id": 14, "name": "tail", "visibility_weight": 0.8},
        {"id": 15, "name": "throat", "visibility_weight": 0.9}
    ]
    
    # Connection definitions (part_id: 1-15)
    connections = {
        "rigid": [
            # Spine chain: beak(2) → crown(5) → nape(10) → back(1)
            [2, 5],   # beak-crown
            [5, 10],  # crown-nape  
            [10, 1],  # nape-back
            # Head structure
            [5, 6],   # crown-forehead
            [6, 15],  # forehead-throat
            [5, 7],   # crown-left_eye
            [5, 11],  # crown-right_eye
        ],
        "flexible": [
            # Torso connections
            [1, 4],   # back-breast
            [4, 3],   # breast-belly
            # Wing connections (from back, anatomically consistent)
            [1, 9],   # back-left_wing
            [1, 13],  # back-right_wing
            # Tail connection
            [1, 14],  # back-tail
            # Leg connections
            [3, 8],   # belly-left_leg
            [3, 12],  # belly-right_leg
        ]
    }
    
    # Full configuration
    skeleton_config = {
        "dataset": "CUB-200-2011",
        "num_keypoints": 15,
        "keypoints": keypoints_info,
        "connections": connections,
        "preprocessing": {
            "image_size": [256, 256],
            "coordinate_normalization": True,  # Normalize to [0,1]
            "augmentation": {
                "rotation": {"range": 15, "prob": 0.5},
                "scaling": {"range": [0.8, 1.2], "prob": 0.5}, 
                "horizontal_flip": {"prob": 0.5},
                "color_jitter": {"brightness": 0.2, "contrast": 0.2, "prob": 0.3}
            }
        },
        "training": {
            "visibility_threshold": 0.5,
            "invisible_weight": 0.1,
            "geometry_constraint_weight": 0.2,
            "rigid_connection_weight": 1.0,
            "flexible_connection_weight": 0.5
        }
    }
    
    return skeleton_config

def load_cub_data(cub_root):
    """Load key information from CUB dataset"""
    cub_root = Path(cub_root)
    
    # Load keypoint data
    parts_file = cub_root / "parts" / "part_locs.txt"
    parts_df = pd.read_csv(parts_file, sep=' ', header=None,
                          names=['image_id', 'part_id', 'x', 'y', 'visible'])
    
    # Load image info
    images_file = cub_root / "images.txt" 
    images_df = pd.read_csv(images_file, sep=' ', header=None,
                           names=['image_id', 'image_path'])
    
    # Load train/test split
    split_file = cub_root / "train_test_split.txt"
    split_df = pd.read_csv(split_file, sep=' ', header=None,
                          names=['image_id', 'is_train'])
    
    print(f"Data statistics:")
    print(f"  - Number of images: {len(images_df)}")
    print(f"  - Number of keypoint records: {len(parts_df)}")
    print(f"  - Training images: {len(split_df[split_df['is_train']==1])}")
    print(f"  - Test images: {len(split_df[split_df['is_train']==0])}")
    
    return parts_df, images_df, split_df

def create_cub_dataset_class():
    """Create CUB dataset class"""
    dataset_code = '''
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

class CUBKeypointDataset(Dataset):
    def __init__(self, cub_root, split='train', transform=None, img_size=256):
        self.cub_root = Path(cub_root)
        self.img_size = img_size
        self.transform = transform
        
        # Load data
        self.parts_df = pd.read_csv(self.cub_root / "parts" / "part_locs.txt", 
                                   sep=' ', header=None,
                                   names=['image_id', 'part_id', 'x', 'y', 'visible'])
        self.images_df = pd.read_csv(self.cub_root / "images.txt", 
                                    sep=' ', header=None,
                                    names=['image_id', 'image_path'])
        self.split_df = pd.read_csv(self.cub_root / "train_test_split.txt",
                                   sep=' ', header=None, 
                                   names=['image_id', 'is_train'])
        
        # Filter data
        is_train = 1 if split == 'train' else 0
        self.image_ids = self.split_df[self.split_df['is_train'] == is_train]['image_id'].tolist()
        
        print(f"Loaded {len(self.image_ids)} {split} images")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = self.images_df[self.images_df['image_id'] == image_id]['image_path'].iloc[0]
        image_full_path = self.cub_root / "images" / image_path
        image = Image.open(image_full_path).convert('RGB')
        
        # Get keypoints
        keypoints_data = self.parts_df[self.parts_df['image_id'] == image_id]
        keypoints = np.zeros((15, 3))  # 15 points, each (x, y, visible)
        
        for _, row in keypoints_data.iterrows():
            part_id = int(row['part_id']) - 1  # convert to 0-14 index
            keypoints[part_id] = [row['x'], row['y'], row['visible']]
        
        # Normalize coordinates
        orig_w, orig_h = image.size
        keypoints[:, 0] = keypoints[:, 0] / orig_w  # normalize x
        keypoints[:, 1] = keypoints[:, 1] / orig_h  # normalize y
        
        # Image preprocessing
        if self.transform:
            image = self.transform(image)
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)
        
        return {
            'image': image,
            'keypoints': torch.FloatTensor(keypoints),
            'image_id': image_id
        }

# Function to create dataloaders
def create_dataloaders(cub_root, batch_size=16, num_workers=4):
    train_dataset = CUBKeypointDataset(cub_root, split='train')
    val_dataset = CUBKeypointDataset(cub_root, split='test')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader
'''
    
    with open("cub_dataset.py", "w") as f:
        f.write(dataset_code)
    print("CUB dataset class created: cub_dataset.py")

def visualize_skeleton(config, save_path="skeleton_structure.png"):
    """Visualize skeleton connections"""
    keypoints = {kp['id']: kp['name'] for kp in config['keypoints']}
    
    # Keypoint positions for visualization
    point_positions = {
        1: (0.5, 0.6),    # back
        2: (0.5, 0.9),    # beak  
        3: (0.5, 0.3),    # belly
        4: (0.5, 0.45),   # breast
        5: (0.5, 0.8),    # crown
        6: (0.5, 0.85),   # forehead
        7: (0.4, 0.8),    # left_eye
        8: (0.4, 0.1),    # left_leg
        9: (0.3, 0.6),    # left_wing
        10: (0.5, 0.7),   # nape
        11: (0.6, 0.8),   # right_eye
        12: (0.6, 0.1),   # right_leg
        13: (0.7, 0.6),   # right_wing
        14: (0.5, 0.5),   # tail
        15: (0.5, 0.75)   # throat
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Draw connections
    for conn_type, connections in config['connections'].items():
        color = 'red' if conn_type == 'rigid' else 'blue'
        linewidth = 2 if conn_type == 'rigid' else 1
        
        for conn in connections:
            p1, p2 = conn
            x1, y1 = point_positions[p1]
            x2, y2 = point_positions[p2]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth)
    
    # Draw keypoints
    for point_id, (x, y) in point_positions.items():
        ax.scatter(x, y, s=100, c='black', zorder=5)
        ax.annotate(f"{point_id}:{keypoints[point_id]}", (x, y), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='red', lw=2, label='hard connection'),
                      Line2D([0], [0], color='blue', lw=1, label='soft connection')]
    ax.legend(handles=legend_elements)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("CUB-15 points skeleton structure")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Skeleton structure saved: {save_path}")

def check_cub_structure(cub_root_path):
    """Check CUB dataset structure"""
    print(f"Check path: {cub_root_path}")
    cub_root = Path(cub_root_path)
    
    if not cub_root.exists():
        print(f"Root directory does not exist: {cub_root_path}")
        return False
    
    print("Files and folders found:")
    for item in sorted(cub_root.iterdir()):
        if item.is_dir():
            print(f"  Folder {item.name}/")
            # Check subfolder content
            if item.name == "parts":
                print("    parts folder content:")
                for subitem in sorted(item.iterdir()):
                    print(f"      File {subitem.name}")
        else:
            print(f"  File {item.name}")
    
    # Check key files
    key_files = [
        "parts/part_locs.txt",
        "images.txt", 
        "train_test_split.txt"
    ]
    
    print("\nCheck key files:")
    all_exist = True
    for file_path in key_files:
        full_path = cub_root / file_path
        if full_path.exists():
            print(f"  Found {file_path}")
        else:
            print(f"  Missing {file_path}")
            all_exist = False
    
    return all_exist

def main():
    """Phase 1 main function"""
    print("Phase 1: CUB skeleton definition\n")
    
    # Set CUB dataset path
    cub_root_path = "./CUB"  # Match your dataset folder
    
    # Step 1. Check dataset structure
    print("1. Checking dataset structure...")
    if not check_cub_structure(cub_root_path):
        print("\nDataset structure check failed, please confirm:")
        print("1. Is the CUB folder path correct?")
        print("2. Is it the complete CUB-200-2011 dataset?")
        print("3. Do key files exist?")
        return
    
    # Step 2. Load and validate data
    print("\n2. Validating CUB data...")
    parts_df, images_df, split_df = load_cub_data(cub_root_path)
    
    
    # Step 3. Create skeleton configuration
    print("\n3. Creating skeleton configuration...")
    skeleton_config = create_cub_skeleton_config()
    
    # Step 4. Save configuration file
    print("\n4. Saving configuration file...")
    with open("skeleton.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(skeleton_config, f, default_flow_style=False, allow_unicode=True, indent=2)
    print("skeleton.yaml saved")
    
    # Step 5. Visualize skeleton
    print("\n5. Generating skeleton structure figure...")
    visualize_skeleton(skeleton_config)
    
    # Step 6. Create dataset class
    print("\n6. Creating dataset class...")
    create_cub_dataset_class()
    
    print("\nPhase 1 completed!")
    print("Output files:")
    print("- skeleton.yaml: skeleton configuration")  
    print("- skeleton_structure.png: skeleton figure")
    print("- cub_dataset.py: dataset loader")
    print("\nPrepare Phase 2: HRNet training")

if __name__ == "__main__":
    main()
