#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_prepare_dataset.py
--------------------
把 CUB-200-2011(15点) 与 AP-10K(bird) 标注
→ 统一 17 点 COCO-Keypoint JSON
✨ 新增：几何估算缺失关键点，自动计算bbox和图像尺寸
"""

import json, random, shutil, cv2
from pathlib import Path
import numpy as np
import yaml

ROOT = Path("data")  # ➜ data/
CUB_ROOT = ROOT/"CUB"
AP10_ROOT = ROOT/"AP10k"
OUT_ROOT = ROOT/"merged17"  # ➜ data/merged17/
SEED = 42
SKE_YAML = Path("skeleton.yaml")  # 已由 01_define_skeleton.py 生成
split_ratio = dict(train=.8, val=.1, test=.1)

# 1) 读取 17 点 schema（顺序 + 名称）
schema = yaml.safe_load(SKE_YAML.read_text())["joints"]
idx2name = [j[1] for j in schema]  # 17-length list

def estimate_virtual_keypoints(kpts15):
    """
    从CUB-15关键点估算缺失的关键点，创建完整的17点结构
    让鸟类能够"模拟"四足动物的骨架
    """
    kpts17 = np.zeros((17, 3))
    
    # 原有的9个直接映射
    map15_17 = {
        6: 0,   # left_eye
        10: 1,  # right_eye
        1: 2,   # beak → nose
        9: 3,   # nape → neck
        13: 4,  # tail tip
        8: 5,   # left_wing root → left_shoulder
        12: 8,  # right_wing root → right_shoulder
        7: 11,  # left_leg root → left_hip
        11: 14  # right_leg root → right_hip
    }
    
    # 直接映射已知点
    for cub_idx, apt_idx in map15_17.items():
        if cub_idx < 15 and apt_idx < 17:
            kpts17[apt_idx] = kpts15[cub_idx]
    
    # 辅助函数：安全获取关键点
    def safe_point(idx):
        return kpts15[idx] if idx < 15 and kpts15[idx, 2] > 0 else None
    
    # 获取参考点
    left_eye = safe_point(6)
    right_eye = safe_point(10)
    left_wing = safe_point(8)
    right_wing = safe_point(12)
    left_leg = safe_point(7)
    right_leg = safe_point(11)
    neck = safe_point(9)  # nape
    beak = safe_point(1)
    
    # 估算"肘部"（翅膀中点）- 让翅膀有"上臂-前臂"结构
    if left_wing is not None and neck is not None:
        kpts17[6] = [(left_wing[0] + neck[0])/2, (left_wing[1] + neck[1])/2, 1.0]  # left_elbow
    
    if right_wing is not None and neck is not None:
        kpts17[9] = [(right_wing[0] + neck[0])/2, (right_wing[1] + neck[1])/2, 1.0]  # right_elbow
    
    # 估算"前爪"（翅膀尖端延伸）- 模拟四足动物前爪
    if left_wing is not None and neck is not None:
        dx, dy = left_wing[0] - neck[0], left_wing[1] - neck[1]
        kpts17[7] = [left_wing[0] + dx*0.3, left_wing[1] + dy*0.3, 1.0]  # left_front_paw
    
    if right_wing is not None and neck is not None:
        dx, dy = right_wing[0] - neck[0], right_wing[1] - neck[1]
        kpts17[10] = [right_wing[0] + dx*0.3, right_wing[1] + dy*0.3, 1.0]  # right_front_paw
    
    # 估算"膝盖"（腿部中点）- 让腿有"大腿-小腿"结构
    if left_leg is not None and neck is not None:
        kpts17[12] = [(left_leg[0] + neck[0])/2, (left_leg[1] + neck[1])/2, 1.0]  # left_knee
    
    if right_leg is not None and neck is not None:
        kpts17[15] = [(right_leg[0] + neck[0])/2, (right_leg[1] + neck[1])/2, 1.0]  # right_knee
    
    # 估算"后爪"（脚部延伸）- 模拟四足动物后爪
    if left_leg is not None:
        kpts17[13] = [left_leg[0], left_leg[1] + 15, 1.0]  # left_back_paw
    
    if right_leg is not None:
        kpts17[16] = [right_leg[0], right_leg[1] + 15, 1.0]  # right_back_paw
    
    return kpts17

def calculate_bbox_and_area(keypoints, img_width, img_height):
    """从关键点计算有效的bbox"""
    visible_kpts = keypoints[keypoints[:, 2] > 0]
    
    if len(visible_kpts) < 2:
        return None, None
    
    x_coords, y_coords = visible_kpts[:, 0], visible_kpts[:, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # 添加边距
    margin = 20
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(img_width, x_max + margin)
    y_max = min(img_height, y_max + margin)
    
    bbox_w, bbox_h = x_max - x_min, y_max - y_min
    
    # 确保bbox有效
    if bbox_w > 10 and bbox_h > 10:
        bbox = [float(x_min), float(y_min), float(bbox_w), float(bbox_h)]
        area = float(bbox_w * bbox_h)
        return bbox, area
    
    return None, None

# ------------ C U B 15 ➜ 17 (改进版) ------------
def convert_cub():
    """
    将 CUB-200-2011 15 点部位 → 对齐到 APT/AP-10K 的 17 点骨架。
    ✨ 新增几何估算，提高关键点质量
    """
    print("🔄 处理CUB-200数据集...")
    
    img_map = {}  # image_id -> filepath
    for ln in (CUB_ROOT / "images.txt").read_text().splitlines():
        _id, fp = ln.strip().split()
        img_map[int(_id)] = CUB_ROOT / "images" / fp

    # CUB 原生：part_locs.txt 15 ×(x,y,v)
    kpt_map = {}  # image_id -> 15×3
    for ln in (CUB_ROOT / "parts" / "part_locs.txt").read_text().splitlines():
        img_id, part_id, x, y, vis = map(float, ln.strip().split())
        img_id, part_id = int(img_id), int(part_id)  # part_id ∈[1,15]
        kpt_map.setdefault(img_id, np.zeros((15, 3)))
        kpt_map[img_id][part_id - 1] = (x, y, vis)

    annos, images = [], []
    aid = 1
    processed = 0
    skipped = 0

    for img_id, kpts15 in kpt_map.items():
        if img_id not in img_map:
            continue
            
        img_path = img_map[img_id]
        
        # 读取图像获取真实尺寸
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            h, w = img.shape[:2]
        except Exception as e:
            print(f"⚠️  无法读取图像 {img_path}: {e}")
            skipped += 1
            continue

        # 使用改进的估算方法生成17个关键点
        kpts17 = estimate_virtual_keypoints(kpts15)
        
        # 质量控制：至少需要8个可见关键点
        visible_count = np.sum(kpts17[:, 2] > 0)
        if visible_count < 8:
            skipped += 1
            continue
        
        # 计算有效的bbox
        bbox, area = calculate_bbox_and_area(kpts17, w, h)
        if bbox is None:
            skipped += 1
            continue

        # 保存图像信息
        images.append({
            'id': img_id,
            'file_name': str(img_path.relative_to(ROOT)),
            'width': w,
            'height': h
        })

        # 保存标注信息
        annos.append({
            'id': aid,
            'image_id': img_id,
            'category_id': 1,
            'keypoints': kpts17.reshape(-1).tolist(),
            'num_keypoints': int(visible_count),
            'bbox': bbox,
            'area': area,
            'iscrowd': 0,
            'segmentation': []
        })
        aid += 1
        processed += 1

    print(f"✅ CUB处理完成: {processed} 个有效样本, {skipped} 个跳过")
    return images, annos

# ------------ A P - 1 0 K (bird subset) ------------
def convert_ap10k():
    """
    读取 ap10k_train/val/test JSON
    → 仅保留 '鸟类' 相关 category
    ✨ 新增bbox和图像尺寸修复
    """
    print("🔄 处理AP-10K鸟类数据...")
    
    ap10k_json = AP10_ROOT / "annotations" / "ap10k_train.json"
    if not ap10k_json.exists():
        print("⚠️  AP-10K数据不存在，跳过")
        return [], []
    
    js = json.loads(ap10k_json.read_text())
    
    bird_kw = [
        "bird", "gull", "swan", "duck", "goose", "owl", "eagle", "falcon",
        "penguin", "albatross", "parrot", "sparrow", "woodpecker", "pigeon",
        "dove", "crow", "raven", "peacock", "kingfisher", "rooster", "hen"
    ]

    bird_cat_ids = [
        c["id"] for c in js["categories"]
        if "ave" in c.get("supercategory", "").lower()
        or "bird" in c.get("supercategory", "").lower()
        or any(kw in c["name"].lower() for kw in bird_kw)
    ]

    if not bird_cat_ids:
        print("⚠️  AP-10K中未找到鸟类类别，跳过")
        return [], []

    # 建立 image_id → file_name 映射
    valid_img_ids = set(
        a["image_id"] for a in js["annotations"] if a["category_id"] in bird_cat_ids
    )

    # 处理图像信息，修复尺寸
    processed_images = []
    img_info_map = {img['id']: img for img in js["images"]}
    
    for img_id in valid_img_ids:
        if img_id not in img_info_map:
            continue
            
        img_info = img_info_map[img_id]
        img_path = AP10_ROOT / img_info['file_name']
        
        # 如果尺寸为0，读取真实尺寸
        if img_info.get('width', 0) == 0 or img_info.get('height', 0) == 0:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    img_info['width'] = w
                    img_info['height'] = h
            except:
                continue
        
        if img_info.get('width', 0) > 0 and img_info.get('height', 0) > 0:
            processed_images.append(img_info)

    # 处理标注信息，修复bbox
    processed_annos = []
    for a in js["annotations"]:
        if a["image_id"] not in valid_img_ids:
            continue
            
        # 统一category_id
        a["category_id"] = 1
        
        # 修复bbox（如果需要）
        if a.get("bbox", [0,0,0,0]) == [0,0,0,0] or any(x <= 0 for x in a.get("bbox", [])):
            img_info = img_info_map.get(a["image_id"])
            if img_info:
                keypoints = np.array(a["keypoints"]).reshape(-1, 3)
                bbox, area = calculate_bbox_and_area(keypoints, img_info['width'], img_info['height'])
                if bbox:
                    a["bbox"] = bbox
                    a["area"] = area
        
        # 确保必要字段存在
        if "iscrowd" not in a:
            a["iscrowd"] = 0
        if "segmentation" not in a:
            a["segmentation"] = []
            
        processed_annos.append(a)

    print(f"✅ AP-10K处理完成: {len(processed_images)} 图像, {len(processed_annos)} 标注")
    return processed_images, processed_annos

# 4) 合并 & 划分 --------------------------------------------------------
def main():
    print("🦜 开始准备鸟类17关键点数据集...")
    print("=" * 60)
    
    random.seed(SEED)
    
    im_cub, an_cub = convert_cub()
    im_ap, an_ap = convert_ap10k()

    images = im_cub + im_ap
    annos = an_cub + an_ap

    print(f"\n📊 数据统计:")
    print(f"   总图像数: {len(images)}")
    print(f"   总标注数: {len(annos)}")
    if annos:
        avg_kpts = np.mean([ann['num_keypoints'] for ann in annos])
        print(f"   平均可见关键点: {avg_kpts:.1f}")

    if not images:
        print("❌ 没有有效数据，请检查数据路径")
        return

    # 随机打乱
    random.shuffle(images)
    
    N = len(images)
    splitN = {k: int(r * N) for k, r in split_ratio.items()}
    
    # 按图像ID划分
    idx = dict()
    s = 0
    for k, n in splitN.items():
        idx[k] = set(i["id"] for i in images[s:s+n])
        s += n

    # 保存各个split
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    for split in ("train", "val", "test"):
        split_images = [i for i in images if i["id"] in idx[split]]
        split_annos = [a for a in annos if a["image_id"] in idx[split]]
        
        out = {
            'images': split_images,
            'annotations': split_annos,
            'categories': [{'id': 1, 'name': 'bird'}]
        }
        
        output_file = OUT_ROOT / f"{split}.json"
        output_file.write_text(json.dumps(out, indent=2))
        
        print(f"✅ {split:5}: {len(split_images):4} 图像, {len(split_annos):4} 标注 → {output_file}")

    print(f"\n🎉 17关键点COCO JSON已生成到: {OUT_ROOT}")
    print("   数据质量已优化，bbox已自动计算，可直接用于训练！")

if __name__ == "__main__":
    main()