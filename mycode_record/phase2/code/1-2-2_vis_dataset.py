#!/usr/bin/env python
# 02-2_vis_dataset.py
# 可视化改进后的数据集，显示更多统计信息

import json, random, cv2, yaml
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

ANN = Path("data/merged17/train.json")  # 使用改进后的数据
OUT = Path("vis_dataset"); OUT.mkdir(exist_ok=True)

sche = yaml.safe_load(Path("skeleton.yaml").read_text())
edges = sche["skeleton"]

def draw(img, kp):
    """绘制关键点和骨架"""
    # 绘制骨架连线
    for (i, j) in edges:
        if kp[i, 2] > 0 and kp[j, 2] > 0:
            cv2.line(img, tuple(kp[i, :2].astype(int)), tuple(kp[j, :2].astype(int)), (0, 255, 0), 2)
    
    # 绘制关键点
    for idx, p in enumerate(kp):
        if p[2] > 0:
            color = (0, 0, 255)  # 红色为可见点
            cv2.circle(img, tuple(p[:2].astype(int)), 4, color, -1)
            # 可选：显示关键点编号
            cv2.putText(img, str(idx), tuple(p[:2].astype(int) + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return img

def main():
    print("🔍 可视化改进后的鸟类数据集")
    print("=" * 50)
    
    if not ANN.exists():
        print(f"❌ 数据文件不存在: {ANN}")
        print("请先运行 02-1_prepare_dataset.py")
        return
    
    js = json.loads(ANN.read_text())
    
    print(f"📊 数据集统计:")
    print(f"   图像数量: {len(js['images'])}")
    print(f"   标注数量: {len(js['annotations'])}")
    print(f"   类别: {js['categories']}")
    
    if not js['annotations']:
        print("❌ 没有标注数据")
        return
    
    # 分析关键点质量
    keypoint_counts = [ann['num_keypoints'] for ann in js['annotations']]
    bbox_areas = [ann.get('area', 0) for ann in js['annotations']]
    
    print(f"\n🎯 关键点质量分析:")
    print(f"   可见关键点数量:")
    print(f"     最少: {min(keypoint_counts)}")
    print(f"     最多: {max(keypoint_counts)}")
    print(f"     平均: {np.mean(keypoint_counts):.1f}")
    print(f"     中位数: {np.median(keypoint_counts):.1f}")
    
    print(f"\n📦 Bbox质量分析:")
    valid_bboxes = [area for area in bbox_areas if area > 0]
    print(f"   有效bbox数量: {len(valid_bboxes)}/{len(bbox_areas)}")
    if valid_bboxes:
        print(f"   平均面积: {np.mean(valid_bboxes):.0f} 像素²")
    
    # 按关键点数量分组统计
    print(f"\n📈 关键点分布:")
    for kpt_count in range(min(keypoint_counts), max(keypoint_counts) + 1):
        count = sum(1 for x in keypoint_counts if x == kpt_count)
        if count > 0:
            print(f"   {kpt_count:2d} 个点: {count:4d} 个样本")
    
    # 可视化样本
    print(f"\n🎨 生成可视化样本...")
    
    # 选择不同质量的样本进行可视化
    samples_to_vis = []
    
    # 高质量样本（关键点多）
    high_quality = [ann for ann in js['annotations'] if ann['num_keypoints'] >= 12]
    if high_quality:
        samples_to_vis.extend(random.sample(high_quality, min(3, len(high_quality))))
    
    # 中等质量样本
    medium_quality = [ann for ann in js['annotations'] if 8 <= ann['num_keypoints'] < 12]
    if medium_quality:
        samples_to_vis.extend(random.sample(medium_quality, min(3, len(medium_quality))))
    
    # 低质量样本
    low_quality = [ann for ann in js['annotations'] if ann['num_keypoints'] < 8]
    if low_quality:
        samples_to_vis.extend(random.sample(low_quality, min(2, len(low_quality))))
    
    # 如果样本不够，随机补充
    while len(samples_to_vis) < 10 and len(js['annotations']) > len(samples_to_vis):
        remaining = [ann for ann in js['annotations'] if ann not in samples_to_vis]
        samples_to_vis.extend(random.sample(remaining, min(10 - len(samples_to_vis), len(remaining))))
    
    saved_count = 0
    for ann in samples_to_vis:
        # 找到对应的图像信息
        img_info = next((i for i in js["images"] if i["id"] == ann["image_id"]), None)
        if not img_info:
            continue
            
        path = Path("data") / img_info["file_name"]
        kp = np.array(ann["keypoints"]).reshape(-1, 3)

        # 读取并可视化图像
        im = cv2.imread(str(path))
        if im is None:
            print(f"⚠️  无法读取图像: {path}")
            continue

        # 绘制关键点和骨架
        vis = draw(im.copy(), kp)
        
        # 添加信息文字
        info_text = f"ID:{img_info['id']} Points:{ann['num_keypoints']} BBox:{ann.get('area', 0):.0f}"
        cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 保存可视化结果
        output_file = OUT / f"{img_info['id']:06d}_kpts{ann['num_keypoints']:02d}.jpg"
        cv2.imwrite(str(output_file), vis)
        saved_count += 1
        
        print(f"   保存: {output_file.name} (关键点: {ann['num_keypoints']})")

    print(f"\n✅ 可视化完成!")
    print(f"   保存了 {saved_count} 个样本到: {OUT}")
    print(f"   可以查看不同质量的关键点标注效果")
    
    # 如果数据质量改善明显，给出提示
    if np.mean(keypoint_counts) >= 10:
        print(f"\n🎉 数据质量良好! 平均 {np.mean(keypoint_counts):.1f} 个可见关键点")
        print(f"   现在可以继续运行 03_train_hrnet.py 进行训练")
    elif np.mean(keypoint_counts) >= 8:
        print(f"\n✅ 数据质量可接受, 平均 {np.mean(keypoint_counts):.1f} 个可见关键点")
        print(f"   可以尝试训练，但可能需要调整训练参数")
    else:
        print(f"\n⚠️  数据质量仍需改进, 平均只有 {np.mean(keypoint_counts):.1f} 个可见关键点")
        print(f"   建议检查数据预处理过程")

if __name__ == "__main__":
    main()