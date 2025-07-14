
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_demo_infer.py img.jpg
------------------------
推理单张图片 → 显示/保存 result.jpg
✨ 适配新的配置和增强可视化
"""

import sys
import cv2
import numpy as np
import yaml
import torch
from pathlib import Path
from mmpose.apis import init_model, inference_topdown

# ================================================================
# 路径配置
# ================================================================
WORK_DIR = Path("workdir/hrnet17_simple")
CFG = WORK_DIR / "bird17_hrnet_simple.py"
CKPT = WORK_DIR / "bird17_detector.pth"
SKELETON_YAML = Path("skeleton.yaml")

def load_skeleton_info():
    """加载骨架连接信息"""
    try:
        with open(SKELETON_YAML) as f:
            skeleton_data = yaml.safe_load(f)
        return skeleton_data.get("skeleton", [])
    except:
        print("⚠️  无法加载骨架信息，将只显示关键点")
        return []

def draw_enhanced_pose(img, keypoints, skeleton_connections=None, confidence_threshold=0.3):
    """
    绘制增强的姿态可视化
    Args:
        img: 输入图像
        keypoints: 关键点坐标 (17, 2) 或 (17, 3)
        skeleton_connections: 骨架连接信息
        confidence_threshold: 置信度阈值
    """
    img_vis = img.copy()
    h, w = img.shape[:2]
    
    # 如果关键点有置信度信息
    if keypoints.shape[1] == 3:
        kpts_xy = keypoints[:, :2]
        kpts_conf = keypoints[:, 2]
    else:
        kpts_xy = keypoints
        kpts_conf = np.ones(len(keypoints))  # 假设都可见
    
    # 绘制骨架连接
    if skeleton_connections:
        for connection in skeleton_connections:
            pt1_idx, pt2_idx = connection
            
            # 检查关键点是否有效且置信度足够
            if (kpts_conf[pt1_idx] > confidence_threshold and 
                kpts_conf[pt2_idx] > confidence_threshold and
                kpts_xy[pt1_idx][0] > 0 and kpts_xy[pt1_idx][1] > 0 and
                kpts_xy[pt2_idx][0] > 0 and kpts_xy[pt2_idx][1] > 0):
                
                pt1 = tuple(map(int, kpts_xy[pt1_idx]))
                pt2 = tuple(map(int, kpts_xy[pt2_idx]))
                
                # 根据置信度调整线条粗细和颜色
                avg_conf = (kpts_conf[pt1_idx] + kpts_conf[pt2_idx]) / 2
                thickness = max(1, int(3 * avg_conf))
                color_intensity = int(255 * avg_conf)
                
                cv2.line(img_vis, pt1, pt2, (0, color_intensity, 0), thickness)
    
    # 绘制关键点
    for i, ((x, y), conf) in enumerate(zip(kpts_xy, kpts_conf)):
        if conf > confidence_threshold and x > 0 and y > 0:
            # 根据置信度调整点的大小和颜色
            radius = max(2, int(6 * conf))
            color_intensity = int(255 * conf)
            
            # 绘制关键点
            cv2.circle(img_vis, (int(x), int(y)), radius, (0, 0, color_intensity), -1)
            cv2.circle(img_vis, (int(x), int(y)), radius+1, (255, 255, 255), 1)
            
            # 可选：显示关键点编号
            if conf > 0.5:  # 只在高置信度时显示编号
                cv2.putText(img_vis, str(i), (int(x)+8, int(y)-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return img_vis

def add_info_overlay(img, keypoints, skeleton_connections):
    """添加信息覆盖层"""
    img_info = img.copy()
    h, w = img.shape[:2]
    
    # 创建半透明覆盖层
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 计算统计信息
    if keypoints.shape[1] == 3:
        visible_count = np.sum(keypoints[:, 2] > 0.3)
        avg_confidence = np.mean(keypoints[keypoints[:, 2] > 0, 2])
    else:
        visible_count = np.sum((keypoints[:, 0] > 0) & (keypoints[:, 1] > 0))
        avg_confidence = 1.0
    
    # 添加文本信息
    info_texts = [
        f"Detected Keypoints: {visible_count}/17",
        f"Avg Confidence: {avg_confidence:.3f}",
        f"Model: Bird17 HRNet",
        f"Framework: MMPose"
    ]
    
    # 绘制信息框
    text_area_height = len(info_texts) * 25 + 20
    cv2.rectangle(overlay, (10, 10), (300, text_area_height), (50, 50, 50), -1)
    
    for i, text in enumerate(info_texts):
        y_pos = 30 + i * 25
        cv2.putText(overlay, text, (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # 混合覆盖层
    alpha = 0.7
    img_info = cv2.addWeighted(img_info, 1-alpha, overlay, alpha, 0)
    
    return img_info

def main():
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("使用方法: python 05_demo_infer.py <image_path>")
        print("示例: python 05_demo_infer.py bird.jpg")
        return
    
    img_path = Path(sys.argv[1])
    
    if not img_path.exists():
        print(f"❌ 图像文件不存在: {img_path}")
        return
    
    print(f"🦜 开始推理鸟类姿态: {img_path}")
    print("=" * 50)
    
    # 检查模型文件
    if not CFG.exists():
        print(f"❌ 配置文件不存在: {CFG}")
        return
    
    if not CKPT.exists():
        print(f"❌ 模型文件不存在: {CKPT}")
        return
    
    # 加载模型
    print("📦 加载模型...")
    try:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = init_model(str(CFG), str(CKPT), device=device)
        print(f"✅ 模型加载成功 (设备: {device})")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 加载骨架信息
    skeleton_connections = load_skeleton_info()
    
    # 推理
    print("🔍 执行姿态检测...")
    try:
        # 新版本MMPose的推理方式
        results = inference_topdown(model, str(img_path))
        
        # 修复：正确访问返回结果
        if not results or len(results) == 0:
            print("❌ 未检测到有效的姿态")
            return
        
        # 获取第一个结果
        result = results[0]
        
        # 检查是否有pred_instances
        if not hasattr(result, 'pred_instances') or len(result.pred_instances.keypoints) == 0:
            print("❌ 未检测到有效的姿态")
            return
            
        keypoints = result.pred_instances.keypoints[0]
        print(f"✅ 检测成功，找到 {len(keypoints)} 个关键点")
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        return
    
    # 读取原图
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ 无法读取图像: {img_path}")
        return
    
    # 生成可视化结果
    print("🎨 生成可视化结果...")
    
    # 基础姿态可视化
    img_pose = draw_enhanced_pose(img, keypoints, skeleton_connections)
    
    # 带信息覆盖的版本
    img_info = add_info_overlay(img_pose, keypoints, skeleton_connections)
    
    # 保存结果
    output_files = []
    
    # 保存基础版本
    basic_output = img_path.with_suffix(f".pose{img_path.suffix}")
    cv2.imwrite(str(basic_output), img_pose)
    output_files.append(basic_output)
    
    # 保存信息版本
    info_output = img_path.with_suffix(f".pose_info{img_path.suffix}")
    cv2.imwrite(str(info_output), img_info)
    output_files.append(info_output)
    
    # 并排对比版本
    comparison = np.hstack([img, img_pose])
    comp_output = img_path.with_suffix(f".comparison{img_path.suffix}")
    cv2.imwrite(str(comp_output), comparison)
    output_files.append(comp_output)
    
    print("✅ 姿态检测完成!")
    print("📁 输出文件:")
    for output_file in output_files:
        print(f"   - {output_file}")
    
    # 显示关键统计信息
    if keypoints.shape[1] == 3:
        visible_kpts = np.sum(keypoints[:, 2] > 0.3)
        avg_conf = np.mean(keypoints[keypoints[:, 2] > 0, 2])
        print(f"\n📊 检测统计:")
        print(f"   可见关键点: {visible_kpts}/17")
        print(f"   平均置信度: {avg_conf:.3f}")
    
    print(f"\n🎯 现在可以继续Pipeline步骤1.3!")

if __name__ == "__main__":
    main()