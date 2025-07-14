#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_eval_hrnet.py - 清理版本
----------------
在 val.json 上跑推理 → 计算 PCK@0.2
+ 输出可视化结果，验证模型性能
"""

import numpy as np
import cv2
import json
import random
import torch
import yaml
import datetime
import time
from pathlib import Path
from mmpose.apis import inference_topdown, init_model

# 配置
# WORK_DIR = Path("workdir/hrnet17")
WORK_DIR = Path("workdir/hrnet17_simple")
# CFG = WORK_DIR / "bird17_hrnet_w32.py"
CFG = WORK_DIR / "bird17_hrnet_simple.py"
CKPT = WORK_DIR / "bird17_detector.pth"
VAL_JSON = Path("data/merged17/val.json")
SKELETON_YAML = Path("skeleton.yaml")
VIS_OUT = Path("vis_eval")
LOGBOOK_FILE = WORK_DIR / "evaluation_logbook.md"
VIS_OUT.mkdir(exist_ok=True)

N_SAMPLES = 500
VIS_RATIO = 0.1
PCK_THRESHOLD = 0.2

class EvaluationLogger:
    def __init__(self, logbook_path):
        self.logbook_path = Path(logbook_path)
        self.start_time = None
        self.eval_stats = {}
        self.logbook_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_logbook()
    
    def _init_logbook(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"""# 🔍 Bird 17-Keypoint HRNet Evaluation Logbook

**评估开始时间**: {timestamp}
**项目**: 鸟类17关键点检测器性能评估
**模型**: HRNet-W32 (`bird17_detector.pth`)

---

## 📋 评估配置

| 参数 | 值 |
|------|-----|
| 评估样本数 | {N_SAMPLES} |
| PCK阈值 | {PCK_THRESHOLD} |
| 设备 | {'CUDA' if torch.cuda.is_available() else 'CPU'} |

---

## 🚀 评估过程

"""
        self.logbook_path.write_text(header)
    
    def log_dataset_info(self, stats):
        stats_text = f"""
| 统计项 | 数值 |
|--------|------|
| 验证图像数 | {stats.get('val_images', 0)} |
| 验证标注数 | {stats.get('val_annotations', 0)} |
| 平均关键点数 | {stats.get('avg_keypoints', 0):.1f} |

"""
        with open(self.logbook_path, 'a') as f:
            f.write(stats_text)
    
    def log_evaluation_start(self):
        self.start_time = time.time()
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_text = f"""
### {timestamp} - 评估开始
- ✅ 模型加载成功
- 🔍 开始批量推理...

"""
        with open(self.logbook_path, 'a') as f:
            f.write(log_text)
    
    def log_evaluation_complete(self, results, success=True, error_msg=None):
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        duration_str = f"{duration//3600:.0f}h {(duration%3600)//60:.0f}m {duration%60:.0f}s"
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        if success and results:
            mean_pck = results.get('mean_pck', 0)
            if mean_pck > 0.7:
                performance = "🌟 优秀"
            elif mean_pck > 0.5:
                performance = "✅ 良好"
            elif mean_pck > 0.3:
                performance = "⚠️ 一般"
            else:
                performance = "❌ 较差"
            
            log_text = f"""
### {timestamp} - 评估完成 ✅

- ⏱️ **总耗时**: {duration_str}
- 📊 **整体性能**: {performance}
- 🎯 **PCK@{PCK_THRESHOLD}**: {mean_pck:.4f}

## 📈 详细结果

| 指标 | 数值 |
|------|------|
| PCK均值 | {mean_pck:.4f} |
| PCK标准差 | {results.get('std_pck', 0):.4f} |
| 最佳PCK | {results.get('max_pck', 0):.4f} |
| 最差PCK | {results.get('min_pck', 0):.4f} |
| 成功率 | {results.get('success_rate', 0):.1%} |

**评估完成时间**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        else:
            log_text = f"""
### {timestamp} - 评估失败 ❌

- ⏱️ **耗时**: {duration_str}
- ❌ **错误**: {error_msg or "未知错误"}

**失败时间**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        with open(self.logbook_path, 'a') as f:
            f.write(log_text)

eval_logger = EvaluationLogger(LOGBOOK_FILE)

def check_files():
    missing_files = []
    if not CFG.exists():
        missing_files.append(f"配置文件: {CFG}")
    if not CKPT.exists():
        missing_files.append(f"模型文件: {CKPT}")
    if not VAL_JSON.exists():
        missing_files.append(f"验证数据: {VAL_JSON}")
    
    if missing_files:
        print("❌ 缺少必要文件:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ 所有必要文件检查通过")
    return True

def calculate_pck(gt_keypoints, pred_keypoints, img_width, img_height, threshold=0.2):
    dists = np.linalg.norm(gt_keypoints - pred_keypoints, axis=1)
    normalize_factor = max(img_width, img_height)
    normalized_dists = dists / normalize_factor
    pck = (normalized_dists < threshold).mean()
    return pck, normalized_dists

def draw_keypoints_and_skeleton(img, keypoints, skeleton_info):
    img_vis = img.copy()
    
    if skeleton_info:
        for connection in skeleton_info:
            pt1_idx, pt2_idx = connection
            if (keypoints[pt1_idx][0] > 0 and keypoints[pt1_idx][1] > 0 and 
                keypoints[pt2_idx][0] > 0 and keypoints[pt2_idx][1] > 0):
                
                pt1 = tuple(map(int, keypoints[pt1_idx]))
                pt2 = tuple(map(int, keypoints[pt2_idx]))
                cv2.line(img_vis, pt1, pt2, (0, 255, 0), 2)
    
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:
            cv2.circle(img_vis, (int(x), int(y)), 4, (0, 0, 255), -1)
    
    return img_vis

def main():
    print("🔍 开始评估鸟类17关键点检测器")
    print("=" * 60)
    
    # 检查文件
    if not check_files():
        eval_logger.log_evaluation_complete({}, success=False, error_msg="必要文件缺失")
        return
    
    # 加载模型
    print("📦 加载模型...")
    try:
        model = init_model(
            str(CFG), 
            str(CKPT), 
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            cfg_options={"test_cfg": {"flip_test": False}}
        )
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        eval_logger.log_evaluation_complete({}, success=False, error_msg=f"模型加载失败: {e}")
        return
    
    # 加载验证数据
    print("📊 加载验证数据...")
    with open(VAL_JSON) as f:
        val_data = json.load(f)
    
    annotations = val_data["annotations"]
    images = {img["id"]: img for img in val_data["images"]}
    
    # 数据集统计
    keypoint_counts = [ann['num_keypoints'] for ann in annotations]
    dataset_stats = {
        'val_images': len(images),
        'val_annotations': len(annotations),
        'avg_keypoints': np.mean(keypoint_counts)
    }
    
    eval_logger.log_dataset_info(dataset_stats)
    
    # 加载骨架信息
    with open(SKELETON_YAML) as f:
        skeleton_data = yaml.safe_load(f)
    skeleton_connections = skeleton_data.get("skeleton", [])
    
    # 开始评估
    print(f"🎯 开始评估 (抽样 {N_SAMPLES} 个)")
    eval_logger.log_evaluation_start()
    
    if len(annotations) < N_SAMPLES:
        eval_annotations = annotations
    else:
        eval_annotations = random.sample(annotations, N_SAMPLES)
    
    pck_scores = []
    valid_predictions = 0
    vis_count = 0
    
    for i, ann in enumerate(eval_annotations):
        if i % 50 == 0:
            print(f"   进度: {i+1}/{len(eval_annotations)}")
        
        img_info = images[ann["image_id"]]
        img_path = Path("data") / img_info["file_name"]
        
        if not img_path.exists():
            continue
        
        try:
            # 新版本MMPose的推理方式
            results = inference_topdown(model, str(img_path))
            
            # 修复：正确访问返回结果
            if not results or len(results) == 0:
                continue
            
            # 获取第一个结果
            result = results[0]
            
            # 检查是否有pred_instances
            if not hasattr(result, 'pred_instances') or len(result.pred_instances.keypoints) == 0:
                continue
            
            pred_keypoints = result.pred_instances.keypoints[0]  # (17, 2)
            gt_keypoints = np.array(ann["keypoints"]).reshape(-1, 3)[:, :2]  # (17, 2)
            
            pck, _ = calculate_pck(
                gt_keypoints, pred_keypoints, 
                img_info["width"], img_info["height"], 
                PCK_THRESHOLD
            )
            
            pck_scores.append(pck)
            valid_predictions += 1
            
            # 可视化
            if random.random() < VIS_RATIO and vis_count < 20:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_pred = draw_keypoints_and_skeleton(img, pred_keypoints, skeleton_connections)
                    
                    for x, y in gt_keypoints:
                        if x > 0 and y > 0:
                            cv2.circle(img_pred, (int(x), int(y)), 3, (0, 255, 0), 2)
                    
                    info_text = f"PCK: {pck:.3f}, ID: {ann['image_id']}"
                    cv2.putText(img_pred, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    output_file = VIS_OUT / f"eval_{ann['image_id']:06d}_pck{pck:.3f}.jpg"
                    cv2.imwrite(str(output_file), img_pred)
                    vis_count += 1
                    
        except Exception as e:
            print(f"⚠️  处理图像时出错: {e}")
            continue
    
    # 计算最终结果
    if pck_scores:
        mean_pck = np.mean(pck_scores)
        std_pck = np.std(pck_scores)
        max_pck = max(pck_scores)
        min_pck = min(pck_scores)
        
        print(f"\n📈 评估结果:")
        print(f"   有效预测数: {valid_predictions}/{len(eval_annotations)}")
        print(f"   PCK@{PCK_THRESHOLD}: {mean_pck:.4f} ± {std_pck:.4f}")
        print(f"   最佳PCK: {max_pck:.4f}")
        print(f"   最差PCK: {min_pck:.4f}")
        
        # 性能评估
        if mean_pck > 0.7:
            print("🎉 模型性能优秀!")
        elif mean_pck > 0.5:
            print("✅ 模型性能良好")
        elif mean_pck > 0.3:
            print("⚠️  模型性能一般")
        else:
            print("❌ 模型性能较差，建议重新训练")
            
        eval_results = {
            'mean_pck': float(mean_pck),
            'std_pck': float(std_pck),
            'max_pck': float(max_pck),
            'min_pck': float(min_pck),
            'valid_predictions': valid_predictions,
            'total_samples': len(eval_annotations),
            'success_rate': valid_predictions / len(eval_annotations),
            'threshold': PCK_THRESHOLD
        }
        
        results_file = WORK_DIR / "eval_results.json"
        with open(results_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        eval_logger.log_evaluation_complete(eval_results, success=True)
        
    else:
        print("❌ 没有有效的预测结果")
        eval_logger.log_evaluation_complete({}, success=False, error_msg="没有有效的预测结果")
        return
    
    print(f"\n✅ 评估完成!")
    print(f"📋 详细评估日志: {LOGBOOK_FILE}")

if __name__ == "__main__":
    main()