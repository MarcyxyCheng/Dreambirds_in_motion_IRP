# """
# 写入 APT-36K 风格的 17 点鸟类骨架到 skeleton.yaml
# """
# import yaml
# import copy 

# OUT = "skeleton.yaml"

# SKELETON = dict(
#     num_joints=17,
#     joints=[
#         # id  name                color(r,g,b)  parent  swap (左右翻转用)
#         (0,  "beak_tip",          (255,0,0),    -1,     -1),
#         (1,  "head_top",          (255,0,0),    0,      -1),
#         (2,  "left_eye",          (255,0,0),    1,      3),
#         (3,  "right_eye",         (255,0,0),    1,      2),
#         (4,  "neck_base",         (255,85,0),   1,      -1),
#         (5,  "left_shoulder",     (255,170,0),  4,      6),
#         (6,  "right_shoulder",    (255,170,0),  4,      5),
#         (7,  "left_wing_elbow",   (255,255,0),  5,      8),
#         (8,  "right_wing_elbow",  (255,255,0),  6,      7),
#         (9,  "left_wing_tip",     (170,255,0),  7,      10),
#         (10, "right_wing_tip",    (170,255,0),  8,      9),
#         (11, "spine_mid",         (0,255,0),    4,      -1),
#         (12, "left_hip",          (0,255,85),   11,     13),
#         (13, "right_hip",         (0,255,85),   11,     12),
#         (14, "left_knee",         (0,255,170),  12,     15),
#         (15, "right_knee",        (0,255,170),  13,     14),
#         (16, "tail_tip",          (0,255,255),  11,     -1),
#     ],
#     skeleton=[
#         (0,1),(1,2),(1,3),(1,4),
#         (4,5),(4,6), (5,7),(7,9), (6,8),(8,10),
#         (4,11),(11,12),(11,13), (12,14),(13,15),
#         (11,16)
#     ],
#     flip_pairs=[(2,3),(5,6),(7,8),(9,10),(12,13),(14,15)]
# )

# # with open(OUT, "w") as f:
# #     yaml.dump(SKELETON,f,sort_keys=False)

# # print(f"[OK] wrote {OUT}")

# #  把 tuple 转成 list，确保是纯 JSON 结构
# safe_skel = copy.deepcopy(SKELETON)
# safe_skel["joints"]     = [list(j) for j in safe_skel["joints"]]
# safe_skel["skeleton"]   = [list(p) for p in safe_skel["skeleton"]]
# safe_skel["flip_pairs"] = [list(p) for p in safe_skel["flip_pairs"]]

# with open(OUT, "w") as f:
#     yaml.safe_dump(safe_skel, f, sort_keys=False)   # 用 safe_dump
# print(f"[OK] wrote {OUT} (tuple → list)")



### ------------
#第二版

# import yaml, copy

# OUT = "skeleton.yaml"

# SKELETON = dict(
#     num_joints=17,
#     joints=[
#         # id  name               rgb          parent  swap
#         (0,  "nose",             (255,  0,  0), -1,  -1),
#         (1,  "left_eye",         (255, 85,  0), 0,   2),
#         (2,  "right_eye",        (255, 85,  0), 0,   1),
#         (3,  "neck",             (255,170,  0), 0,  -1),
#         (4,  "tail",             (255,255,  0), 3,  -1),

#         (5,  "left_shoulder",    (  0,255,  0), 3,   6),
#         (6,  "right_shoulder",   (  0,255,  0), 3,   5),
#         (7,  "left_elbow",       (  0,170,255), 5,   8),
#         (8,  "right_elbow",      (  0,170,255), 6,   7),
#         (9,  "left_front_paw",   (  0, 85,255), 7,  10),
#         (10, "right_front_paw",  (  0, 85,255), 8,   9),

#         (11, "left_hip",         (170,  0,255), 3,  12),
#         (12, "right_hip",        (170,  0,255), 3,  11),
#         (13, "left_knee",        (170, 85,127),11,  14),
#         (14, "right_knee",       (170, 85,127),12,  13),
#         (15, "left_back_paw",    (170,170,127),13,  16),
#         (16, "right_back_paw",   (170,170,127),14,  15),
#     ],

#     # 连线关系（APT-36K 论文原图的骨骼拓扑）
#     skeleton=[
#         (0,1),(0,2),(0,3),
#         (3,5),(3,6),(5,7),(7,9),(6,8),(8,10),
#         (3,11),(3,12),(11,13),(13,15),(12,14),(14,16),
#         (3,4)
#     ],

#     flip_pairs=[(1,2),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
# )

# # -------- 写入 YAML (tuple → list) --------
# safe = copy.deepcopy(SKELETON)
# safe["joints"]     = [list(j) for j in safe["joints"]]
# safe["skeleton"]   = [list(p) for p in safe["skeleton"]]
# safe["flip_pairs"] = [list(p) for p in safe["flip_pairs"]]

# with open(OUT, "w") as f:
#     yaml.safe_dump(safe, f, sort_keys=False)

# print(f"[OK] wrote {OUT}  (APT-17 schema)")


###----------------------
#第三版，更加完全和原本一样

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_define_skeleton.py
---------------------
把 APT-36K / AP-10K 官方 17-点骨架写入 skeleton.yaml
• 完全遵循官方关节顺序与语义
• 先用 tuple 描述，再写文件时转换成 list，确保 YAML 无自定义标签
"""

import yaml
import copy
from pathlib import Path

OUT = Path("skeleton.yaml")

# ----------------------------------------------------------------------
# 1. 关节 / 骨骼 / 镜像配对 —— “官方 17-KPT 顺序”
# id, name,             rgb,          parent, swap
SKELETON = dict(
    num_joints=17,
    joints=[
        (0,  "left_eye",          (255,  85,   0), -1,   1),
        (1,  "right_eye",         (255,  85,   0), -1,   0),
        (2,  "nose",              (255,   0,   0), -1,  -1),
        (3,  "neck",              (255, 170,   0),  2,  -1),
        (4,  "tail",              (255, 255,   0),  3,  -1),

        (5,  "left_shoulder",     (  0, 255,   0),  3,   8),
        (6,  "left_elbow",        (  0, 170, 255),  5,   9),
        (7,  "left_front_paw",    (  0,  85, 255),  6,  10),

        (8,  "right_shoulder",    (  0, 255,   0),  3,   5),
        (9,  "right_elbow",       (  0, 170, 255),  8,   6),
        (10, "right_front_paw",   (  0,  85, 255),  9,   7),

        (11, "left_hip",          (170,   0, 255),  3,  14),
        (12, "left_knee",         (170,  85, 127), 11,  15),
        (13, "left_back_paw",     (170, 170, 127), 12,  16),

        (14, "right_hip",         (170,   0, 255),  3,  11),
        (15, "right_knee",        (170,  85, 127), 14,  12),
        (16, "right_back_paw",    (170, 170, 127), 15,  13),
    ],

    # skeleton: 按官方示意图连线
    skeleton=[
        (0, 2), (1, 2), (2, 3),            # 头部
        (3, 5), (5, 6), (6, 7),            # 左前肢
        (3, 8), (8, 9), (9,10),            # 右前肢
        (3,11), (11,12), (12,13),          # 左后肢
        (3,14), (14,15), (15,16),          # 右后肢
        (3, 4)                              # 尾巴
    ],

    flip_pairs=[(0,1),(5,8),(6,9),(7,10),(11,14),(12,15),(13,16)]
)

# ----------------------------------------------------------------------
# 2. tuple → list  (SafeLoader 兼容)，写入 YAML
safe = copy.deepcopy(SKELETON)
for key in ("joints", "skeleton", "flip_pairs"):
    safe[key] = [list(item) for item in safe[key]]

with OUT.open("w") as f:
    yaml.safe_dump(safe, f, sort_keys=False)

print(f"[OK] wrote {OUT.resolve()}  (APT-17 schema)")