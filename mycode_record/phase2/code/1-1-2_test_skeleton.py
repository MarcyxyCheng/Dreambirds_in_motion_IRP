# """
# 可视化 APT-17 骨架（与 APT-36K / AP-10K 同结构）
# —— 关节编号 + 分部位配色
# """

# import yaml
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# # -------------------------------------------------
# # 1. 读取 skeleton.yaml
# # -------------------------------------------------
# with open("skeleton.yaml") as f:
#     skel_cfg = yaml.safe_load(f)

# joints   = skel_cfg["joints"]          # list [id, name, color, parent, swap]
# edges    = skel_cfg["skeleton"]        # list [i, j]
# N        = skel_cfg["num_joints"]

# # -------------------------------------------------
# # 2. 预设一组示例 2D 坐标（只是为了把线画整齐）
# #    可自行调整以符合观感
# # -------------------------------------------------
# pose = np.zeros((N, 2))
# pose[:] = np.nan   # 初始化为空，方便调试

# # 头 & 躯干
# pose[0]  = (0.50, 0.90)   # nose
# pose[1]  = (0.46, 0.94)   # L eye
# pose[2]  = (0.54, 0.94)   # R eye
# pose[3]  = (0.50, 0.80)   # neck
# pose[4]  = (0.60, 0.70)   # tail

# # 前肢（左 = 图左侧）
# pose[5]  = (0.40, 0.76)   # L shoulder
# pose[6]  = (0.60, 0.76)   # R shoulder
# pose[7]  = (0.35, 0.66)   # L elbow
# pose[8]  = (0.65, 0.66)   # R elbow
# pose[9]  = (0.30, 0.56)   # L front paw
# pose[10] = (0.70, 0.56)   # R front paw

# # 后肢
# pose[11] = (0.45, 0.72)   # L hip
# pose[12] = (0.55, 0.72)   # R hip
# pose[13] = (0.40, 0.62)   # L knee
# pose[14] = (0.60, 0.62)   # R knee
# pose[15] = (0.35, 0.48)   # L back paw
# pose[16] = (0.65, 0.48)   # R back paw

# # -------------------------------------------------
# # 3. 身体分区 & 颜色
# # -------------------------------------------------
# part_colors = {
#     "head":            "#e41a1c",
#     "neck":            "#ff77ff",
#     "torso":           "#4daf4a",
#     "front_left_leg":  "#377eb8",
#     "front_right_leg": "#1f78b4",
#     "back_left_leg":   "#ff7f00",
#     "back_right_leg":  "#ffbb78",
#     "tail":            "#984ea3",
# }

# # 方便判断：把关键点 id 做成集合
# parts = dict(
#     head  = {0,1,2},
#     neck  = {3},
#     tail  = {4},
#     front_left_leg  = {5,7,9},
#     front_right_leg = {6,8,10},
#     back_left_leg   = {11,13,15},
#     back_right_leg  = {12,14,16},
# )
# # 反向索引：每个关节 → 所属部位名
# joint2part = {idx: pname for pname, ids in parts.items() for idx in ids}

# def edge_part(i: int, j: int) -> str:
#     """根据两个端点的分区，决定边的颜色。
#        - 若同属一个分区 → 返回该分区
#        - 否则默认分到 torso
#     """
#     pi, pj = joint2part.get(i), joint2part.get(j)
#     if pi == pj and pi is not None:
#         return pi
#     return "torso"          # 如 0-3, 3-11 之类跨区连线

# # -------------------------------------------------
# # 4. 画图
# # -------------------------------------------------
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.set_title("Bird / Mammal 17-KPT skeleton (APT schema)")
# ax.set_xlim(0, 1); ax.set_ylim(0, 1)
# ax.set_aspect("equal"); ax.invert_yaxis()     # 让 (0,0) 在左上

# # (a) 画骨骼
# for i, j in edges:
#     p = edge_part(i, j)
#     ax.add_line(Line2D(
#         [pose[i, 0], pose[j, 0]],
#         [pose[i, 1], pose[j, 1]],
#         lw=3, color=part_colors[p], alpha=0.9)
#     )

# # (b) 画关节点编号
# for idx, (x, y) in enumerate(pose):
#     if np.isnan(x):                 # 仅防守性检查
#         continue
#     ax.scatter(x, y, s=40, color="black", zorder=3)
#     ax.text(x, y, str(idx),
#             color="white", fontsize=8,
#             ha="center", va="center", zorder=4)

# # (c) 自定义图例
# handles = [Line2D([0], [0], lw=4, color=c)
#            for c in part_colors.values()]
# labels = list(part_colors.keys())
# ax.legend(handles, labels, loc="lower left", fontsize=8, framealpha=0.8)

# plt.tight_layout()
# plt.savefig("skeleton_pretty.png", dpi=150)
# print("[OK] skeleton_pretty.png generated — 请肉眼核对拓扑与分区配色")


"""
修正版 skeleton 可视化 —— 适配官方 APT-17 顺序
"""

import yaml, numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------- 1. 读取骨架 ----------
with open("skeleton.yaml") as f:
    cfg = yaml.safe_load(f)
N = cfg["num_joints"]
edges = cfg["skeleton"]

# ---------- 2. 示例坐标 ----------
pose = np.array([
    # 0-16 按官方顺序
    [0.46,0.94],  # 0 left_eye
    [0.54,0.94],  # 1 right_eye
    [0.50,0.90],  # 2 nose
    [0.50,0.80],  # 3 neck
    [0.60,0.70],  # 4 tail

    [0.40,0.76],  # 5 L shoulder
    [0.35,0.66],  # 6 L elbow
    [0.30,0.56],  # 7 L front paw

    [0.60,0.76],  # 8 R shoulder
    [0.65,0.66],  # 9 R elbow
    [0.70,0.56],  #10 R front paw

    [0.45,0.74],  #11 L hip
    [0.40,0.64],  #12 L knee
    [0.35,0.50],  #13 L back paw

    [0.55,0.74],  #14 R hip
    [0.60,0.64],  #15 R knee
    [0.65,0.50],  #16 R back paw
])

# ---------- 3. 分区 & 颜色 ----------
part_colors = {
    "head": "#e41a1c",
    "neck": "#ff77ff",
    "torso": "#4daf4a",
    "front_left_leg":  "#377eb8",
    "front_right_leg": "#1f78b4",
    "back_left_leg":   "#ff7f00",
    "back_right_leg":  "#ffbb78",
    "tail": "#984ea3",
}

parts = dict(
    head            ={0,1,2},
    neck            ={3},
    torso           ={3,4},          # 把颈根与尾根也视为 torso 连线可选
    front_left_leg  ={5,6,7},
    front_right_leg ={8,9,10},
    back_left_leg   ={11,12,13},
    back_right_leg  ={14,15,16},
    tail            ={4},
)
joint2part = {j: p for p, ids in parts.items() for j in ids}

def edge_part(i: int, j: int) -> str:
    """改进版——如果一端属于 limb 而另一端 torso/neck，也算 limb"""
    pi, pj = joint2part.get(i), joint2part.get(j)
    if pi == pj:
        return pi or "torso"
    # limb ↔ neck/torso 的边归 limb
    limb_set = {"front_left_leg","front_right_leg","back_left_leg","back_right_leg"}
    if pi in limb_set and pj in {"neck","torso"}:  return pi
    if pj in limb_set and pi in {"neck","torso"}:  return pj
    # 其他跨 limb 或未知 → torso
    return "torso"

# ---------- 4. 绘制 ----------
fig, ax = plt.subplots(figsize=(6,6))
ax.set_title("Bird / Mammal 17-KPT skeleton (APT schema)")
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_aspect("equal"); ax.invert_yaxis()

# 画骨骼
for i,j in edges:
    p = edge_part(i,j)
    ax.add_line(Line2D([pose[i,0],pose[j,0]],
                       [pose[i,1],pose[j,1]],
                       lw=3,color=part_colors[p],alpha=0.9))

# 画关节点
for idx,(x,y) in enumerate(pose):
    ax.scatter(x,y,s=40,color="black",zorder=3)
    ax.text(x,y,str(idx),color="white",fontsize=8,
            ha="center",va="center",zorder=4)

# 图例
handles = [Line2D([0],[0],lw=4,color=c) for c in part_colors.values()]
ax.legend(handles, list(part_colors.keys()),
          loc="lower left", fontsize=8, framealpha=0.8)

plt.tight_layout()
plt.savefig("skeleton_pretty.png", dpi=150)
print("[OK] skeleton_pretty.png updated — 颜色已按肢体分区")
