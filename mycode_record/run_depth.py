# run_depth.py  ── write 8-bit 512×512 PNG depth map
from PIL import Image
import numpy as np
import torchvision.transforms as T
from transformers import pipeline

depth_pipe = pipeline(
    "depth-estimation",
    model="LiheYoung/depth-anything-large-hf",
    device="cpu"                       # GPU 不必要
)

out = depth_pipe("mybird.jpg")
# 最新 transformers 返回字典键名有时是 'predicted_depth'，有时是 'depth'
depth_img = out.get("depth") or out.get("predicted_depth")  # PIL.Image (mode=F)

# 转 numpy → 归一化 → uint8
d_np = np.array(depth_img, dtype="float32")
d_np = 255 * (d_np / d_np.max())          # 0-255
d_u8 = d_np.astype("uint8")

# 保存 512×512 灰度 PNG
png = Image.fromarray(d_u8, mode="L").resize((512, 512), Image.BILINEAR)
png.save("depth_map.png")
print("✅ depth_map.png saved (512×512)")
