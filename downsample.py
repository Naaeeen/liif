# batch_downsample.py
import cv2
import os

# 1. 配置
scales = [2, 4, 6, 8, 12, 16, 18]
input_dir  = 'data/single'
input_name = 'input.jpg'      # 原图文件名
output_dir = input_dir        # 输出到同一目录

# 2. 读取原图
input_path = os.path.join(input_dir, input_name)
img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"Cannot read {input_path}")
h, w = img.shape[:2]

# 3. 循环下采样并保存
os.makedirs(output_dir, exist_ok=True)
for s in scales:
    new_w = w // s
    new_h = h // s
    # INTER_AREA：下采样首选
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out_name = f"{new_w}x{new_h}.jpg"
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, small)
    print(f"Downsampled 1/{s}: {w}×{h} → {new_w}×{new_h}, saved as {out_name}")
