import os
from PIL import Image

# 1. 配置路径
input_dir  = r"C:\Users\lizhi\Downloads\liif-main\liif\data\single\valid"
output_dir = r"C:\Users\lizhi\Downloads\liif-main\liif\outputs"
os.makedirs(output_dir, exist_ok=True)

# 2. 目标分辨率
target_w, target_h = 2808, 1872

# 3. 三种插值方法
methods = {
    "nearest":  Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic":  Image.BICUBIC,
}

# 4. 遍历目录
for fname in os.listdir(input_dir):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        continue
    src_path = os.path.join(input_dir, fname)
    img = Image.open(src_path)

    base, ext = os.path.splitext(fname)
    for name, interp in methods.items():
        # 放大
        resized = img.resize((target_w, target_h), interp)
        # 保存，文件名加分辨率和方法后缀
        out_name = f"{base}_{target_w}x{target_h}_{name}{ext}"
        resized.save(os.path.join(output_dir, out_name))

print("Done!")
