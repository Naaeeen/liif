import os
from PIL import Image

"""
Image Resizing Script with Multiple Interpolation Methods

This script reads all image files from a specified input directory and resizes them
to a given target resolution using three different interpolation methods:
nearest neighbor, bilinear, and bicubic.

The resized images are saved to the output directory, with filenames that include
the target resolution and the interpolation method used.

Intended for use in benchmarking or visual comparison of interpolation techniques
(e.g., as baselines for super-resolution tasks).

"""

# 1. Configure input and output directories
input_dir  = r"C:\Users\lizhi\Downloads\liif-main\liif\data\single\test"
output_dir = r"C:\Users\lizhi\Downloads\liif-main\liif\outputs"
os.makedirs(output_dir, exist_ok=True)

# 2. Target resolution (width, height)
target_w, target_h = 1404, 936

# 3. Define three interpolation methods
methods = {
    "nearest":  Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic":  Image.BICUBIC,
}

for fname in os.listdir(input_dir):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        continue
    src_path = os.path.join(input_dir, fname)
    img = Image.open(src_path)

    base, ext = os.path.splitext(fname)
    for name, interp in methods.items():
        resized = img.resize((target_w, target_h), interp)
        out_name = f"{base}_{target_w}x{target_h}_{name}{ext}"
        resized.save(os.path.join(output_dir, out_name))

print("Done!")
