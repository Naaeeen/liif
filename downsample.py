"""
Batch Downsampling Script

This script downsamples a single high-resolution input image into multiple
lower-resolution versions using fixed integer scaling factors. It uses OpenCV's
`INTER_AREA` interpolation method, which is recommended for downsampling tasks.

Each downsampled image is saved in the same directory with a filename indicating
its new resolution (e.g., 936x624.jpg).

Typical use case: generating low-resolution test inputs for super-resolution models.

"""

import cv2
import os

scales = [2, 4, 6, 8, 12, 16, 18]
input_dir  = 'data/single'
input_name = 'input.jpg'
output_dir = input_dir

input_path = os.path.join(input_dir, input_name)
img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"Cannot read {input_path}")
h, w = img.shape[:2]

os.makedirs(output_dir, exist_ok=True)
for s in scales:
    new_w = w // s
    new_h = h // s
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out_name = f"{new_w}x{new_h}.jpg"
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, small)
    print(f"Downsampled 1/{s}: {w}×{h} → {new_w}×{new_h}, saved as {out_name}")
