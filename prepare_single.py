import numpy as np
import cv2
import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img',  type=str, default='input.jpg',
                    help='path to your single image')
parser.add_argument('--out_dir', type=str, default='data',
                    help='folder to save .npy files')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# 1️⃣ 读取图像并转成 RGB、归一化
bgr = cv2.imread(args.img, cv2.IMREAD_COLOR)                 # BGR numpy array  :contentReference[oaicite:2]{index=2}
rgb = bgr[..., ::-1] / 255.0                                 # -> H×W×3, float32
np.save(f'{args.out_dir}/img.npy', rgb)                      # 保存 img.npy  :contentReference[oaicite:3]{index=3}

# 2️⃣ 生成归一化坐标网格
H, W = rgb.shape[:2]
ys = np.linspace(0, 1, H, dtype=np.float32)
xs = np.linspace(0, 1, W, dtype=np.float32)
grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')          #  :contentReference[oaicite:4]{index=4}
coord = np.stack([grid_x, grid_y], axis=-1)                  # H×W×2
np.save(f'{args.out_dir}/coord.npy', coord)                  # 保存 coord.npy

print('Saved:', f'{args.out_dir}/img.npy', f'{args.out_dir}/coord.npy')
