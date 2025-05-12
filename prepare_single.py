import numpy as np
import cv2
import os, argparse

# Argument parser to specify input image and output directory
parser = argparse.ArgumentParser()
parser.add_argument('--img',  type=str, default='input.jpg',
                    help='path to your single image')
parser.add_argument('--out_dir', type=str, default='data',
                    help='folder to save .npy files')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.out_dir, exist_ok=True)

# Read the input image and normalize it to [0, 1] in RGB format
bgr = cv2.imread(args.img, cv2.IMREAD_COLOR)
rgb = bgr[..., ::-1] / 255.0
np.save(f'{args.out_dir}/img.npy', rgb)
# Generate normalized coordinate grid for the image
H, W = rgb.shape[:2]
ys = np.linspace(0, 1, H, dtype=np.float32)
xs = np.linspace(0, 1, W, dtype=np.float32)
grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')
coord = np.stack([grid_x, grid_y], axis=-1)
np.save(f'{args.out_dir}/coord.npy', coord)

print('Saved:', f'{args.out_dir}/img.npy', f'{args.out_dir}/coord.npy')
