import argparse
import math
from PIL import Image
import torch
from torchvision import transforms as T

def load_image(path):
    """Load an image file and convert to a FloatTensor in [0,1] with shape (C,H,W)."""
    img = Image.open(path).convert('RGB')
    tensor = T.ToTensor()(img)
    return tensor

def compute_l1(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute L1 loss (mean absolute error) between two image tensors."""
    return torch.mean(torch.abs(img1 - img2)).item()

def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute PSNR (Peak Signal-to-Noise Ratio) in dB between two image tensors."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))

def main():
    parser = argparse.ArgumentParser(description='Compute L1 loss and PSNR between two images')
    parser.add_argument('img1', type=str, help='Path to the first image')
    parser.add_argument('img2', type=str, help='Path to the second image')
    args = parser.parse_args()

    img1 = load_image(args.img1)
    img2 = load_image(args.img2)

    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes differ: {img1.shape} vs {img2.shape}")

    l1 = compute_l1(img1, img2)
    psnr = compute_psnr(img1, img2)

    print(f"L1 loss: {l1:.6f}")
    print(f"PSNR: {psnr:.2f} dB")

if __name__ == '__main__':
    main()
