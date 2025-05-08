import math
import torch
import torch.nn as nn


class FourierMapping(nn.Module):
    """
    把 2-D 坐标 (x,y) → 2*K 维正余弦特征：
        γ(x,y) = [sin(2πf1x),cos(2πf1x), … , sin(2πfK y),cos(2πfK y)]
    """
    def __init__(self, num_bands=5, base_freq=1.0):
        super().__init__()
        # 频率序列：base, 2*base, 4*base, ...
        freqs = base_freq * (2.0 ** torch.arange(num_bands))  # (K,)
        self.register_buffer('freqs', freqs)  # 不参与梯度更新

    def forward(self, coord):
        # coord 形状 (B, N, 2) ，值域已在 [-1,1]
        out = []
        for f in self.freqs:
            out.append(torch.sin(coord * f * math.pi))  # *2π 亦可
            out.append(torch.cos(coord * f * math.pi))
        return torch.cat(out, dim=-1)  # (B, N, 2*K)
