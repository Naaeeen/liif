import math
import torch
import torch.nn as nn


class FourierMapping(nn.Module):
    """
    Fourier Feature Mapping for 2D Coordinates

    Given 2D coordinates (x, y), this module encodes them into a higher-dimensional
    space using a bank of sinusoidal functions at exponentially increasing frequencies.

    The output embedding is:
        γ(x, y) = [
            sin(π f₁ x), cos(π f₁ x), ..., sin(π f_K x), cos(π f_K x),
            sin(π f₁ y), cos(π f₁ y), ..., sin(π f_K y), cos(π f_K y)
        ]

    This is used to mitigate the spectral bias of coordinate-based MLPs
    and helps the network better learn high-frequency details.
    """
    def __init__(self, num_bands=5, base_freq=1.0):
        super().__init__()
        freqs = base_freq * (2.0 ** torch.arange(num_bands))
        self.register_buffer('freqs', freqs)# Register as non-trainable buffer

    def forward(self, coord):
        out = []
        for f in self.freqs:
            out.append(torch.sin(coord * f * math.pi))
            out.append(torch.cos(coord * f * math.pi))
        return torch.cat(out, dim=-1)
