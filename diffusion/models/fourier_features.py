import torch
import torch.nn as nn

class FourierFeatures(nn.Module):
    def __init__(self, in_features: int, n_min: int = 7, n_max: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = in_features + in_features * 2 * (n_max - n_min + 1)
        self.register_buffer('weight', torch.pi *  2**torch.arange(n_min, n_max+1))

    def forward(self, x: torch.Tensor):
        f = (x[:, :, None] * self.weight[:, None, None]).flatten(1, 2)
        out = torch.cat([x, f.cos(), f.sin()], dim=1)
        return out