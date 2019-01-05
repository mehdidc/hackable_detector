import torch.nn as nn
import torch
from torch.nn.functional import normalize


class L2Norm(nn.Module):
    """L2Norm layer across all channels."""

    def __init__(self, in_features, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters(scale)

    def reset_parameters(self, scale):
        nn.init.constant(self.weight, scale)

    def forward(self, x):
        x = normalize(x, dim=1)
        scale = self.weight[None, :, None, None]
        return scale * x
