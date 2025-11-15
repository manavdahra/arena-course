from jaxtyping import Float, Int
import torch
from torch import Tensor
import torch.nn as nn
from config import Config

class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model, device=cfg.device))
        self.b = nn.Parameter(torch.zeros(cfg.d_model, device=cfg.device))

    def forward(self, res: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        mean = res.mean(dim=-1, keepdim=True)
        std = torch.sqrt(res.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps)

        return ((res - mean) / std) * self.w + self.b
