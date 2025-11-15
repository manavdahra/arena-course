from jaxtyping import Float, Int
import torch
from torch import Tensor
import torch.nn as nn
from config import Config
import einops
from transformer_lens.utils import gelu_new

class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp), device=cfg.device))
        self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp), device=cfg.device))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model), device=cfg.device))
        self.b_out = nn.Parameter(torch.zeros((cfg.d_model), device=cfg.device))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
    
    def forward(self, res: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        pre = einops.einsum(res, self.W_in, "batch posn d_model,d_model d_mlp -> batch posn d_mlp") + self.b_in
        mid = gelu_new(pre)
        post = einops.einsum(mid, self.W_out, "batch posn d_mlp,d_mlp d_model -> batch posn d_model") + self.b_out

        return post
