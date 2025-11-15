import torch
from torch import Tensor
import torch.nn as nn
from config import Config
from jaxtyping import Float, Int
import einops

class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty((cfg.d_vocab, cfg.d_model), device=cfg.device))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)
    
    def forward(self, tokens: Float[Tensor, "batch posn"]) -> Float[Tensor, "batch posn d_model"]:
        return self.W_E[tokens]

class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model), device=cfg.device))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)
    
    def forward(self, tokens: Int[Tensor, "batch posn"]) -> Float[Tensor, "batch posn d_model"]:
        """Lookup in W_pos (n_ctx x d_model) 
        to get an output tensor of shape (batch posn d_model)
        
        This implies we need to build index tensor (idx) of shape (batch x posn)
        doing self.W_pos[idx] yields output shape (batch x posn x d_model)
        """
        batch, posn = tokens.shape
        idx = torch.tile(torch.arange(0, posn), (batch, 1)).to(self.cfg.device)
        return self.W_pos[idx]

class Unembed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab), device=cfg.device))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(torch.zeros((cfg.d_vocab), device=cfg.device, requires_grad=False))

    def forward(self, res: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_vocab"]:
        return einops.einsum(
            res, 
            self.W_U, 
            "batch posn d_model,d_model d_vocab -> batch posn d_vocab",
        ) + self.b_U
