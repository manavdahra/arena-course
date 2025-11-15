from jaxtyping import Float
from torch import Tensor
import torch.nn as nn
from config import Config
from attention import Attention
from layer_norm import LayerNorm
from mlp import MLP

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)
    
    def forward(self, res: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        res_attn = res + self.attn(self.ln1(res))
        res_mlp = res_attn + self.mlp(self.ln2(res_attn))
        return res_mlp
