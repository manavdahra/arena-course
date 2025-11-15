from dataclasses import dataclass
import torch
from torch import nn, Tensor
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new
from config import Config
from embedding import Embed, PosEmbed, Unembed
from layer_norm import LayerNorm
from block import TransformerBlock

class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)
    
    def forward(self, tokens: Int[Tensor, "batch posn"]) -> Float[Tensor, "batch posn d_vocab"]:
        res = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            res = block(res)
        
        return self.unembed(self.ln_final(res))

