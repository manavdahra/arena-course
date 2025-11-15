from jaxtyping import Float, Int
import torch
from torch import Tensor
import torch.nn as nn
from config import Config
import einops
import math

class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head), device=cfg.device))
        self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head), device=cfg.device))
        self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head), device=cfg.device))
        self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model), device=cfg.device))

        self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head), device=cfg.device))
        self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head), device=cfg.device))
        self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head), device=cfg.device))
        self.b_O = nn.Parameter(torch.zeros((cfg.d_model), device=cfg.device))

        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", torch.tensor(float("-inf"), dtype=torch.float32, device=cfg.device))

    
    def forward(self, res: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        """Project from embedding space to attention space
        """
        Q = einops.einsum(res, self.W_Q, "batch posn d_model,n_heads d_model d_head -> batch posn n_heads d_head") + self.b_Q
        K = einops.einsum(res, self.W_K, "batch posn d_model,n_heads d_model d_head -> batch posn n_heads d_head") + self.b_K
        V = einops.einsum(res, self.W_V, "batch posn d_model,n_heads d_model d_head -> batch posn n_heads d_head") + self.b_V

        """A = softmax(Q^T.K/sqrt(d_head))
        """
        attn_score = einops.einsum(Q, K, "batch query_pos n_heads d_head,batch key_pos n_heads d_head -> batch n_heads query_pos key_pos")
        attn_score = attn_score / math.sqrt(self.cfg.d_head) # normalize to avoid vanishing gradients
        attn_score = self.apply_causal_mask(attn_score)
        attn_score = attn_score.softmax(dim=-1) # softmax

        """Z = A*V
        """
        z = einops.einsum(attn_score, V, "batch n_heads query_pos key_pos,batch key_pos n_heads d_head -> batch n_heads query_pos d_head") 

        """Project from attention space to embedding space
        """
        return einops.einsum(z, self.W_O, "batch n_heads query_pos d_head,n_heads d_head d_model -> batch query_pos d_model") + self.b_O
    
    def apply_causal_mask(self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        query_pos, key_pos = attn_scores.shape[-2:]
        mask = torch.tril(torch.ones((query_pos, key_pos), device=self.cfg.device)).bool()
        return torch.where(
            mask,
            attn_scores,
            self.IGNORE,
        )
