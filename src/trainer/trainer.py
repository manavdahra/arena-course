from dataclasses import dataclass
import torch
import datasets

from gpt.transformer import DemoTransformer
from gpt.config import Config
from transformer_lens import HookedTransformer

"""Set GPU config
"""
device = torch.device(
    "mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu"
)
"""Init reference GPT2
"""
reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,
    device=device,
)

"""Create model
"""
model_cfg = Config(
    debug=False,
    d_model=32,
    n_heads=16,
    d_head=2,
    d_mlp=32 * 4,
    n_layers=4,
    n_ctx=128,
    d_vocab=reference_gpt2.cfg.d_vocab,
)
model = DemoTransformer(model_cfg)

"""Trainer config
"""
@dataclass
class TransformerTrainingArgs:
    batch_size: int = 32
    epochs: int = 10
    max_steps_per_epoch: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-2
    # wandb_project: str | None = "day1-demotxn"
    # wandb_name: str | None = None

"""Load Tinystories dataset
"""
dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
print(dataset)
print(dataset[0]["text"])
