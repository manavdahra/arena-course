import torch
from transformer_lens import HookedTransformer
from config import Config
from transformer import DemoTransformer
from loss import get_log_probs
import math

def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )

    cfg = Config(device=device, debug=True)
    reference_gpt2 = HookedTransformer.from_pretrained(
        "gpt2-small",
        fold_ln=False,
        center_unembed=False,
        center_writing_weights=False,
        device=device,
    )

    demo_gpt2 = DemoTransformer(cfg)
    demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)

    text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
    tokens = reference_gpt2.to_tokens(text)
    demo_logits = demo_gpt2(tokens)

    pred_log_probs = get_log_probs(demo_logits, tokens)
    print(f"Avg cross entropy loss: {-pred_log_probs.mean():.4f}")
    print(f"Avg cross entropy loss for uniform distribution: {math.log(demo_gpt2.cfg.d_vocab):4f}")
    print(f"Avg probability assigned to correct token: {pred_log_probs.exp().mean():4f}")

    prompt = """Mitigating the risk of extinction from AI should be a global priority alongside other societal-scale risks such as"""
    print(f"Prompt:\n{prompt}")
    for _ in range(100):
        test_tokens = reference_gpt2.to_tokens(prompt)
        demo_logits = demo_gpt2(test_tokens)
        prompt += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())

    print(f"Completion:\n{prompt}")

if __name__ == "__main__":
    main()
