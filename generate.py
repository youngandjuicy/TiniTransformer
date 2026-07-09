import argparse

import torch
import torch.nn.functional as F

from config import Config
from models.transformer import TinyTransformer
from tokenizer import CharTokenizer


@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temperature=1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

    return idx


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    cfg = Config()
    for key, value in checkpoint["config"].items():
        setattr(cfg, key, value)
    cfg.device = device

    tokenizer = CharTokenizer(stoi=checkpoint["tokenizer"]["stoi"])
    model = TinyTransformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        num_layers=cfg.num_layers,
        block_size=cfg.block_size,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, tokenizer, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--prompt", default="To be")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, cfg = load_model(args.checkpoint, device)
    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    generated = generate(
        model,
        idx,
        args.max_new_tokens,
        cfg.block_size,
        args.temperature,
    )
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
