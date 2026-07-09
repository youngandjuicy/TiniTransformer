import os

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import TextDataset
from engine.evaluate import evaluate
from engine.train import train_one_epoch
from models.transformer import TinyTransformer
from tokenizer import CharTokenizer
from utils.seed import set_seed


def build_model(cfg):
    return TinyTransformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        num_layers=cfg.num_layers,
        block_size=cfg.block_size,
    ).to(cfg.device)


def save_checkpoint(path, model, optimizer, cfg, tokenizer, epoch, train_loss, val_loss):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": cfg.__dict__.copy(),
            "tokenizer": {
                "stoi": tokenizer.stoi,
            },
        },
        path,
    )


def main():
    cfg = Config()
    set_seed(cfg.seed)

    with open(cfg.data_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    tokens = tokenizer.encode(text)
    cfg.vocab_size = tokenizer.vocab_size

    split = int(0.9 * len(tokens))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    train_dataset = TextDataset(train_tokens, cfg.block_size)
    val_dataset = TextDataset(val_tokens, cfg.block_size)
    pin_memory = cfg.device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    print(
        f"Training on {cfg.device} | "
        f"tokens={len(tokens)} | vocab={cfg.vocab_size} | "
        f"train_batches={len(train_loader)} | val_batches={len(val_loader)}"
    )

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg)
        val_loss = evaluate(model, val_loader, criterion, cfg)

        print(
            f"Epoch [{epoch + 1}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                os.path.join(cfg.checkpoint_dir, "best.pth"),
                model,
                optimizer,
                cfg,
                tokenizer,
                epoch + 1,
                train_loss,
                val_loss,
            )

    save_checkpoint(
        os.path.join(cfg.checkpoint_dir, "final.pth"),
        model,
        optimizer,
        cfg,
        tokenizer,
        cfg.epochs,
        train_loss,
        val_loss,
    )


if __name__ == "__main__":
    main()
