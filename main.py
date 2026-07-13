import os
import json

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.amp import GradScaler

from config import Config
from dataset import TextDataset
from engine.evaluate import evaluate
from engine.train import train_one_epoch
from models.transformer import TinyTransformer
from tokenizer import CharTokenizer
from utils.plot import plot_loss_curve
from utils.seed import set_seed


def build_model(cfg):

    Model = TinyTransformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        num_layers=cfg.num_layers,
        block_size=cfg.block_size,
    ).to(cfg.device)

    if cfg.device == "cuda":
        if hasattr(torch, "compile"):
            Model = torch.compile(Model)

    return Model

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
    scaler = GradScaler("cuda")

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
    train_losses = []
    val_losses = []
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(cfg.tensorboard_log_dir, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("experiment", "Training TinyTransformer on text data")
    writer.add_text("config", json.dumps(cfg.__dict__, indent=4))
    writer.add_text("model", str(model))

    print(
        f"Training on {cfg.device} | "
        f"tokens={len(tokens)} | vocab={cfg.vocab_size} | "
        f"train_batches={len(train_loader)} | val_batches={len(val_loader)}"
    )

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg, scaler)
        val_loss = evaluate(model, val_loader, criterion, cfg)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        writer.add_scalar("loss/train", train_loss, epoch + 1)
        writer.add_scalar("loss/val", val_loss, epoch + 1)

        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("learning_rate", lr, epoch + 1)

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
    writer.close()
    plot_loss_curve(train_losses, val_losses, cfg.loss_plot_path)
    print(f"Saved loss curve to {cfg.loss_plot_path}")


if __name__ == "__main__":
    main()
