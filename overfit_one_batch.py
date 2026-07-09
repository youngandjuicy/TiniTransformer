import json
import random
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from config import Config
from tokenizer import CharTokenizer
from dataset import TextDataset
from models.transformer import TinyTransformer
from engine.train import train_one_epoch
from utils.plot import plot_loss_curve


# -----------------------
# 固定随机种子
# -----------------------
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -----------------------
# 配置
# -----------------------
cfg = Config()

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text=text)
tokens = tokenizer.encode(text)

cfg.vocab_size = tokenizer.vocab_size


dataset = TextDataset(tokens, cfg.block_size)

loader = DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
)


# -----------------------
# 固定一个 batch
# -----------------------
fixed_inputs, fixed_targets = next(iter(loader))

fixed_loader = [
    (
        fixed_inputs.to(cfg.device),
        fixed_targets.to(cfg.device),
    )
]


# -----------------------
# 模型
# -----------------------
model = TinyTransformer(
    vocab_size=cfg.vocab_size,
    d_model=cfg.d_model,
    num_heads=cfg.num_heads,
    d_ff=cfg.d_ff,
    num_layers=cfg.num_layers,
    block_size=cfg.block_size,
).to(cfg.device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.learning_rate,
)

criterion = nn.CrossEntropyLoss()
os.makedirs(cfg.checkpoint_dir, exist_ok=True)
run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(cfg.overfit_tensorboard_log_dir, run_name)
writer = SummaryWriter(log_dir=log_dir)
writer.add_text("experiment", "Overfit one batch test")
writer.add_text("config", json.dumps(cfg.__dict__, indent=4))
writer.add_text("model", str(model))
losses = []


# -----------------------
# Overfit One Batch
# -----------------------
epochs = 500

for epoch in range(epochs):

    loss = train_one_epoch(
        model=model,
        loader=fixed_loader,
        optimizer=optimizer,
        criterion=criterion,
        cfg=cfg,
    )
    losses.append(loss)
    writer.add_scalar("loss/overfit_one_batch", loss, epoch + 1)
    lr = optimizer.param_groups[0]["lr"]
    writer.add_scalar("learning_rate", lr, epoch + 1)

    if epoch % 10 == 0 or epoch == epochs - 1:

        model.eval()

        with torch.no_grad():

            outputs = model(fixed_inputs.to(cfg.device))

            pred = outputs.argmax(dim=-1)

            print(f"Epoch {epoch:03d}")
            print(f"Loss: {loss:.4f}")
            print("Pred:  ", pred[0][:20].tolist())
            print("Target:", fixed_targets[0][:20].tolist())
            print("-" * 60)

writer.close()
plot_loss_curve(losses, save_path=cfg.overfit_loss_plot_path)
print(f"Saved loss curve to {cfg.overfit_loss_plot_path}")
