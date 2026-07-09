import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from tokenizer import CharTokenizer
from dataset import TextDataset
from models.transformer import TinyTransformer
from engine.train import train_one_epoch


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