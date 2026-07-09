import os
import torch
import numpy as np
from dataset import TextDataset
from torch.utils.data import DataLoader
from models.transformer import TinyTransformer
from config import Config
from tokenizer import CharTokenizer
from utils.seed import set_seed
from engine.train import train_one_epoch
from engine.evaluate import evaluate

cfg = Config()
set_seed(cfg.seed)
# text = np.random.randint(5, 105, size=(10000,)).tolist()  # 示例文本
with open(
    "data/input.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

# 创建 tokenizer 并编码文本
tokenizer = CharTokenizer(text)
tokens = tokenizer.encode(text)
cfg.vocab_size = tokenizer.vocab_size

# 创建数据集和数据加载器
dataset = TextDataset(tokens, cfg.block_size)
split = int(0.9 * len(tokens))
train_tokens = tokens[:split]
val_tokens = tokens[split:]
train_dataset = TextDataset(train_tokens, cfg.block_size)
val_dataset = TextDataset(val_tokens, cfg.block_size)
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

# 创建模型、优化器和损失函数
model = TinyTransformer(vocab_size = cfg.vocab_size, d_model=cfg.d_model, num_heads=cfg.num_heads, d_ff=cfg.d_ff, num_layers=cfg.num_layers, block_size=cfg.block_size).to(cfg.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
criterion = torch.nn.CrossEntropyLoss() # 自带softmax

# 训练和评估循环
for epoch in range(cfg.epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg)
    val_loss = evaluate(model, val_loader, criterion, cfg)

    print(f"Epoch [{epoch+1}/{cfg.epochs}], Loss: {train_loss:.4f}")
    print(f"Epoch [{epoch+1}/{cfg.epochs}], Validation Loss: {val_loss:.4f}")

# # 取出一个固定的 batch（用于过拟合验证）
# fixed_inputs, fixed_targets = next(iter(train_loader))
# fixed_inputs = fixed_inputs.to(cfg.device)
# fixed_targets = fixed_targets.to(cfg.device)

# for epoch in range(cfg.epochs):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(fixed_inputs)
#     loss = criterion(outputs.reshape(-1, outputs.size(-1)), fixed_targets.reshape(-1))
#     loss.backward()
#     optimizer.step()

#     print(f"Epoch [{epoch+1}/{cfg.epochs}], Loss: {loss.item():.4f}")
#     # 每 10 个 epoch 打印一次预测对比
#     if epoch % 10 == 0:
#         with torch.no_grad():
#             pred = outputs.argmax(dim=-1)  # 注意：此时 outputs 是最后一次前向的结果
#             print(f"Epoch {epoch}:")
#             print("Pred:  ", pred[0][:20].tolist())
#             print("Target:", fixed_targets[0][:20].tolist())
#             print("Loss:", loss.item())

os.makedirs("./checkpoints", exist_ok=True)
torch.save(
    {
        "model": model.state_dict(),

        "config": cfg.__dict__,

        "tokenizer": {
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
        }
    },
    "./checkpoints/checkpoint.pth"
)
