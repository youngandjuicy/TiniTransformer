import os
import torch
import numpy as np
from dataset import TextDataset
from torch.utils.data import DataLoader
from models.transformer import TinyTransformer
from config import Config
from tokenizer import CharTokenizer

cfg = Config()
# text = np.random.randint(5, 105, size=(10000,)).tolist()  # 示例文本
with open(
    "data/input.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

tokenizer = CharTokenizer(text)
tokens = tokenizer.encode(text)
cfg.vocab_size = tokenizer.vocab_size

dataset = TextDataset(tokens, cfg.block_size)
train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

model = TinyTransformer(vocab_size = cfg.vocab_size, d_model=cfg.d_model, num_heads=cfg.num_heads, d_ff=cfg.d_ff, num_layers=cfg.num_layers).to(cfg.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# for epoch in range(cfg.epochs):
#     model.train()
#     running_loss = 0.0
#     for inputs, targets in train_loader:
#         inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * targets.numel()  # 累加损失，乘以样本数
#     avg_loss = running_loss / len(dataset) / cfg.block_size  # 计算平均损失
#     print(f"Epoch [{epoch+1}/{cfg.epochs}], Loss: {avg_loss:.4f}")

# 取出一个固定的 batch（用于过拟合验证）
fixed_inputs, fixed_targets = next(iter(train_loader))
fixed_inputs = fixed_inputs.to(cfg.device)
fixed_targets = fixed_targets.to(cfg.device)

for epoch in range(cfg.epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(fixed_inputs)
    loss = criterion(outputs.reshape(-1, outputs.size(-1)), fixed_targets.reshape(-1))
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{cfg.epochs}], Loss: {loss.item():.4f}")
    # 每 10 个 epoch 打印一次预测对比
    if epoch % 10 == 0:
        with torch.no_grad():
            pred = outputs.argmax(dim=-1)  # 注意：此时 outputs 是最后一次前向的结果
            print(f"Epoch {epoch}:")
            print("Pred:  ", pred[0][:20].tolist())
            print("Target:", fixed_targets[0][:20].tolist())
            print("Loss:", loss.item())

os.makedirs("./checkpoints", exist_ok=True)
torch.save(
    {
        "model": model.state_dict(),
        "config": cfg,
        "tokenizer": tokenizer,
    },
    "./checkpoints/checkpoint.pth"
)
