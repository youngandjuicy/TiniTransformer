import torch

@torch.no_grad()
def evaluate(model, loader, criterion, cfg):

    model.eval()

    total_loss = 0
    total_tokens = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        outputs = model(inputs)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

        total_loss += loss.item() * targets.numel() # 累加得到一轮（所有batch各走一步后）的总损失
        total_tokens += targets.numel() # 累加实际处理的 token 总数

    return total_loss / total_tokens  # 平均损失