import torch


@torch.no_grad()
def evaluate(model, loader, criterion, cfg):

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        outputs = model(inputs)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()

    if total_tokens == 0:
        raise ValueError("validation loader produced no tokens")

    return total_loss / total_tokens
