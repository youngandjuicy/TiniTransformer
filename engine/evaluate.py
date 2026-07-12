import torch
from tqdm import tqdm

@torch.no_grad()
def evaluate(model, loader, criterion, cfg):

    model.eval()
    pbar = tqdm(loader, desc="Validation", leave=False)

    total_loss = 0.0
    total_tokens = 0

    for inputs, targets in pbar:
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        outputs = model(inputs)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()
        pbar.set_postfix(loss=total_loss / total_tokens if total_tokens > 0 else 0)

    if total_tokens == 0:
        raise ValueError("validation loader produced no tokens")

    return total_loss / total_tokens
