from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, cfg):

    model.train()
    pbar = tqdm(loader, desc="Training", leave=False)

    total_loss = 0.0
    total_tokens = 0

    for inputs, targets in pbar:
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()
        pbar.set_postfix(loss=total_loss / total_tokens if total_tokens > 0 else 0)
 
    if total_tokens == 0:
        raise ValueError("train loader produced no tokens")

    return total_loss / total_tokens
