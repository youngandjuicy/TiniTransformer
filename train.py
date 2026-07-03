import torch
from dataset import TextDataset
from torch.utils.data import DataLoader
from models.transformer import TiniTransformer

tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
block_size = 3
dataset = TextDataset(tokens, block_size)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = TiniTransformer(vocab_size = len(tokens), d_model=512, num_heads=8, d_ff=512*4, num_layers=6)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(dataset):.4f}")
