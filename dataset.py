import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):

    def __init__(self, tokens, block_size):
        if len(tokens) <= block_size:
            raise ValueError("text length must be larger than block_size")
        self.block_size = block_size
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.block_size
        return torch.tensor(self.tokens[start_idx:end_idx], dtype=torch.long), torch.tensor(self.tokens[start_idx + 1:end_idx + 1], dtype=torch.long)

if __name__ == "__main__":
    # Example usage
    tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    block_size = 3
    dataset = TextDataset(tokens, block_size)
    
    print(f"Length of dataset: {len(dataset)}")
    for i in range(len(dataset)):
        print(f"Item {i}: {dataset[i]}")
