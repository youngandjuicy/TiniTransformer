import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, block_size, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.token_embedding.weight,mean = 0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x):
        token_embeds = self.token_embedding(x)
        seq_len = x.size(1)
        pos_embeds = self.position_embedding(torch.arange(seq_len, device=x.device))
        output = token_embeds + pos_embeds
        return self.dropout(output)