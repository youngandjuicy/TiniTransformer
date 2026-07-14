import torch
import torch.nn as nn
from models.embedding import Embedding
from models.block import TransformerBlock

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, block_size, dropout):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, block_size, dropout)
        self.layernorm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, block_size, dropout)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.lm_head.weight = self.embedding.token_embedding.weight  # weight tying


    def forward(self, x):
        # embedding
        x = self.embedding(x)
        # transformer blocks * num_layers
        for block in self.blocks:
            x = block(x)
        # layer norm
        x = self.layernorm(x)
        # linear
        x = self.lm_head(x)
        return x
    
if __name__ == "__main__":
    x = torch.randint(0, 1000, (8, 16))  # batch_size=8, seq_len=16, vocab_size=1000
    transformer = TinyTransformer(vocab_size=1000, d_model=64, num_heads=8, d_ff=64*4, num_layers=3, block_size=16)
    output = transformer(x)
    print("output:", output.shape)