import torch
import torch.nn as nn
from models.attention import MultiHeadAttention
from models.feedforward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, block_size):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, block_size=block_size)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)
        self.residual = nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        input_shape = x.shape
        residual = self.residual(x)
        # norm
        x = self.norm1(x)
        # attention
        x = self.attention(x)
        # residual
        x = x + residual
        assert x.shape == input_shape

        # norm
        residual = self.residual(x)
        x = self.norm2(x)
        # ffn
        x = self.ffn(x)
        # residual
        x = x + residual
        assert x.shape == input_shape

        return x
    

if __name__ == "__main__":
    x = torch.randn(8, 16, 64)  # batch_size=8, seq_len=16, d_model=64
    transformer = TransformerBlock(d_model=64, num_heads=8, d_ff=64*4, block_size=16)
    output = transformer(x)
    print("output:", output.shape)