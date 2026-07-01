import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feedforward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)
        self.residual = nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        input_shape = x.shape
        # norm
        x = self.norm1(x)
        # attention
        residual = self.residual(x)
        x = self.attention(x)
        # residual
        x = x + residual
        assert x.shape == input_shape

        # norm
        x = self.norm2(x)
        # ffn
        residual = self.residual(x)
        x = self.ffn(x)
        # residual
        x = x + residual
        assert x.shape == input_shape

        return x
    

if __name__ == "__main__":
    x = torch.randn(8, 16, 64)  # batch_size=8, seq_len=16, d_model=64
    transformer = TransformerBlock(d_model=64, num_heads=8, d_ff=64*4)
    output = transformer(x)
    print("output:", output.shape)