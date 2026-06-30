import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(in_features=d_model, out_features=d_model)
        self.k_proj = nn.Linear(in_features=d_model, out_features=d_model)
        self.v_proj = nn.Linear(in_features=d_model, out_features=d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # 修改为num_heads头注意力（batch_size, seq_len, d_model） -> (batch_size, seq_len, num_heads, d_head) -> (batch_size, num_heads, seq_len, d_head)
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, d_head)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        scores_weights = F.softmax(scores, dim=-1)

        output_heads = scores_weights @ V
        output = output_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # shape: (batch_size, seq_len, d_model)
        output = self.out_proj(output)

        assert output.shape == x.shape
        return output
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        output = self.ffn(x)
        return output

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
        # attention
        residual = self.residual(x)
        x = self.attention(x)
        # residual
        x = x + residual
        # norm
        x = self.norm1(x)
        assert x.shape == input_shape

        # ffn
        residual = self.residual(x)
        x = self.ffn(x)
        # residual
        x = x + residual
        # norm
        x = self.norm2(x)
        assert x.shape == input_shape

        return x
    
class Embedding(nn.Module):
    def __init__(self, token_nums, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(token_nums, d_model))
    def forward(self, x):
        output = self.weight[x]
        return output


if __name__ == "__main__":
    x = torch.randn(8, 16, 64)  # batch_size=8, seq_len=16, d_model=64
    transformer = TransformerBlock(d_model=64, num_heads=8, d_ff=64*4)
    output = transformer(x)
    print("output:", output.shape)


