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
        print("Q:", Q.shape)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        print("scores:", scores.shape)
        scores_weights = F.softmax(scores, dim=-1)
        print("scores_weight[0, 0, 0]:", scores_weights[0, 0, 0])
        print("scores_weight[0, 0, 0].sum():", scores_weights[0, 0, 0].sum().item())
        output_heads = scores_weights @ V
        output = output_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # shape: (batch_size, seq_len, d_model)
        output = self.out_proj(output)
        print("output_proj:", output.shape)
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

if __name__ == "__main__":
    x = torch.randn(8, 16, 64)  # batch_size=8, seq_len=16, d_model=64
    multi_head_attention = MultiHeadAttention(d_model=64, num_heads=8)
    output = multi_head_attention(x)
    print("output:", output.shape)


