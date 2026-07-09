import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(in_features=d_model, out_features=d_model)
        self.k_proj = nn.Linear(in_features=d_model, out_features=d_model)
        self.v_proj = nn.Linear(in_features=d_model, out_features=d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)

        # 新增：创建 causal mask
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # 修改为num_heads头注意力（batch_size, seq_len, d_model） -> (batch_size, seq_len, num_heads, d_head) -> (batch_size, num_heads, seq_len, d_head)
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, d_head)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        seq_len = x.size(1)
        mask = self.mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        scores_weights = F.softmax(scores, dim=-1)

        output_heads = scores_weights @ V
        output = output_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # shape: (batch_size, seq_len, d_model)
        output = self.out_proj(output)

        assert output.shape == x.shape
        return output
    


