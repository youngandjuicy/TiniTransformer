import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(in_features=d_model, out_features=d_model)
        self.k_proj = nn.Linear(in_features=d_model, out_features=d_model)
        self.v_proj = nn.Linear(in_features=d_model, out_features=d_model)
    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        print("Q:", Q.shape)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_model)
        print("scores:", scores.shape)
        scores_weights = F.softmax(scores, dim=-1)
        print("scores_weight[0, 0]:", scores_weights[0, 0])
        print("scores_weight[0, 0].sum():", scores_weights[0, 0].sum().item())
        output = scores_weights @ V
        return output
    
if __name__ == "__main__":
    x = torch.randn(8, 16, 64)  # batch_size=8, seq_len=16, d_model=64
    self_attention = SelfAttention(d_model=64)
    output = self_attention(x)
    print("output:", output.shape)
    assert output.shape == x.shape

