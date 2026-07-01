import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, token_nums, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(token_nums, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=0.02) # 提高训练的稳定性
    def forward(self, x):
        output = self.weight[x]
        return output