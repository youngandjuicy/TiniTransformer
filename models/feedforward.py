import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.Linear(d_model, d_ff)
        self.ln2 = nn.Linear(d_ff, d_model)
        self.ffn = nn.Sequential(
            self.ln1,
            nn.GELU(),
            self.ln2
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.ln1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.ln2.weight, mean=0.0, std=0.02)

    def forward(self, x):
        output = self.ffn(x)
        output = self.dropout(output)
        return output