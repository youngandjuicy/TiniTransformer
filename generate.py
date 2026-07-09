import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(self, idx, max_new_tokens):

    for _ in range(max_new_tokens):

        logits = self(idx)

        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1) # 模型输出的 logits 转换为概率分布

        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        idx = torch.cat(
            [idx, next_token],
            dim=1
        )

    return idx