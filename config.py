import torch

class Config:
    vocab_size = None      # 后面由 tokenizer 决定
    block_size = 128

    d_model = 256
    num_heads = 8
    num_layers = 6
    d_ff = 256*4

    batch_size = 32

    learning_rate = 3e-4

    epochs = 500

    device = "cuda" if torch.cuda.is_available() else "cpu"