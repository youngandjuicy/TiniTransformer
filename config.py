import torch


class Config:
    def __init__(self):
        self.vocab_size = None
        self.block_size = 128

        self.d_model = 256
        self.num_heads = 8
        self.num_layers = 6
        self.d_ff = 256 * 4

        self.batch_size = 32
        self.learning_rate = 3e-4
        self.epochs = 500
        self.seed = 42

        self.data_path = "data/input.txt"
        self.checkpoint_dir = "checkpoints"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
