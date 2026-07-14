import torch


class Config:
    def __init__(self):
        self.vocab_size = None
        self.block_size = 128

        self.d_model = 128
        self.num_heads = 8
        self.num_layers = 4
        self.d_ff = self.d_model * 4
        self.dropout = 0.1
        self.weight_decay = 0.0
        self.scheduler = "cosine"

        self.batch_size = 512
        self.learning_rate = 3e-4
        self.epochs = 20
        self.seed = 42
        self.num_workers = 8

        self.data_path = "data/input.txt"
        self.checkpoint_dir = "checkpoints"
        self.loss_plot_path = "checkpoints/loss.png"
        self.overfit_loss_plot_path = "checkpoints/overfit_one_batch_loss.png"
        self.tensorboard_log_dir = "runs/tiny_transformer"
        self.overfit_tensorboard_log_dir = "runs/overfit_one_batch"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
