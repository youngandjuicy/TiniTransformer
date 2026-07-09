# TinyTransformer

A minimal character-level Transformer language model trained on Tiny Shakespeare.

This project is intentionally small and readable. It includes a causal self-attention model, a character tokenizer, training and evaluation loops, TensorBoard logging, checkpoint saving, static loss plotting, an overfit-one-batch sanity check, and text generation from saved checkpoints.

## Project Structure

```text
TinyTransformer/
|-- config.py                 # Training and model configuration
|-- dataset.py                # Character-level next-token dataset
|-- tokenizer.py              # Character tokenizer
|-- main.py                   # Full training entry point
|-- overfit_one_batch.py      # Single-batch overfit sanity check
|-- generate.py               # Text generation from checkpoint
|-- engine/
|   |-- train.py              # One-epoch training loop
|   `-- evaluate.py           # Evaluation loop
|-- models/
|   |-- attention.py          # Causal multi-head self-attention
|   |-- block.py              # Pre-LN Transformer block
|   |-- embedding.py          # Token and position embeddings
|   |-- feedforward.py        # Feed-forward network
|   `-- transformer.py        # TinyTransformer model
|-- utils/
|   |-- plot.py               # Static loss curve plotting
|   `-- seed.py               # Reproducibility helper
`-- data/
    `-- input.txt             # Training text
```

## Requirements

Recommended environment:

```bash
pip install torch numpy matplotlib tensorboard
```

On Google Colab, PyTorch is usually preinstalled. If needed:

```python
!pip install matplotlib tensorboard
```

## Data

Place the training text at:

```text
data/input.txt
```

The current code expects a plain text file and trains a character-level language model.

## Configuration

Main hyperparameters are in `config.py`:

```python
self.block_size = 128
self.d_model = 256
self.num_heads = 8
self.num_layers = 6
self.d_ff = 256 * 4
self.batch_size = 32
self.learning_rate = 3e-4
self.epochs = 500
```

The device is selected automatically:

```python
"cuda" if torch.cuda.is_available() else "cpu"
```

## Sanity Check: Overfit One Batch

Before full training, run:

```bash
python overfit_one_batch.py
```

This trains repeatedly on a fixed batch. The loss should drop clearly. If it does not, check the model, data pipeline, or optimizer before starting a long run.

Outputs:

```text
checkpoints/overfit_one_batch_loss.png
runs/overfit_one_batch/
```

## Train

Run full training:

```bash
python main.py
```

The script:

- builds a character tokenizer from `data/input.txt`
- splits tokens into 90% train and 10% validation
- trains the Transformer model
- logs train and validation loss to TensorBoard
- saves the best checkpoint by validation loss
- saves the final checkpoint
- writes a static loss curve

Outputs:

```text
checkpoints/best.pth
checkpoints/final.pth
checkpoints/loss.png
runs/tiny_transformer/
```

## TensorBoard

Start TensorBoard from the project root:

```bash
tensorboard --logdir runs
```

In Colab:

```python
%load_ext tensorboard
%tensorboard --logdir runs
```

Then run training in another cell:

```python
!python main.py
```

## Generate Text

Generate from the best checkpoint:

```bash
python generate.py --checkpoint checkpoints/best.pth --prompt "To be" --max-new-tokens 300
```

Useful arguments:

```bash
--checkpoint checkpoints/best.pth
--prompt "To be"
--max-new-tokens 300
--temperature 1.0
```

Lower temperature makes output more conservative. Higher temperature makes it more random.

## Checkpoints

Each checkpoint stores:

- model parameters
- optimizer state
- epoch
- train loss
- validation loss
- config values
- tokenizer vocabulary

This makes the checkpoint sufficient for later inference with `generate.py`.

## Notes

- `checkpoints/` and `runs/` are ignored by Git because they are training outputs.
- For Colab training, keep a copy of important checkpoints in Google Drive or download them after training.
- The model is intentionally compact and educational rather than optimized for maximum throughput.
