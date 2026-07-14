# TinyTransformer

一个从零实现的字符级 Transformer 语言模型。

本项目参考 GPT 类 Decoder-only Transformer 架构，使用 PyTorch 手写实现完整训练流程，包括：

- 字符级 Tokenizer
- Token Embedding + Position Embedding
- Causal Self-Attention（因果自注意力）
- Multi-Head Attention（多头注意力）
- Pre-LN Transformer Block
- Feed Forward Network
- 训练与验证流程
- AMP 混合精度训练
- torch.compile 加速
- TensorBoard 日志记录
- Checkpoint 保存与加载
- 文本生成推理

项目目标不是训练大规模语言模型，而是通过一个小型可运行模型理解 GPT 类模型的核心结构与训练过程。


---

# 项目特点

## 1. 从零实现 Transformer Decoder

模型结构：

```

Input Text
|
Character Tokenizer
|
Token Embedding
+
Position Embedding
|
Transformer Blocks
|
LayerNorm
|
Linear LM Head
|
Next Token Prediction

```


每个 Transformer Block：

```

x
|
LayerNorm
|
Causal Multi-head Attention
|
Residual Connection
|
LayerNorm
|
Feed Forward Network
|
Residual Connection

```

采用 Pre-LN 结构，提高深层 Transformer 训练稳定性。


---

# 项目结构

```

TinyTransformer/

├── config.py                 # 模型与训练参数配置
├── dataset.py                # Next-token prediction 数据集
├── tokenizer.py              # 字符级 tokenizer
├── main.py                   # 完整训练入口
├── generate.py               # 加载 checkpoint 进行文本生成
├── overfit_one_batch.py      # 单 batch 过拟合测试

├── engine/
│   ├── train.py              # 单 epoch 训练逻辑
│   └── evaluate.py           # 验证逻辑

├── models/
│   ├── attention.py           # Causal Multi-head Attention
│   ├── block.py               # Transformer Block
│   ├── embedding.py           # Token/Position Embedding
│   ├── feedforward.py         # FFN
│   └── transformer.py         # TinyTransformer主体

├── utils/
│   ├── plot.py                # loss曲线绘制
│   └── seed.py                # 随机种子控制

├── data/
│   └── input.txt              # Shakespeare训练文本

├── checkpoints/
│   ├── lr_1e3.pth             # learning rate=1e-3训练结果
│   └── lr_3e4.pth             # learning rate=3e-4训练结果

└── exp_log.md                 # 实验记录

```


---

# 环境要求

推荐环境：

```

Python >= 3.10
PyTorch >= 2.0
CUDA >= 12

````

安装依赖：

```bash
pip install torch numpy matplotlib tensorboard tqdm
````

---

# 数据集

实验使用 Tiny Shakespeare 数据集。

数据经过字符级编码：

例如：

```
"To be or not to be"
```

转换为：

```
[19, 34, 5, 1, 14, ...]
```

模型任务：

给定前面的字符，预测下一个字符。

训练目标：

$$
P(x_t|x_1,x_2,...,x_{t-1})
$$

采用：

* 90% training data
* 10% validation data

---

# 模型配置

主要参数位于：

```
config.py
```

典型配置：

```python
block_size = 128

d_model = 128

num_heads = 8

num_layers = 2

d_ff = 512

dropout = 0.1

batch_size = 512

learning_rate = 3e-4
```

参数含义：

| 参数            | 含义                  |
| ------------- | ------------------- |
| block_size    | 输入上下文长度             |
| d_model       | hidden dimension    |
| num_heads     | attention head数量    |
| num_layers    | Transformer Block数量 |
| d_ff          | FFN隐藏层大小            |
| dropout       | dropout概率           |
| batch_size    | 每次训练样本数量            |
| learning_rate | AdamW学习率            |

---

# 训练方式

运行：

```bash
python main.py
```

训练过程包括：

1. 构建 tokenizer

2. 创建 Dataset 和 DataLoader

3. 初始化 Transformer

4. AdamW 优化

5. AMP 混合精度训练

6. 验证集评估

7. 保存最佳模型

训练输出：

```
Epoch [1/40]
train_loss=...
val_loss=...
```

---

# 加速优化

## AMP Mixed Precision

训练过程中使用：

```python
torch.amp.autocast()
```

降低显存占用，提高 GPU Tensor Core 利用率。

## torch.compile

使用：

```python
torch.compile(model)
```

通过 TorchInductor 对计算图进行优化。

## DataLoader 并行

使用：

```python
num_workers
pin_memory
```

提升数据加载效率。

---

# 实验记录

详细实验过程记录于：

```
exp_log.md
```

主要研究内容：

## 1. 模型深度实验

比较：

```
num_layers=1
num_layers=2
num_layers=3
num_layers=6
```

观察模型容量和过拟合关系。

结果：

* 更深模型训练 loss 降低更快
* 但容易出现 validation loss 上升
* 小数据集限制了模型泛化能力

---

## 2. 模型宽度实验

比较：

```
d_model=128
d_model=256
```

发现：

* 增大 hidden dimension 提升模型拟合能力
* 但可能加剧过拟合

---

## 3. Dropout实验

加入：

```python
dropout=0.1
```

观察到：

* 训练速度降低
* validation loss更加稳定
* 泛化能力提升

---

## 4. Weight Decay实验

优化器：

```python
AdamW
```

加入：

```python
weight_decay
```

观察：

* 对当前小模型提升有限
* 主要作用是抑制参数过拟合

---

## 5. Learning Rate实验

比较：

```
lr=3e-4

lr=1e-3
```

结果：

* 更大学习率前期下降更快
* 两者最终 validation loss 接近
* lr=3e-4训练更加平滑

详细结果见：

```
exp_log.md
```

---

# Checkpoint

训练结果保存：

```
checkpoints/
```

checkpoint包含：

* model parameters
* optimizer state
* epoch
* validation loss
* config
* tokenizer vocabulary

可以直接用于推理。

---

# 文本生成

运行：

```bash
python generate.py \
--checkpoint checkpoints/lr_3e4.pth \
--prompt "To be" \
--max-new-tokens 300
```

参数：

| 参数             | 说明   |
| -------------- | ---- |
| checkpoint     | 模型文件 |
| prompt         | 起始文本 |
| max-new-tokens | 生成长度 |
| temperature    | 随机程度 |

示例：

输入：

```
To be
```

输出：

```
To be in my great work...
```

模型能够学习 Shakespeare 文本中的：

* 词汇分布
* 标点规律
* 戏剧角色格式

---

# 训练总结

通过本项目完成：

* 手写 Transformer Decoder
* 理解 Self-Attention 数学过程
* 理解 Causal Mask
* 理解 Transformer Block结构
* 完成语言模型训练pipeline
* 掌握 checkpoint保存和推理
* 进行模型容量、正则化、学习率实验

该项目主要用于学习 Transformer 原理与训练流程，而非追求生成质量。

---

# Future Work

可能的改进方向：

* 更大规模文本训练
* Byte Pair Encoding tokenizer
* RoPE位置编码
* KV Cache推理优化
* Flash Attention
* 更完整GPT架构
* 分布式训练






