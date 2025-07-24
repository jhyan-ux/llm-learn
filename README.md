# LLM学习项目

这是一个专注于大语言模型（LLM）学习和实现的项目仓库。

## 项目结构

```
LLM/
├── transformer/          # Transformer架构实现
│   ├── transformer.py   # 完整的Transformer Encoder实现
│   ├── pe_demo.py       # Position Encoding演示
│   └── ...              # 其他相关文件
└── README.md            # 项目说明
```

## Transformer实现

- ✅ **MultiHeadAttention** - 多头注意力机制
- ✅ **PositionEncoding** - 位置编码
- ✅ **FeedForward** - 前馈神经网络
- ✅ **EncoderLayer** - 编码器层
- ✅ **TransformerEncoder** - 完整的Transformer编码器

### 特点

- 🔥 采用现代Pre-LayerNorm架构
- 🔥 完整的残差连接和Dropout正则化
- 🔥 支持任意层数和维度配置
- 🔥 GPU兼容设计

## 使用方法

```python
from transformer.transformer import TransformerEncoder

# 创建Transformer模型
model = TransformerEncoder(
    vocab_size=1000,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    dropout=0.1
)

# 前向传播
token_ids = torch.randint(0, 1000, (batch_size, seq_length))
output = model(token_ids)
```

## 学习资源

- 📚 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原始论文
- 🎓 详细的代码注释和维度标注
- 💡 完整的测试用例

## 环境要求

- Python 3.7+
- PyTorch 1.8+

## 作者

计算机视觉研究生，专注于LLM学习与实现

---

*这个项目是学习Transformer架构的完整实现，从零开始构建，包含详细的教学性注释。* 