# LLM学习项目 🚀

这是一个专注于大语言模型（LLM）学习和实现的项目仓库，从底层架构开始，逐步构建现代LLM的核心组件。

## 📁 项目结构

```
LLM/
├── README.md                    # 项目总览和结构说明
├── .gitignore                   # Git忽略文件配置
│
└── transformer/                 # Transformer架构实现
    ├── README.md               # Transformer详细实现说明
    ├── transformer.py          # 完整的Transformer Encoder实现
```

## 🎯 项目特色

### 📚 **教学导向**
- **从零实现**：不依赖任何高级库，纯PyTorch构建
- **详细注释**：每个函数都有完整的维度标注和原理解释
- **渐进式学习**：从基础组件到完整架构，层层递进

### 🔧 **工程实践**
- **现代架构**：采用Pre-LayerNorm设计，训练更稳定
- **完整功能**：包含残差连接、Dropout、位置编码等所有关键组件
- **可扩展性**：支持任意层数和维度配置

### 🧪 **实验验证**
- **单元测试**：每个组件都有独立的测试和演示
- **性能分析**：内存使用、参数统计、计算复杂度分析
- **数值验证**：确保实现的数学正确性

## 🚀 快速开始

### 环境要求
```bash
Python 3.7+
PyTorch 1.8+
```

### 基础使用
```python
from transformer.transformer import TransformerEncoder

# 创建模型
model = TransformerEncoder(
    vocab_size=10000,      # 词汇表大小
    d_model=512,           # 模型维度
    num_heads=8,           # 注意力头数
    d_ff=2048,             # 前馈网络维度
    num_layers=6,          # 编码器层数
    dropout=0.1            # Dropout比例
)

# 前向传播（输入为token IDs）
import torch
token_ids = torch.randint(0, 10000, (2, 50))  # (batch_size, seq_len)
output = model(token_ids)  # (batch_size, seq_len, d_model)
```

## 📚 参考资料

- 📄 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原论文