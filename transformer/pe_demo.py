import torch
import math
import numpy as np

print("=== Position Encoding 计算逻辑演示 ===\n")

# 小规模示例参数
d_model = 8  # 简化为8维
max_seq_length = 5  # 只看5个位置

print(f"参数: d_model={d_model}, max_seq_length={max_seq_length}")
print(f"目标矩阵形状: ({max_seq_length}, {d_model})\n")

# 创建位置和维度的索引
pos = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)  # (5, 1)
i = torch.arange(0, d_model, 2, dtype=torch.float)  # [0, 2, 4, 6] - 偶数索引

print("位置索引 pos:")
print(pos.squeeze().tolist())  # [0, 1, 2, 3, 4]

print("\n维度索引 i (偶数):")
print(i.tolist())  # [0, 2, 4, 6]

# 计算除数项：10000^(2i/d_model)
div_term = torch.exp(i * (-math.log(10000.0) / d_model))  # (4,)
print(f"\n除数项 div_term:")
print(div_term.tolist())

# 计算所有位置和维度的组合
pos_i = pos * div_term  # (5, 1) * (4,) -> (5, 4) 广播
print(f"\npos * div_term 的形状: {pos_i.shape}")
print("pos * div_term 矩阵:")
print(pos_i.numpy())

# 创建最终的位置编码矩阵
pe = torch.zeros(max_seq_length, d_model)
pe[:, 0::2] = torch.sin(pos_i)  # 偶数位置用sin
pe[:, 1::2] = torch.cos(pos_i)  # 奇数位置用cos

print(f"\n最终Position Encoding矩阵形状: {pe.shape}")
print("Position Encoding矩阵:")
print(pe.numpy().round(3))

print("\n=== 使用方式演示 ===")
# 模拟输入
batch_size = 2
seq_len = 3  # 只用前3个位置
input_embeddings = torch.randn(batch_size, seq_len, d_model)

print(f"输入embeddings形状: {input_embeddings.shape}")
print(f"需要的PE形状: {pe[:seq_len].shape}")

# 添加位置编码
output = input_embeddings + pe[:seq_len]  # 广播：(2,3,8) + (3,8) -> (2,3,8)
print(f"输出形状: {output.shape}")
print("✅ 位置编码成功添加到输入上！") 