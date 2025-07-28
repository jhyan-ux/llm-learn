import torch
import torch.nn as nn

print("=== 什么时候需要 .contiguous() ===\n")

# 创建测试数据
x = torch.randn(2, 3, 4)
print(f"原始形状: {x.shape}, 连续性: {x.is_contiguous()}")

# Case 1: transpose后直接使用（大多数操作都OK）
print("\n--- Case 1: transpose后直接使用 ---")
x_t = x.transpose(0, 1)
print(f"transpose后形状: {x_t.shape}, 连续性: {x_t.is_contiguous()}")

# 这些操作都不需要contiguous
y1 = x_t + 1                    # ✅ 加法OK
y2 = torch.relu(x_t)           # ✅ 激活函数OK  
y3 = x_t * 2                   # ✅ 乘法OK
print("✅ 基本数学运算都不需要contiguous")

# Case 2: transpose后使用view()
print("\n--- Case 2: transpose后使用view ---")
try:
    y4 = x_t.view(-1)  # ❌ 可能失败
    print("✅ 不用contiguous的view居然成功了！")
except RuntimeError as e:
    print(f"❌ 不用contiguous的view失败: {str(e)[:50]}...")

# 使用contiguous保证成功
y5 = x_t.contiguous().view(-1)  # ✅ 一定成功
print("✅ 使用contiguous的view一定成功")

# Case 3: 在实际的MultiHeadAttention中
print("\n--- Case 3: MultiHeadAttention实际情况 ---")
batch_size, num_heads, seq_len, d_k = 2, 8, 10, 64
attention_output = torch.randn(batch_size, num_heads, seq_len, d_k)

# 我们的操作序列
step1 = attention_output.transpose(1, 2)  # (batch, seq, heads, d_k)  
print(f"Step1 transpose: 连续性 = {step1.is_contiguous()}")

# 不同的后续操作
print("\n这些操作不需要contiguous:")
result1 = step1 + 1                                    # ✅ 加法
result2 = torch.matmul(step1, torch.randn(d_k, 100))  # ✅ 矩阵乘法
linear = nn.Linear(d_k, 100)
result3 = linear(step1)                                # ✅ 线性层
print("✅ 数学运算、矩阵乘法、线性层都OK")

print("\n这个操作需要contiguous:")
try:
    result4 = step1.view(batch_size, seq_len, num_heads * d_k)  # 可能失败
    print("✅ view操作成功（但不保证总是成功）")
except RuntimeError:
    print("❌ view操作失败（需要contiguous）")

# 保险的做法
result5 = step1.contiguous().view(batch_size, seq_len, num_heads * d_k)  # ✅ 保证成功
print("✅ contiguous后的view保证成功")

print("\n=== 最佳实践建议 ===")
print("1. 只在需要view()操作前使用contiguous()")
print("2. 如果不确定，使用contiguous()是安全的（几乎无性能损失）")
print("3. 大多数PyTorch操作都能处理非连续tensor")
print("4. 在生产代码中，transpose().contiguous().view()是标准模式") 