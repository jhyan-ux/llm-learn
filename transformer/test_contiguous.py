import torch

# 演示contiguous的必要性
print("=== 演示 .contiguous() 的必要性 ===\n")

# 创建一个简单的tensor
x = torch.randn(2, 3, 4)
print(f"原始tensor形状: {x.shape}")
print(f"原始tensor是否连续: {x.is_contiguous()}")

# 进行transpose操作
x_transposed = x.transpose(0, 1)  # 交换第0和第1维
print(f"\n转置后形状: {x_transposed.shape}")
print(f"转置后是否连续: {x_transposed.is_contiguous()}")

# 尝试直接view (这会报错!)
try:
    x_viewed = x_transposed.view(-1)  # 尝试flatten
    print("✅ view操作成功!")
except RuntimeError as e:
    print(f"❌ view操作失败: {e}")

# 使用contiguous()后再view
x_contiguous = x_transposed.contiguous()
print(f"\ncontiguous()后是否连续: {x_contiguous.is_contiguous()}")

try:
    x_viewed = x_contiguous.view(-1)  # 现在可以flatten了
    print(f"✅ contiguous()后view成功! 结果形状: {x_viewed.shape}")
except RuntimeError as e:
    print(f"❌ 仍然失败: {e}")

print("\n=== Transformer中的具体情况 ===")
# 模拟我们代码中的情况
batch_size, num_heads, seq_len, d_k = 2, 8, 10, 64
attention_output = torch.randn(batch_size, num_heads, seq_len, d_k)

print(f"attention_output形状: {attention_output.shape}")
print(f"是否连续: {attention_output.is_contiguous()}")

# transpose操作
transposed = attention_output.transpose(1, 2)
print(f"\ntranspose后形状: {transposed.shape}")
print(f"transpose后是否连续: {transposed.is_contiguous()}")

# 不使用contiguous的情况
try:
    wrong_view = transposed.view(batch_size, seq_len, num_heads * d_k)
    print("✅ 不用contiguous也成功了!")
except RuntimeError as e:
    print(f"❌ 不用contiguous失败: {e}")
    
# 使用contiguous的情况（这是好习惯）
correct_view = transposed.contiguous().view(batch_size, seq_len, num_heads * d_k)
print(f"✅ 使用contiguous成功! 形状: {correct_view.shape}") 