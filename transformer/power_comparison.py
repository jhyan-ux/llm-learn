import torch
import math
import numpy as np

print("=== 为什么使用 exp(log()) 而不是直接 pow ===\n")

d_model = 512
i = torch.arange(0, d_model, 2, dtype=torch.float)  # [0, 2, 4, 6, ..., 510]

print(f"d_model = {d_model}")
print(f"i 的范围: {i[:5].tolist()} ... {i[-5:].tolist()}")
print(f"指数 2i/d_model 的范围: {(2*i[0]/d_model):.3f} 到 {(2*i[-1]/d_model):.3f}")

print("\n=== 方法1: 直接使用 pow (可能有问题) ===")
try:
    # 直接计算 10000^(-2i/d_model)
    exponent = -2 * i / d_model
    direct_pow = torch.pow(10000.0, exponent)
    
    print(f"✅ 直接pow成功")
    print(f"最小值: {direct_pow.min():.10f}")
    print(f"最大值: {direct_pow.max():.10f}")
    print(f"前5个值: {direct_pow[:5].tolist()}")
    print(f"后5个值: {direct_pow[-5:].tolist()}")
    
except Exception as e:
    print(f"❌ 直接pow失败: {e}")

print("\n=== 方法2: 使用 exp(log()) (推荐方式) ===")
try:
    # 使用 exp(log()) 形式
    exp_log = torch.exp(i * (-math.log(10000.0) / d_model))
    
    print(f"✅ exp(log())成功")
    print(f"最小值: {exp_log.min():.10f}")
    print(f"最大值: {exp_log.max():.10f}")
    print(f"前5个值: {exp_log[:5].tolist()}")
    print(f"后5个值: {exp_log[-5:].tolist()}")
    
except Exception as e:
    print(f"❌ exp(log())失败: {e}")

print("\n=== 数值比较 ===")
try:
    # 比较两种方法的结果
    diff = torch.abs(direct_pow - exp_log)
    print(f"两种方法的最大差异: {diff.max():.15f}")
    print(f"是否数值相等: {torch.allclose(direct_pow, exp_log)}")
    
except:
    print("无法比较（某种方法失败了）")

print("\n=== 计算详解 ===")
print("exp(log())方法的数学推导:")
print("10000^(-2i/d_model) = exp(log(10000^(-2i/d_model)))")
print("                    = exp(-2i/d_model * log(10000))")
print("                    = exp(i * (-2*log(10000)/d_model))")
print(f"其中 -log(10000) = {-math.log(10000.0):.6f}")

print("\n=== 极端情况测试 ===")
# 测试非常大的指数
large_i = torch.tensor([500.0])
large_exp = -2 * large_i / d_model

print(f"极端情况: i={large_i.item()}, 指数={large_exp.item():.6f}")

try:
    result1 = torch.pow(10000.0, large_exp)
    print(f"直接pow结果: {result1.item():.15f}")
except:
    print("❌ 直接pow在极端情况下失败")

try:
    result2 = torch.exp(large_i * (-math.log(10000.0) / d_model))
    print(f"exp(log())结果: {result2.item():.15f}")
except:
    print("❌ exp(log())在极端情况下失败")

print("\n=== 结论 ===")
print("1. 数学上两种方法完全等价")
print("2. exp(log())形式在数值计算上更稳定")
print("3. 这是深度学习中的标准做法")
print("4. 避免了直接计算大指数可能的精度问题") 