"""
LoRA参数效率分析
帮助理解什么时候LoRA有效，什么时候无效
"""

def analyze_efficiency(in_features, out_features, rank):
    """分析给定配置下的参数效率"""
    original_params = in_features * out_features
    lora_params = rank * (in_features + out_features)
    
    efficiency = (original_params - lora_params) / original_params * 100
    is_efficient = lora_params < original_params
    
    return {
        'original_params': original_params,
        'lora_params': lora_params,
        'efficiency_percentage': efficiency,
        'is_efficient': is_efficient,
        'reduction_ratio': original_params / lora_params if lora_params > 0 else float('inf')
    }

def find_optimal_rank(in_features, out_features):
    """找到使LoRA有效的最大rank"""
    # LoRA有效条件: r(d + k) < d × k
    # 即: r < (d × k) / (d + k)
    max_effective_rank = (in_features * out_features) / (in_features + out_features)
    return int(max_effective_rank)

def comprehensive_analysis():
    """全面分析不同场景下的LoRA效率"""
    print("=== LoRA参数效率分析 ===\n")
    
    # 测试场景
    scenarios = [
        ("小矩阵(当前)", 4, 3, 2),
        ("小矩阵(优化rank)", 4, 3, 1),
        ("中等矩阵", 64, 32, 8),
        ("大矩阵(GPT-like)", 768, 768, 64),
        ("超大矩阵", 4096, 4096, 256),
    ]
    
    print("场景分析:")
    print("-" * 80)
    print(f"{'场景':<15} {'矩阵尺寸':<12} {'rank':<6} {'原始参数':<10} {'LoRA参数':<10} {'效率':<10} {'状态'}")
    print("-" * 80)
    
    for name, d, k, r in scenarios:
        result = analyze_efficiency(d, k, r)
        status = "✅有效" if result['is_efficient'] else "❌无效"
        
        print(f"{name:<15} {d}×{k:<9} {r:<6} {result['original_params']:<10} "
              f"{result['lora_params']:<10} {result['efficiency_percentage']:>6.1f}% {status}")
    
    print("-" * 80)
    
    # 分析当前问题
    print(f"\n当前问题分析:")
    current_result = analyze_efficiency(4, 3, 2)
    optimal_rank = find_optimal_rank(4, 3)
    
    print(f"  矩阵尺寸: 4×3")
    print(f"  当前rank: 2")
    print(f"  最大有效rank: {optimal_rank}")
    print(f"  当前效率: {current_result['efficiency_percentage']:.1f}%")
    print(f"  问题: rank太大，超过了有效范围")
    
    # 修复建议
    print(f"\n修复建议:")
    if optimal_rank >= 1:
        fixed_result = analyze_efficiency(4, 3, optimal_rank)
        print(f"  1. 将rank降低到 {optimal_rank}")
        print(f"     → 参数减少: {fixed_result['efficiency_percentage']:.1f}%")
    else:
        print(f"  1. 这个矩阵太小，不适合用LoRA")
        print(f"  2. 建议在更大的矩阵上测试LoRA")
    
    print(f"  3. 或者使用更大的测试矩阵")

def interactive_calculator():
    """交互式效率计算器"""
    print(f"\n=== 效率计算器 ===")
    print("你可以测试不同的参数组合:\n")
    
    test_cases = [
        (4, 3, 1),      # 修复后的小矩阵
        (100, 50, 10),  # 中等矩阵
        (512, 256, 16), # 典型Transformer
    ]
    
    for d, k, r in test_cases:
        result = analyze_efficiency(d, k, r)
        status = "有效" if result['is_efficient'] else "无效"
        print(f"矩阵{d}×{k}, rank={r}:")
        print(f"  原始参数: {result['original_params']:,}")
        print(f"  LoRA参数: {result['lora_params']:,}")
        print(f"  参数减少: {result['efficiency_percentage']:.1f}%")
        print(f"  状态: {status}")
        if result['is_efficient']:
            print(f"  压缩比: {result['reduction_ratio']:.1f}:1")
        print()

if __name__ == "__main__":
    comprehensive_analysis()
    interactive_calculator() 