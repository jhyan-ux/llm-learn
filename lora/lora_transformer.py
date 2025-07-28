"""
LoRA版本的Transformer组件
集成LoRA到MultiHeadAttention中，只对W_q和W_v应用LoRA
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lora_layer import LoRALinear
from transformer.transformer import MultiHeadAttention, EncoderLayer, TransformerEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRAMultiHeadAttention(MultiHeadAttention):
    """
    集成LoRA的多头注意力机制
    只对W_q和W_v应用LoRA，W_k和W_o保持原始Linear层
    """
    
    def __init__(self, d_model, num_heads, lora_rank=16):
        """
        初始化LoRA版本的多头注意力
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数  
            lora_rank: LoRA的秩参数
        """
        # TODO 1: 如何正确初始化父类？
        # 你需要调用父类的__init__，这样会创建原始的W_q, W_k, W_v, W_o
        super().__init__(d_model, num_heads)
        
        # TODO 2: 替换特定的线性层为LoRA版本
        # 保存lora_rank以备后用
        self.lora_rank = lora_rank
        
        # 替换W_q为LoRA版本
        # 思考：原始的W_q已经在父类中创建了，现在需要替换它
        # 你需要用LoRALinear替换self.W_q
        original_W_q = self.W_q  # 保存原来的权重以便复制
        self.W_q = LoRALinear(d_model, d_model, lora_rank)
        
        # TODO 3: 如何将预训练权重复制到LoRA层？
        # 提示：LoRALinear继承了nn.Linear，所以有weight和bias属性
        # 需要将original_W_q的权重复制到新的LoRA层中
        with torch.no_grad():
            self.W_q.weight.copy_(original_W_q.weight)
            if original_W_q.bias is not None:
                self.W_q.bias.copy_(original_W_q.bias)
        
        # TODO 4: 对W_v做同样的替换
        original_W_v = self.W_v
        self.W_v = LoRALinear(d_model, d_model, lora_rank)
        
        with torch.no_grad():
            self.W_v.weight.copy_(original_W_v.weight)
            if original_W_v.bias is not None:
                self.W_v.bias.copy_(original_W_v.bias)
        
        self.W_k.weight.requires_grad = False
        self.W_k.bias.requires_grad = False
        self.W_o.weight.requires_grad = False
        self.W_o.bias.requires_grad = False
        # 注意：W_k和W_o保持原样，不应用LoRA
        
    def forward(self, query, key, value, mask=None):
        return super().forward(query, key, value, mask)
        
    def get_lora_parameters(self):
        """
        获取所有LoRA参数，用于单独保存或优化
        """
        lora_params = []
        # TODO 7: 收集所有LoRA相关的参数
        # 提示：需要收集W_q和W_v的lora_A和lora_B参数
        lora_params.extend([self.W_q.lora_A, self.W_q.lora_B])  # W_q的LoRA参数
        lora_params.extend([self.W_v.lora_A, self.W_v.lora_B])  # W_v的LoRA参数
        return lora_params
        
    def merge_and_save(self, path):
        """
        将LoRA权重合并到原始权重中并保存
        用于推理时的优化
        """
        # TODO 8: 实现权重合并逻辑
        # 提示：对于每个LoRA层，计算W_new = W_original + scaling * B @ A
        W_q_new = self.W_q.weight + self.W_q.lora_B @ self.W_q.lora_A * self.W_q.scaling 
        W_v_new = self.W_v.weight + self.W_v.lora_B @ self.W_v.lora_A * self.W_v.scaling 

        self.W_q.weight.copy_(W_q_new)
        self.W_v.weight.copy_(W_v_new)

        torch.save(self.state_dict(), path)

        

# 测试代码
def test_lora_attention():
    """测试LoRA注意力层"""
    print("测试LoRA多头注意力层...")
    
    try:
        # 参数设置
        d_model = 512
        num_heads = 8
        lora_rank = 16
        batch_size = 2
        seq_len = 10
        
        print(f"参数: d_model={d_model}, num_heads={num_heads}, lora_rank={lora_rank}")
        
        # 1. 创建LoRA注意力层
        lora_attention = LoRAMultiHeadAttention(d_model, num_heads, lora_rank)
        print("✓ LoRA注意力层创建成功")
        
        # 2. 创建测试输入 (模拟token embeddings)
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)  
        value = torch.randn(batch_size, seq_len, d_model)
        print(f"✓ 测试输入创建成功: {query.shape}")
        
        # 3. 前向传播
        output = lora_attention(query, key, value)
        print(f"✓ 前向传播成功，输出形状: {output.shape}")
        
        # 4. 检查输出形状
        expected_shape = (batch_size, seq_len, d_model)
        assert output.shape == expected_shape, f"输出形状错误: {output.shape} vs {expected_shape}"
        print("✓ 输出形状验证通过")
        
        # 5. 验证LoRA参数是否可训练
        lora_params = lora_attention.get_lora_parameters()
        print(f"✓ LoRA参数数量: {len(lora_params)}")
        
        for i, param in enumerate(lora_params):
            print(f"  参数{i+1}: {param.shape}, requires_grad={param.requires_grad}")
            assert param.requires_grad, f"LoRA参数{i+1}不可训练！"
        
        # 6. 验证原始权重被冻结
        assert not lora_attention.W_k.weight.requires_grad, "W_k应该被冻结"
        assert not lora_attention.W_o.weight.requires_grad, "W_o应该被冻结"  
        assert not lora_attention.W_q.weight.requires_grad, "W_q的原始权重应该被冻结"
        assert not lora_attention.W_v.weight.requires_grad, "W_v的原始权重应该被冻结"
        print("✓ 权重冻结验证通过")
        
        # 7. 计算参数效率
        original_params = 4 * d_model * d_model  # 4个线性层
        lora_params_count = 2 * lora_rank * (d_model + d_model)  # 只有W_q和W_v用LoRA
        efficiency = (original_params - lora_params_count) / original_params * 100
        
        print(f"✓ 参数效率分析:")
        print(f"  原始参数: {original_params:,}")
        print(f"  LoRA参数: {lora_params_count:,}") 
        print(f"  参数减少: {efficiency:.1f}%")
        
        print("\n🎉 所有测试通过！LoRA注意力层工作正常！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_lora_attention() 