import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Linear):
    """
    LoRA (Low-Rank Adaptation) Linear层
    
    实现公式: h = W_0 * x + B * (A * x)
    其中:
    - W_0: 冻结的原始权重 (继承自nn.Linear)
    - A: 低秩矩阵A, shape=(in_featuers, rank)  
    - B: 低秩矩阵B, shape=(rank, out_features)
    - rank: 低秩维度，通常 << min(in_features, out_features)
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        """
        初始化LoRA Linear层
        """
        # 继承父类
        super().__init__(in_features, out_features, bias)
        
        # TODO 1: 保存低秩维度
        self.rank = rank  # 你来填写正确的值
        
        # TODO 2: 冻结原始权重W_0 
        self.weight.requires_grad = False  # 你来判断：True还是False？
        if self.bias is not None:
            self.bias.requires_grad = False  # 同样需要你判断
            
        # TODO 3: 创建LoRA矩阵A和B
        # 关键问题：
        # - A的形状应该是什么？ (rank, in_features) 还是 (in_features, rank)?
        # - B的形状应该是什么？ 
        # - 如何初始化才能让开始时 BA = 0？
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))  # 你来填写正确的形状和初始化
        # A矩阵用什么分布初始化？
        
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))  # 你来填写正确的形状和初始化  
        # B矩阵用什么值初始化？
        
        # 缩放因子
        self.scaling = 1.0 / rank
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        关键思考：
        1. 如何计算原始输出？
        2. 如何高效计算 B(Ax) 而不是 (BA)x？
        3. 最终如何组合两个输出？
        """
        # TODO 4: 计算原始linear层输出
        original_output = F.linear(x, self.weight, self.bias)
        
        # TODO 5: 计算LoRA路径 - 注意计算顺序！
        # 第一步：计算什么？
        lora_output = F.linear(x, self.lora_A, None)
        
        # 第二步：计算什么？  
        lora_output = F.linear(lora_output, self.lora_B, None)  # 你来实现第二步
        
        # TODO 6: 返回最终结果
        return original_output + lora_output * self.scaling  # 临时返回，你来修改
        
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}'


def test_lora_layer():
    """测试函数 - 你也需要完善这个！"""
    print("开始测试LoRA层...")
    
    # 创建一个简单的测试用例
    try:
        lora_layer = LoRALinear(in_features=4, out_features=3, rank=2)
        print(f"✓ LoRA层创建成功: {lora_layer}")
        
        # 测试输入
        x = torch.randn(2, 4)  # batch_size=2, in_features=4
        print(f"输入形状: {x.shape}")
        
        # 前向传播
        output = lora_layer(x)
        print(f"输出形状: {output.shape}")
        print("✓ 前向传播成功!")
        
        # 检查参数冻结
        print(f"原始权重是否冻结: {not lora_layer.weight.requires_grad}")
        print(f"LoRA_A是否可训练: {lora_layer.lora_A.requires_grad}")
        print(f"LoRA_B是否可训练: {lora_layer.lora_B.requires_grad}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("让我们分析一下问题...")
        
        # 维度分析
        print(f"\n维度分析:")
        print(f"输入 x: {x.shape if 'x' in locals() else 'undefined'}")
        if 'lora_layer' in locals():
            print(f"lora_A形状: {lora_layer.lora_A.shape}")
            print(f"lora_B形状: {lora_layer.lora_B.shape}")


if __name__ == "__main__":
    test_lora_layer() 