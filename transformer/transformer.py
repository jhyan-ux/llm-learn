import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # TODO: 你需要在这里定义什么参数？
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads # 多头中每一个头的维度
        
        # 注意：这些线性层要为所有头同时生成Q、K、V
        self.W_q = nn.Linear(d_model, d_model)  # 输出所有头的Query
        self.W_k = nn.Linear(d_model, d_model)  # 输出所有头的Key
        self.W_v = nn.Linear(d_model, d_model)  # 输出所有头的Value
        self.W_o = nn.Linear(d_model, d_model)  # 合并所有头的输出

    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 1. 生成Q、K、V矩阵
        Q = self.W_q(query)  # (batch_size, seq_len, d_model)
        K = self.W_k(key)    # (batch_size, seq_len, d_model)
        V = self.W_v(value)  # (batch_size, seq_len, d_model)
        
        # 2. Reshape为多头格式并转置
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        
        # 3. 计算注意力分数
        score = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(score, dim=-1)
        attention_output = attention_weights @ V # (batch_size, num_heads, seq_len, d_k)
        
        # 4. 合并多头结果
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_k * self.num_heads)
        output = self.W_o(attention_output)
        
        return output

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) # (max_seq_length, 1)
        i = torch.arange(0, d_model, 2, dtype=torch.float) # (d_model // 2)
        
        # 修正公式：注意括号和顺序
        div_term = torch.exp(i * (-math.log(10000.0) / d_model))
        
        # 偶数位置用sin，奇数位置用cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置 [0,2,4,6,...]
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置 [1,3,5,7,...]
        
        # 注册为buffer，跟随模型移动
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # pe: (max_seq_len, d_model)
        # 取出需要的长度并添加到输入上
        seq_len = x.size(1)
        return x + self.pe[:seq_len]  # 广播：(batch,seq,d_model) + (seq,d_model)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # d_model -> d_ff
        self.linear2 = nn.Linear(d_ff, d_model)  # d_ff -> d_model

    
    def forward(self, x):
        # x -> Linear1 -> ReLU -> Linear2
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-LayerNorm 版本（推荐，训练更稳定）
        # Self-Attention子层
        norm_x = self.layer_norm1(x)
        attention_output = self.mha(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(attention_output)  # 残差连接 + Dropout
        
        # Feed-Forward子层  
        norm_x = self.layer_norm2(x)
        ff_output = self.ff(norm_x)
        x = x + self.dropout(ff_output)  # 残差连接 + Dropout
        
        return x
        
        # 原始Post-LN版本（你的实现，也完全正确）：
        # attention_output = self.mha(x, x, x, mask)
        # x = self.layer_norm1(x + attention_output)
        # ff_output = self.ff(x)
        # return self.layer_norm2(ff_output + x)

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        
        # 4个核心组件
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # x: (batch_size, seq_len) - Token IDs
        
        # 1. Token IDs -> Dense Embeddings
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        
        # 2. Add Position Encoding  
        x = self.pos_encoding(x)  # (batch_size, seq_len, d_model)
        
        # 3. Pass through all Encoder layers
        for layer in self.layers:
            x = layer(x, mask)  # (batch_size, seq_len, d_model)
        
        # 4. Final LayerNorm (for Pre-LN architecture)
        x = self.norm(x)  # (batch_size, seq_len, d_model)
        
        return x

# 测试用的简单示例
if __name__ == "__main__":
    # 创建一个简单的测试
    vocab_size = 1000
    seq_length = 10
    batch_size = 2
    d_model = 512
    num_heads = 8
    
    # 测试MultiHeadAttention
    print("🧪 测试MultiHeadAttention...")
    mha = MultiHeadAttention(d_model, num_heads)
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_length, d_model)
    
    # Self-attention (Q=K=V=x)
    output = mha(x, x, x)
    
    print(f"✅ 输入形状: {x.shape}")
    print(f"✅ 输出形状: {output.shape}")
    print(f"✅ 形状匹配: {output.shape == x.shape}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"✅ 参数总数: {total_params:,}")
    
    print("🎉 MultiHeadAttention测试通过！")
    
    print("\n" + "="*50)
    print("🧪 测试PositionEncoding...")
    pe_layer = PositionEncoding(d_model)
    
    # 测试不同长度的序列
    for test_seq_len in [5, 10]:
        test_input = torch.randn(batch_size, test_seq_len, d_model)
        pe_output = pe_layer(test_input)
        
        print(f"✅ 序列长度 {test_seq_len}: 输入{test_input.shape} -> 输出{pe_output.shape}")
        print(f"   形状匹配: {pe_output.shape == test_input.shape}")
        
        # 检查是否确实添加了位置信息（输出应该不等于输入）
        is_different = not torch.allclose(pe_output, test_input)
        print(f"   位置编码生效: {is_different}")
    
        print("🎉 PositionEncoding测试通过！")
    
    print("\n" + "="*50)
    print("🧪 测试FeedForward...")
    d_ff = d_model * 4  # 4倍关系
    ff_layer = FeedForward(d_model, d_ff)
    
    test_input = torch.randn(batch_size, seq_length, d_model)
    ff_output = ff_layer(test_input)
    
    print(f"✅ 输入形状: {test_input.shape}")  
    print(f"✅ 输出形状: {ff_output.shape}")
    print(f"✅ 形状匹配: {ff_output.shape == test_input.shape}")
    print(f"✅ d_ff = {d_ff} (d_model × 4)")
    
    # 检查参数数量
    ff_params = sum(p.numel() for p in ff_layer.parameters())
    print(f"✅ FeedForward参数数: {ff_params:,}")
    
    print("🎉 FeedForward测试通过！")
    
    print("\n" + "="*50)
    print("🧪 测试EncoderLayer...")
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout=0.1)
    
    test_input = torch.randn(batch_size, seq_length, d_model)
    layer_output = encoder_layer(test_input)
    
    print(f"✅ 输入形状: {test_input.shape}")
    print(f"✅ 输出形状: {layer_output.shape}")
    print(f"✅ 形状匹配: {layer_output.shape == test_input.shape}")
    
    # 计算参数数量
    layer_params = sum(p.numel() for p in encoder_layer.parameters())
    print(f"✅ EncoderLayer参数数: {layer_params:,}")
    
    print("🎉 EncoderLayer测试通过！")
    
    print("\n" + "="*50)
    print("🧪 测试完整的TransformerEncoder...")
    
    # 完整的Transformer参数
    num_layers = 6  # 原始Transformer使用6层
    
    transformer = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model, 
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=0.1
    )
    
    # 创建token ID输入（而不是embedding）
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    print(f"✅ Token IDs输入形状: {token_ids.shape}")
    print(f"✅ Token IDs范围: [0, {vocab_size-1}]")
    
    # 完整前向传播
    final_output = transformer(token_ids)
    
    print(f"✅ 最终输出形状: {final_output.shape}")
    print(f"✅ 预期输出形状: {(batch_size, seq_length, d_model)}")
    print(f"✅ 形状匹配: {final_output.shape == (batch_size, seq_length, d_model)}")
    
    # 计算总参数数量
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"✅ 总参数数量: {total_params:,}")
    print(f"✅ 模型层数: {num_layers}")
    
    print("\n🎉🎉🎉 完整的Transformer Encoder实现成功！🎉🎉🎉")
    print("🌟 恭喜！你已经从零实现了完整的Transformer架构！")
    
    # 架构总结
    print(f"\n📋 架构总结:")
    print(f"   📚 词汇表大小: {vocab_size:,}")
    print(f"   🔢 模型维度: {d_model}")
    print(f"   🧠 注意力头数: {num_heads}")
    print(f"   ⚡ 前馈维度: {d_ff}")
    print(f"   🏗️  编码器层数: {num_layers}")
    print(f"   💫 总参数量: {total_params:,}")