import torch
import torch.nn as nn
# Multi-scal Chanel Enhancement Attention
class MCEA(nn.Module):
    def __init__(self, c1, reduction=32, scales=[3, 5, 7], d=2):
        """
        多尺度增强通道注意力模块
        :param c1: 输入输出通道数
        :param reduction: 注意力压缩比 (默认32)
        :param scales: 多尺度卷积核大小列表 (默认[3,5,7])
        :param d: 通道减少因子 (默认2)
        """
        super().__init__()
        self.c1 = c1
        self.scales = scales
        self.d = d
        self.c_prime = max(c1 // d, 1)   
        # 确保中间通道数至少为1
        mid_channels = max(c1 // reduction, 1)
        # 多尺度深度可分离卷积分支
        self.conv_branches = nn.ModuleList()
        for k in scales:
            conv = nn.Sequential(
                nn.Conv2d(c1, c1, kernel_size=k, padding=k // 2, groups=c1, bias=False),   
                nn.Conv2d(c1, self.c_prime, kernel_size=1, bias=False)   
            )
            self.conv_branches.append(conv)
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 特征压缩层: 将拼接后的向量压缩回c1通道
        self.compress = nn.Conv2d(len(scales) * self.c_prime, c1, kernel_size=1, bias=False)
        # 通道注意力机制 - 修复中间通道数
        self.channel_attention = nn.Sequential(
            nn.Conv2d(c1, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, c1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        identity = x
        # 多尺度深度可分离卷积
        branch_outputs = []
        for conv in self.conv_branches:
            y = conv(x)  
            branch_outputs.append(y)
        # 全局平均池化
        gap_features = [self.gap(y) for y in branch_outputs]  
        # 拼接多尺度特征向量
        fused = torch.cat(gap_features, dim=1)  
        # 特征压缩
        compressed = self.compress(fused)  
        # 通道注意力
        attention = self.channel_attention(compressed)  
        # 广播注意力权重并应用
        return identity * attention
