import torch
import torch.nn as nn

class MCEA(nn.Module):
    def __init__(self, c1, reduction=32, scales=[3, 5, 7], d=2):
        """
        多尺度增强通道注意力模块
        :param c1: 输入输出通道数
        :param reduction: 注意力压缩比 (默认32)
        :param scales: 多尺度卷积核大小列表 (默认[3,5,7])
        :param d: 通道扩展/减少因子 (默认2)
        """
        super().__init__()
        self.c1 = c1
        self.scales = scales
        self.d = d
        self.c_prime = max(c1 // d, 1)
        self.expanded_channels = max(c1 * d, 1)
        # 确保中间通道数至少为1
        mid_channels = max(c1 // reduction, 1)
        # 多尺度分支
        self.conv_branches = nn.ModuleList()
        for k in scales:
            branch = nn.Sequential(
                # 初始扩展卷积块
                nn.Conv2d(c1, self.expanded_channels, kernel_size=1, bias=False),
                nn.SiLU(),
                nn.GroupNorm(num_groups=8, num_channels=self.expanded_channels),
                # 深度卷积
                nn.Conv2d(self.expanded_channels, self.expanded_channels,
                          kernel_size=k, padding=k // 2,
                          groups=self.expanded_channels, bias=False),
                nn.BatchNorm2d(self.expanded_channels),
                nn.SiLU(),
                # 压缩卷积
                nn.Conv2d(self.expanded_channels, self.c_prime, kernel_size=1, bias=False),
                nn.SiLU(),
                nn.GroupNorm(num_groups=8, num_channels=self.c_prime)
            )
            self.conv_branches.append(branch)
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 特征压缩层
        self.compress = nn.Conv2d(len(scales) * self.c_prime, c1, kernel_size=1, bias=False)
        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.Conv2d(c1, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, c1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        identity = x
        # 多尺度特征提取
        branch_outputs = []
        for branch in self.conv_branches:
            # 获取分支的卷积核大小
            k = branch[3].kernel_size[0]
            # 前向传播 (需要手动实现残差连接)
            # 初始扩展卷积块
            expanded = branch[0:3](x)
            # 深度卷积
            dw_out = branch[3:6](expanded)
            #残差连接
            residual_out = dw_out + expanded
            compressed_out = branch[6:](residual_out)
            branch_outputs.append(compressed_out)
        # 全局平均池化
        gap_features = [self.gap(y) for y in branch_outputs]
        # 拼接多尺度特征向量
        fused = torch.cat(gap_features, dim=1)
        # 特征压缩
        compressed = self.compress(fused)
        # 通道注意力
        attention = self.channel_attention(compressed)
        return identity * attention

