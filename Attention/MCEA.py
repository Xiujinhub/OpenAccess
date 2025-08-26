import torch
import torch.nn as nn

# Residual残差连接模块
class Residual(nn.Module):
    def __init__(self, block):
        super(Residual, self).__init__()
        self.block = block
    def forward(self, x):
        return x + self.block(x)  # 残差连接

def DcovN(c1, c2, depth, kernel_size=3, patch_size=3):
    return nn.Sequential(
        nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size),
        nn.SiLU(),
        nn.GroupNorm(1, c2),
        *[
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(c2, c2, kernel_size, stride=1,
                                  padding=1, groups=c2),
                        nn.SiLU(),
                        nn.GroupNorm(1, c2)
                    )
                ),
                nn.Conv2d(c2, c2, 1, stride=1, padding=0, groups=1),
                nn.SiLU(),
                nn.GroupNorm(1, c2)
            ) for _ in range(depth)
        ]
    )



class MCEA(nn.Module):
    def __init__(self, c1, depth=1, kernel_size=3, patch_size=[3, 5, 7],
                 reduction=16):
        super(MCEA, self).__init__()
        c2 = c1
        self.patch_size = patch_size
        self.branches = nn.ModuleList([
            DcovN(c1, c2, depth, kernel_size, ps) for ps in patch_size
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # 处理各分支输出
        branch_outputs = [branch(x) for branch in self.branches]
        y_list = [self.avg_pool(y).view(b, c) for y in branch_outputs]

        # 添加原始特征
        y_origin = self.avg_pool(x).view(b, c)
        y_list.append(y_origin)

        # 特征融合
        y = sum(y_list) / len(y_list)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)
