import torch
from model.base_layers import DoubleCNA3x3, CNA3x3, MLP
from model.attentions import CrossAttention
import torch.nn.functional as F

#######3
class Hint(torch.nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch == None:
            mid_ch = out_ch

        avgpool_k = 3
        self.avgpool_padding = (avgpool_k - 1) // 2
        self.avgpool = torch.nn.AvgPool2d(avgpool_k, 1, padding=self.avgpool_padding)

        maxpool_k = 3
        self.maxpool_padding = (maxpool_k - 1) // 2
        self.maxpool = torch.nn.MaxPool2d(maxpool_k, 1, padding=self.maxpool_padding)

        self.x_proj = CNA3x3(in_ch, mid_ch)
        self.hint_proj = DoubleCNA3x3(in_ch * 2, mid_ch)
        self.cross_attn = CrossAttention(mid_ch, mid_ch)
        self.mlp_fnorm = torch.nn.BatchNorm2d(mid_ch)
        self.mlp = MLP(mid_ch, mid_ch * 2, out_ch)

    def forward(self, x):
        avgpooling = self.avgpool(x.detach())
        maxpooling = self.maxpool(x.detach())
        hint = self.hint_proj(torch.cat([x - avgpooling, x - maxpooling], dim=1))#计算
        x_d = self.x_proj(x)#整合一下
        attn = self.cross_attn(x_d, hint)
        x_d = self.mlp_fnorm(x_d + attn)#正则化
        mlp_res = self.mlp(x_d)#多层感知机
        return mlp_res

class SobelConv(torch.nn.Module):#边缘检测
    def __init__(self):
        super(SobelConv, self).__init__()
        # 定义 Sobel 核
        sobel_kernel_x = torch.tensor([[-1, 0, 1],#水平核
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = torch.tensor([[-1, -2, -1],#垂直核
                                       [0, 0, 0],
                                       [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # 拼接 Sobel 核
        self.sobel_kernel = torch.cat([sobel_kernel_x, sobel_kernel_y], dim=0)

    def forward(self, x):
        # 扩展 kernel 以匹配输入的通道数
        sobel_kernel = self.sobel_kernel.repeat(x.size(1), 1, 1, 1).cuda()
        # 进行卷积
        x = F.conv2d(x, sobel_kernel, padding=1, groups=x.size(1))
        return x

class Sobel(torch.nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch ==None:
            mid_ch = out_ch

        self.sobel_conv = SobelConv()

        self.x_proj = CNA3x3(in_ch, mid_ch)
        self.hint_proj = DoubleCNA3x3(in_ch*2, mid_ch)
        self.cross_attn = CrossAttention(mid_ch, mid_ch)
        self.mlp_fnorm = torch.nn.BatchNorm2d(mid_ch)
        self.mlp = MLP(mid_ch, mid_ch*2, out_ch)

    def forward(self, x):
        sobel = self.sobel_conv(x.detach())
        sobel = self.hint_proj(sobel)
        x_d = self.x_proj(x)
        attn = self.cross_attn(x_d, sobel)#跨注意力
        x_d = self.mlp_fnorm(x_d + attn)
        mlp_res = self.mlp(x_d)
        return mlp_res



class IHint(torch.nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch == None:
            mid_ch = out_ch

        avgpool_k = 3
        self.avgpool_padding = (avgpool_k - 1) // 2
        self.avgpool = torch.nn.AvgPool2d(avgpool_k, 1, padding=self.avgpool_padding)

        maxpool_k = 3
        self.maxpool_padding = (maxpool_k - 1) // 2
        self.maxpool = torch.nn.MaxPool2d(maxpool_k, 1, padding=self.maxpool_padding)

        self.x_proj = CNA3x3(in_ch, mid_ch)
        self.hint_proj = DoubleCNA3x3(in_ch * 2, mid_ch)
        self.mlp_fnorm = torch.nn.BatchNorm2d(mid_ch)
        self.mlp = MLP(mid_ch, mid_ch * 2, out_ch)

    def forward(self, x):
        avgpooling = self.avgpool(x.detach())
        maxpooling = self.maxpool(x.detach())
        hint = self.hint_proj(torch.cat([x - avgpooling, x - maxpooling], dim=1))
        mlp_res = self.mlp(hint)
        return mlp_res
