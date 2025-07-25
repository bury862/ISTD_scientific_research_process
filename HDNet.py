import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import os
from matplotlib import pyplot as plt
import torch.fft
import numpy as np
from model.MAC_Kernel import GenerateKernels, GenerateKernels3, GenerateKernels4

from model.hint import Hint#要有文件夹名（（

from model.PConv_loss_model.APConv import PConv
from model.fusion import AsymBiChaFuseReduce, BiLocalChaFuseReduce, BiGlobalChaFuseReduce #导入模块

os.environ['CUDA_VISIBLE_DEVICES']="2"

kernels = GenerateKernels()
weights = [
            nn.Parameter(data = torch.FloatTensor(k).unsqueeze(0).unsqueeze(0), requires_grad=False).cuda()#不参与训练
            for ks in kernels for k in ks
        ]
kernels2 = GenerateKernels3()
weights2 = [
            nn.Parameter(data = torch.FloatTensor(k).unsqueeze(0).unsqueeze(0), requires_grad=False).cuda()
            for ks in kernels2 for k in ks
        ]
kernels3 = GenerateKernels4()
weights3 = [
            nn.Parameter(data = torch.FloatTensor(k).unsqueeze(0).unsqueeze(0), requires_grad=False).cuda()
            for ks in kernels3 for k in ks
        ]

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, max(1, in_planes // 16), 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(max(1, in_planes // 16), in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

    #增加频率注意力机制..
class FourierAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        ratio = max(1, in_channels // 16)  # 动态计算 ratio
        self.in_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 频域特征映射到注意力权重
        self.freq_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels),
            nn.Sigmoid()
        )
        
        # 可学习的频域增强参数
        self.energy = nn.Parameter(torch.tensor(0.5))  # 初始保留50%低频
        self.energy.data.clamp_(0.01, 0.99)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 傅里叶变换
        f = torch.fft.fft2(x)
        fshift = torch.fft.fftshift(f)
        magnitude = torch.abs(fshift)  # 幅度谱
        phase = torch.angle(fshift)    # 相位谱
        
        # 2. 生成频域注意力权重
        avg_magnitude = self.avg_pool(magnitude).view(B, C)
        max_magnitude = self.max_pool(magnitude).view(B, C)
        attention_weights = self.freq_mlp(avg_magnitude + max_magnitude)  # 通道注意力
        
        # 3. 应用注意力权重（增强高频区域）
        enhanced_magnitude = magnitude * attention_weights.view(B, C, 1, 1)
        
        # 4. 保留部分低频（通过energy参数控制）
        cutoff = int(self.energy.item() * min(H, W) // 2)
        enhanced_magnitude = self._apply_energy_mask(enhanced_magnitude, cutoff)
        
        # 5. 逆变换回空域
        enhanced_magnitude = enhanced_magnitude.to(torch.complex64) # 确保数据类型正确
        enhanced_fshift = enhanced_magnitude * torch.exp(1j * phase)
        ishift = torch.fft.ifftshift(enhanced_fshift)
        enhanced_x = torch.fft.ifft2(ishift).real  # 取实部作为输出
        
        return enhanced_x

    def _apply_energy_mask(self, magnitude, cutoff):
        # 生成低频掩码（中心区域保留）
        B, C, H, W = magnitude.shape
        mask = torch.zeros((H, W), device=magnitude.device)
        crow, ccol = H // 2, W // 2
        mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)# 扩展 mask 到 [B, C, H, W]
        # 保留低频区域，增强高频区域
        return magnitude * (1 - mask) + magnitude * mask * 0.5  # 低频衰减，高频增强
    

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        #添加频率注意力机制
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )
        #self.freq_attention = FourierAttention(out_channels)  # 新增频域注意力.

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)

        #freq_mask = self.freq_attention(x)#频率注意力权重

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        #out = self.freq_attention(out)  # 应用频率注意力
        out = self.ca(out) * out
        out = self.sa(out) * out
        #out = out * freq_mask#应用频率注意力权重

        out += residual
        out = self.relu(out)

        # out = out + residual
        return out

class PConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.pconv1 = PConv(in_channels, out_channels, k=4, s=stride)#更改k:3-->4
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pconv2 = PConv(out_channels, out_channels, k=3, s=1)#更改k
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        #print("here!")
        out = self.pconv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pconv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


class MAC(nn.Module):#输入通道，输出通道，不同卷积核的尺寸，分组数
    def __init__(self, inplanes, outplanes, one, two, three, scales = 4):
        super(MAC, self).__init__()
        if outplanes % scales != 0: 
            raise ValueError('Planes must be divisible by scales')
        self.weights = weights[:]#使用MAC_Kernel.py中生成的卷积核参数(在同一个模块下，全局变量可以直接调用)
        self.weights2 = weights2
        self.weights3 = weights3
        self.scales = scales 
        self.relu = nn.ReLU(inplace = True)
        self.spx = outplanes // scales#按通道四个过程分组后   再拼接
        self.inconv = nn.Sequential(   #初始化卷积一下
            nn.Conv2d(inplanes, outplanes, 1, 1, 0),
            nn.BatchNorm2d(outplanes)
        )
        self.conv1 = nn.Sequential(#普通卷积   输入通道==输出通道，卷积核大小
            nn.Conv2d(self.spx, self.spx, one, 1, one // 2, groups = self.spx),
            nn.BatchNorm2d(self.spx),
        )
        self.conv1[0].weight.data = self.weights[one // 2 - 1].repeat(self.spx, 1, 1, 1)#通过尺寸选取卷积核，[one // 2 - 1]

        self.conv2 = nn.Sequential(#空洞卷积，dilation=2
            nn.Conv2d(self.spx, self.spx, two, 1, 2, groups = self.spx, dilation=2),
            nn.BatchNorm2d(self.spx),
        )
        self.conv2[0].weight.data = self.weights[two // 2 - 1].repeat(self.spx, 1, 1, 1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.spx, self.spx, three, 1, 1, groups = self.spx),
        )
        self.conv3[0].weight.data = self.weights2[0].repeat(self.spx, 1, 1, 1)

        self.conv4 = nn.Sequential(
            nn.Conv2d(self.spx, self.spx, three, 1, 2, groups = self.spx, dilation=2),
        )
        self.conv4[0].weight.data = self.weights3[0].repeat(self.spx, 1, 1, 1)
        
        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(self.spx)
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, 3, 1, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.ca = ChannelAttention(outplanes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.inconv(x)
        inputt = x
        xs = torch.chunk(x, self.scales, 1)#分组
        ys = []
        ys.append(xs[0])#第一组直连
        ys.append(self.relu(self.conv1(xs[1])))
        ys.append(self.relu(self.conv2(xs[2] + ys[1])))
        temp = xs[3] + ys[2]
        temp1 = self.conv5(self.conv3(temp) + self.conv4(temp))
        ys.append(self.relu(temp1))
        y = torch.cat(ys, 1)

        y = self.outconv(y)#融合一下

        output = self.relu(y + inputt)
        return output

class DHPF(nn.Module):
    def __init__(self, energy):
        super(DHPF, self).__init__()
        self.energy = energy
    
    """def __init__(self, initial_energy=0.5):#初始值
        super(DHPF, self).__init__()
        self.energy = nn.Parameter(torch.tensor(initial_energy))#可学习的energy
        self.energy.data.clamp_(0.01,0.99)#限制在合理范围内"""

        #self.freq_attention = FourierAttention(1)  # 加入单通道频域注意力。注意函数参数匹配-->0
    
    def _determine_cutoff_frequency(self, f_transform, target_ratio):
        total_energy = self._calculate_total_energy(f_transform)
        target_low_freq_energy = total_energy * target_ratio  #用能量对比计算

        for cutoff_frequency in range(1, min(f_transform.shape[0], f_transform.shape[1]) // 2):
            low_freq_energy = self._calculate_low_freq_energy(f_transform, cutoff_frequency)
            if low_freq_energy >= target_low_freq_energy:
                return cutoff_frequency
        return 5 
    
    def _calculate_total_energy(self, f_transform):
        magnitude_spectrum = torch.abs(f_transform)
        total_energy = torch.sum(magnitude_spectrum ** 2)
        return total_energy
    
    def _calculate_low_freq_energy(self, f_transform, cutoff_frequency):
        magnitude_spectrum = torch.abs(f_transform)
        height, width = magnitude_spectrum.shape

        low_freq_energy = torch.sum(magnitude_spectrum[
            height // 2 - cutoff_frequency:height // 2 + cutoff_frequency,
            width // 2 - cutoff_frequency:width // 2 + cutoff_frequency
        ] ** 2)
    
        return low_freq_energy

    def forward(self, x):
        B, C, H, W = x.shape
        f = torch.fft.fft2(x)# 傅里叶变换，得到频谱（复数）
        fshift = torch.fft.fftshift(f)# 将低频分量移到频谱中心
        crow, ccol = H // 2, W // 2 # 计算频谱中心坐标

        for i in range(B):

            #dynamic_energy = torch.sigmoid(self.energy) * 0.9 + 0.05  # 保持在0.05-0.95之间

            # 动态确定截止频率，使低频能量达到设定比例
            cutoff_frequency = self._determine_cutoff_frequency(fshift[i, 0], self.energy)
            #cutoff_frequency = self._determine_cutoff_frequency(fshift[i, 0], dynamic_energy.item())

            #应用频率注意力"""
            """magnitude = torch.abs(fshift[i, 0])
            phase = torch.angle(fshift)
            attention_weights = self.freq_attention(x)
            enhanced_magnitude = magnitude * attention_weights
            fshift = enhanced_magnitude * torch.exp(1j * phase)"""

            # 将中心区域（低频）置零，实现高通滤波
            fshift[i, :, crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 0
            #magnitude[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 0# 仅对幅度进行操作（避免复数到实数的隐式转换）
            #fshift[i, 0] = magnitude * torch.exp(1j * phase)# 重构复数频谱

        ishift = torch.fft.ifftshift(fshift)# 频谱逆移回原位置..
        ideal_high_pass = torch.abs(torch.fft.ifft2(ishift)) # 逆傅里叶变换回图像域，取模长
        return ideal_high_pass 

class HDNet(nn.Module):
    def __init__(self, input_channels, block=ResNet):#block=ResNet-->PConvBlock
        super(HDNet, self).__init__()
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]
        energy = [0.1, 0.2, 0.4, 0.8]#化为初始值

        #增加
        """
        self.CNA3x3 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(16), nn.ReLU())                 #通道数统一为16，可以试试调节
        self.DoubleCNA3x3 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1),  #in_channels=32拼接
                                    nn.BatchNorm2d(16), nn.ReLU(),
                                          nn.Conv2d(16, 16, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(16), nn.ReLU()
                                          )
        self.hint = Hint(16, 16, 16 * 4)
        self.x_proj = self.CNA3x3#CNA3x3：卷积（3*3）、归一、激活
        self.hintx_proj = self.DoubleCNA3x3"""

        #添加ACM初始化（传入通道数）,可更改模块
        self.acm3 = BiGlobalChaFuseReduce(param_channels[4], param_channels[3], out_channels=param_channels[3])
        self.acm2 = BiGlobalChaFuseReduce(param_channels[3], param_channels[2], out_channels=param_channels[2])
        self.acm1 = BiGlobalChaFuseReduce(param_channels[2], param_channels[1], out_channels=param_channels[1])
        self.acm0 = BiGlobalChaFuseReduce(param_channels[1], param_channels[0], out_channels=param_channels[0])
        ##################################

        self.pool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)
        self.py_init = self._make_layer2(input_channels, 1, block=ResNet)#掩码的block不能用PConvBlock,输出通道数不合理..
        #self.py_init = self._make_layer2(16, 1, block)#输入通道数变了

        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block)
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block, param_blocks[0])
        self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block, param_blocks[1])
        self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block, param_blocks[2])

        self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block, param_blocks[3])

        #解码器的通道数也变了
        #self.decoder_3 = self._make_layer2(param_channels[3]+param_channels[4], param_channels[3], block, param_blocks[2])
        #self.decoder_2 = self._make_layer2(param_channels[2]+param_channels[3], param_channels[2], block, param_blocks[1])
        #self.decoder_1 = self._make_layer2(param_channels[1]+param_channels[2], param_channels[1], block, param_blocks[0])
        #self.decoder_0 = self._make_layer2(param_channels[0]+param_channels[1], param_channels[0], block)
        self.decoder_3 = self._make_layer2(param_channels[3] , param_channels[3], block, param_blocks[2])
        self.decoder_2 = self._make_layer2(param_channels[2] , param_channels[2], block, param_blocks[1])
        self.decoder_1 = self._make_layer2(param_channels[1] , param_channels[1], block, param_blocks[0])
        self.decoder_0 = self._make_layer2(param_channels[0] , param_channels[0], block)

        self.py3 = DHPF(energy[3])
        self.py2 = DHPF(energy[2])
        self.py1 = DHPF(energy[1])
        self.py0 = DHPF(energy[0])
        #self.py3 = DHPF(initial_energy=0.8)
        #self.py2 = DHPF(initial_energy=0.4)
        #self.py1 = DHPF(initial_energy=0.2)
        #self.py0 = DHPF(initial_energy=0.1)

        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

        self.final = nn.Conv2d(4, 1, 3, 1, 1)


    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []        
        layer.append(MAC(in_channels, out_channels, 3, 3, 3))#MAC卷积（创新加入）.
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels))#残差块
        return nn.Sequential(*layer)
    
    def _make_layer2(self, in_channels, out_channels, block, block_num = 1):
        layer= []
        layer.append(block(in_channels, out_channels))#残差块
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)

    #添加参数截断方法
    def clamp_energy_parameters(self):
        """确保energy参数在[0.01, 0.99]范围内"""
        for module in [self.py0, self.py1, self.py2, self.py3]:
            if hasattr(module, 'energy'):
                module.energy.data.clamp_(0.01, 0.99)

    def forward(self, x, warm_flag):
        #x = self.conv_init(x)#初始化单独放前面
        """
        hint = self.hint(x)  # 预处理一下
        x_p = self.x_proj(x)  # 简单初始化一下
        x = torch.cat([x_p, hint], dim=1)  # 拼接起来
        x = self.hintx_proj(x)  # 融合拼接数据    通道16"""

        #以x输入到HDNet
        #x_e0 = self.encoder_0(x)
        x_e0 = self.encoder_0(self.conv_init(x)) #编码器,初始化移到开始
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        x_m = self.middle_layer(self.pool(x_e3))
        
        #x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], 1))#解码器
        #x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], 1))
        #x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], 1))
        #x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], 1))
        x_d3 = self.decoder_3(self.acm3(self.up(x_m), x_e3))#替换解码器，特征融合部分：cat-->acm
        x_d2 = self.decoder_2(self.acm2(self.up(x_d3), x_e2))
        x_d1 = self.decoder_1(self.acm1(self.up(x_d2), x_e1))
        x_d0 = self.decoder_0(self.acm0(self.up(x_d1), x_e0))

        mask0 = self.output_0(x_d0)#输出层
        mask1 = self.output_1(x_d1)
        mask2 = self.output_2(x_d2)
        mask3 = self.output_3(x_d3)
        
        if warm_flag: #后阶段复杂处理--滤波器

            x_py_init = self.py_init(x)
            x_py_v3 = x_py_init * self.sigmoid(self.up_8(mask3)) + x_py_init #上采样、联级、激活函数、乘法
            x_py_v3 = self.py3(x_py_v3)#滤波

            x_py_v2 = x_py_v3 * self.sigmoid(self.up_4(mask2)) + x_py_v3 
            x_py_v2 = self.py2(x_py_v2)

            x_py_v1 = x_py_v2 * self.sigmoid(self.up(mask1)) + x_py_v2 
            x_py_v1 = self.py1(x_py_v1)

            x_py_v0 = x_py_v1 * self.sigmoid(mask0) + x_py_v1 
            x_py_v0 = self.sigmoid(self.py0(x_py_v0))
            #x_py_v0 = self.sigmoid(x_py_v0)

            output = self.final(torch.cat([mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)], dim=1))
            output = output * x_py_v0 + output
            return [mask0, mask1, mask2, mask3], output
    
        else:
            output = self.output_0(x_d0)
            output = output
            return [], output
