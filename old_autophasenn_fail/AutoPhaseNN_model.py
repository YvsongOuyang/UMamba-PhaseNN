import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
from tqdm.notebook import tqdm
import numpy as np
from numpy.fft import fftn, fftshift
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--shape', type=int, default=64)  # 输入维度64x64x64
parser.add_argument('--T', type=float, default=0.1)       # 时间维度（假设）
parser.add_argument('--nconv', type=int, default=32)   # 初始卷积数（对应TF的32）
parser.add_argument('--use_down_stride', type=bool, default=False)  # 按你的实际配置
parser.add_argument('--use_up_stride', type=bool, default=False)    # 按你的实际配置
parser.add_argument('--n_blocks', type=int, default=4)  # Encoder 4个block（对应TF 4个MaxPool）
parser.add_argument('--unsupervise', type=bool, default=True) 
parser.add_argument('--scale_I', type=int, default=1) 
argv = parser.parse_args([])  # 空列表表示不读取命令行参数


import torch
import torch.nn as nn
import torch.fft
import numpy as np

# --- 对应 TF 索引 89: phi (Lambda) ---
class PhiLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # 注册 PI 为常量
        self.register_buffer('pi', torch.tensor(np.pi))

    def forward(self, x):
        # 假设输入 x 已经是 decoder2 的输出 (Tanh)
        return x * self.pi

# --- 对应 TF 索引 91: support (Lambda) ---
class SupportLayer(nn.Module):
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

    def forward(self, amp):
        # 生成 0/1 Mask
        return torch.where(
            amp >= self.threshold,
            torch.ones_like(amp),
            torch.zeros_like(amp)
        )

# --- 对应 TF 索引 90: Obj (Lambda) ---
class ObjLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, amp, phi):
        # 合成复数场: Amp * e^(j*Phi)
        return torch.polar(amp, phi)

# --- 对应 TF 索引 92: masked_obj (Lambda) ---
class MaskedObjLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, obj, support):
        # 复数 Obj * 实数 Support
        # 需要把 support 转为 complex 才能相乘 (或者利用广播)
        return obj * support.type_as(obj)

# --- 对应 TF 索引 93: farfield_diff (Lambda) ---
class FarfieldDiffLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, masked_obj):
        # 对应 TF: ifftshift -> fft3d -> fftshift -> abs
        # 针对 PT 格式 (B, C, D, H, W)，对最后三维操作
        spatial_dims = (-3, -2, -1)
        
        x = torch.fft.ifftshift(masked_obj, dim=spatial_dims)
        x = torch.fft.fftn(x, dim=spatial_dims, norm=None)
        x = torch.fft.fftshift(x, dim=spatial_dims)
        
        intensity = torch.abs(x).to(torch.float32)
        return intensity



class Network(nn.Module):
    def __init__(self, argv):
        super(Network, self).__init__()
        self.argv = argv
        self.H, self.W = argv.shape, argv.shape
        self.T = argv.T
        self.nconv = argv.nconv
        self.use_down_stride = argv.use_down_stride
        self.use_up_stride = argv.use_up_stride
        self.n_blocks = argv.n_blocks

        def get_down_blocks():
            down_blocks_all = []
            n_filt_in = 1
            for block_indx in range(self.n_blocks):
                n_filt_out = self.nconv * 2 ** block_indx
                block = self.down_block(
                    n_filt_in, n_filt_out, self.use_up_stride)
                down_blocks_all += block
                n_filt_in = n_filt_out
                n_filt_out = n_filt_out*2
            down_blocks_all += self.down_block(n_filt_in, n_filt_out, not self.use_down_stride)
            return down_blocks_all

        self.encoder = nn.Sequential(*get_down_blocks())
                # Decoder 1 (Amp) 结构: 256 -> 128 -> 64 -> 32
        self.decoder1 = nn.Sequential(
            *self.up_block(512, 256, self.use_up_stride), # fnum*8
            *self.up_block(256, 128, self.use_up_stride), # fnum*4
            *self.up_block(128, 64, self.use_up_stride),  # fnum*2
            *self.up_block(64, 32, not self.use_up_stride),  # fnum*2
            nn.Conv3d(32, 1, 3, 1, padding=1),
            nn.Sigmoid()
        )

        # Decoder 2 (Phase) 结构: 128 -> 128 -> 64 -> 32
        self.decoder2 = nn.Sequential(
            *self.up_block(512, 128, self.use_up_stride), # fnum*4 <--- 不对称点！
            *self.up_block(128, 128, self.use_up_stride), # fnum*4
            *self.up_block(128, 64, self.use_up_stride),  # fnum*2
            *self.up_block(64, 32, not self.use_up_stride),  # fnum*2
            nn.Conv3d(32, 1, 3, 1, padding=1),
            nn.Tanh() # 注意: 原PT代码这里没乘PI，forward里乘了
        )

        self.phi_layer = PhiLayer()                 # Index 89
        self.obj_layer = ObjLayer()                 # Index 90
        self.support_layer = SupportLayer(0.1)# Index 91
        self.masked_obj_layer = MaskedObjLayer()    # Index 92
        self.farfield_layer = FarfieldDiffLayer()   # Index 93


    def down_block(self, filters_in, filters_out, use_down_stride=False):
        block = [
            nn.Conv3d(in_channels=filters_in, out_channels=filters_out,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(filters_out,  momentum=0.01, eps=1e-3)]
        if use_down_stride: 
            block += [
                nn.Conv3d(filters_out, filters_out, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.BatchNorm3d(filters_out,  momentum=0.01, eps=1e-3)]
        else:
            block += [
                nn.Conv3d(filters_out, filters_out, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.BatchNorm3d(filters_out, momentum=0.01, eps=1e-3),
                nn.MaxPool3d((2, 2, 2))]
        return block

    def up_block(self, filters_in, filters_out, use_down_stride=False):
        if not use_down_stride:
            block = [
                nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),
                nn.Conv3d(filters_in, filters_out, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.BatchNorm3d(filters_out, momentum=0.01, eps=1e-3),
                nn.Conv3d(filters_out, filters_out, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.BatchNorm3d(filters_out, momentum=0.01, eps=1e-3)
            ]
        else:
            block = [
                nn.ZeroPad3d(padding=16),
                nn.Conv3d(filters_in, filters_out, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm3d(filters_out,  momentum=0.01, eps=1e-3),
                nn.Conv3d(filters_out, filters_out, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm3d(filters_out,  momentum=0.01, eps=1e-3),
            ]

        return block
    

    def get_mask(self, input_tensor: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        PyTorch 版掩码生成函数，完全对应 TF 的 get_mask 功能
        :param input_tensor: 输入张量（对应 TF 的 input）
        :param threshold: 阈值（对应 TF 的 args.T，显式传参更符合 PyTorch 风格）
        :return: 0/1 掩码张量（与输入同形状、同 dtype、同设备）
        """
        # 核心逻辑：条件判断 → 生成掩码（对应 TF 的 tf.where + ones_like/zeros_like）
        mask = torch.where(
            condition=input_tensor >= threshold,  # 与 TF 条件 input >= args.T 完全一致
            input=torch.ones_like(input_tensor),      # 满足条件填 1（对应 tf.ones_like）
            other=torch.zeros_like(input_tensor)      # 不满足条件填 0（对应 tf.zeros_like）
        )
        return mask
    

    def combine_complex(self, amp: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        :param amp: 幅度张量（实数类型，如 float32/float64）
        :param phi: 相位张量（实数类型，与 amp 同形状、同设备）
        :return: 合成的复数张量（dtype=torch.complex64，与输入同形状、同设备）
        """
        # 1. 将实数张量 amp/phi 转换为复数类型（对应 TF 的 tf.cast(..., tf.complex64)）
        # PyTorch 中复数类型用 torch.complex64（对应 TF 的 tf.complex64）
        amp_complex = amp.to(dtype=torch.complex64)
        phi_complex = phi.to(dtype=torch.complex64)
        
        # 2. 计算复数指数 exp(1j * phi)（对应 TF 的 exp(1j * phi_complex)）
        # torch.exp 支持复数输入，1j 用 torch.tensor(0+1j, dtype=torch.complex64) 表示
        exp_term = torch.exp(torch.tensor(0 + 1j, dtype=torch.complex64) * phi_complex)
        
        # 3. 幅度 × 复数指数（对应 TF 的 amp_complex * exp_term）
        output = amp_complex * exp_term
        
        return output
    

    def masked_obj_func(self,obj, support):
        """
        与 TF Lambda 层逻辑完全对齐的 PyTorch 函数：
        接收 [obj, support] 列表输入，返回 obj * 转换为复数的 support
        
        参数:
            inputs: 列表，格式为 [obj, support]
                - obj: PyTorch 张量（需为复数类型，如 torch.complex64，对应 TF 的 x[0]）
                - support: PyTorch 张量（布尔/浮点型，对应 TF 的 x[1]）
        
        返回:
            torch.Tensor: 逐元素乘法结果，类型为 torch.complex64
        """
        # 2. 将 support 转为复数类型（对应 TF 的 tf.cast(x[1], tf.complex64)）
        support_complex = support.to(torch.complex64)
        # 3. 逐元素乘法（与 TF 的 x[0] * x[1] 逻辑一致）
        return obj * support_complex
    

    def fourier_transform(self,data: torch.Tensor) -> torch.Tensor:
        """
        PyTorch 版傅里叶变换（匹配 TF 衍射计算的 _fourier_transform 逻辑）：
        1. 对输入场数据进行中心化（光学衍射计算的标准预处理）
        2. 执行 2D 快速傅里叶变换（FFT）
        3. 输出复数类型的频域场数据（与 TF 傅里叶变换的复数输出一致）
        
        参数:
            data: 源平面场数据，PyTorch 张量（支持形状：[B, H, W] 或 [H, W]，需为复数类型，如 torch.complex64）
        
        返回:
            torch.Tensor: 频域场数据，复数类型（torch.complex64），形状与输入一致
        """
        # 步骤1：中心化（对应 TF 的 tf.fftshift 前处理，消除低频分量在角落的问题）
        # 计算中心偏移（对每个空间维度，偶数/奇数尺寸均适用）
        shift = [dim // 2 for dim in data.shape[-2:]]  # 仅对最后两个空间维度（H, W）处理
        data_centered = torch.roll(data, shifts=shift, dims=(-2, -1))  # 沿 H、W 维度滚动中心化
        
        # 步骤2：2D FFT（对应 TF 的 tf.signal.fft2d 或 tf.fft2d，根据输入维度自动适配）
        # 若输入带 batch 维度（[B, H, W]），需指定对最后两个维度做 FFT
        diff = torch.fft.fft2(data_centered, dim=(-2, -1))
        
        # 步骤3：（可选）若原 TF 函数有逆中心化，可添加 torch.fft.ifftshift(diff, dim=(-2, -1))
        # （根据光学衍射场景默认不添加，若需匹配特定 TF 实现可调整）
        return diff

    # 4. 对应 TF 的 ff_propagation (核心修正！！！)
    def ff_propagation(self, data: torch.Tensor) -> torch.Tensor:
        """
        对应 TF 的 _fourier_transform 逻辑：
        TF: Permute -> ifftshift -> fft3d -> fftshift -> Permute -> abs
        PT: 直接对 (D, H, W) 维度进行 ifftshift -> fftn -> fftshift -> abs
        """
        # data shape 在 PT 中应该是: (Batch, Channel, Depth, Height, Width)
        # 我们针对最后三个维度 (D, H, W) 进行操作
        spatial_dims = (-3, -2, -1)

        # 步骤 A: IFFT Shift (将零频分量移到角落，准备做 FFT)
        x = torch.fft.ifftshift(data, dim=spatial_dims)

        # 步骤 B: 3D FFT (对应 TF 的 fft3d)
        # 注意：fftn 会自动处理多维，我们指定作用在 D,H,W 上
        x = torch.fft.fftn(x, dim=spatial_dims)

        # 步骤 C: FFT Shift (将零频分量移回中心，对应 TF 的 fftshift)
        x = torch.fft.fftshift(x, dim=spatial_dims)

        # 步骤 D: 取模并转为 float32 (对应 TF 的 abs + cast)
        intensity = torch.abs(x).to(torch.float32)

        return intensity
    


    def forward(self, x):
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph_raw = self.decoder2(x1)



        ph = self.phi_layer(ph_raw)

        # --- Index 91: support ---
        # 对应 TF: Lambda(lambda x: get_mask(x))
        # 注意：TF 图中 support 依赖 amp，虽然索引在 Obj 后面，但数据流通常是并行的
        support = self.support_layer(amp)

        # --- Index 90: Obj ---
        # 对应 TF: Lambda(lambda x: combine_complex...)([decoded1, decoded2])
        obj = self.obj_layer(amp, ph)

        # --- Index 92: masked_obj ---
        # 对应 TF: Lambda(lambda x: x[0] * x[1])([obj, support])
        masked_obj = self.masked_obj_layer(obj, support)

        preds_amp = torch.abs(masked_obj)

        # --- Index 93: farfield_diff ---
        # 对应 TF: Lambda(lambda x: ff_propagation(x))(masked_obj)
        psi = self.farfield_layer(masked_obj)
        

        return psi, obj, preds_amp, ph, support
    
