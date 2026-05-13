import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
from tqdm.notebook import tqdm
import numpy as np
from numpy.fft import fftn, fftshift
import argparse  # 新增这一行


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
        print("n_filt_in:", filters_in )
        print("n_filt_out:", filters_out)
        print("use_down_stride", use_down_stride)
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
        ph_raw = self.decoder2(x1)*np.pi



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

        # --- Index 93: farfield_diff ---
        # 对应 TF: Lambda(lambda x: ff_propagation(x))(masked_obj)
        psi = self.farfield_layer(masked_obj)
        

        return psi, obj, amp, ph, support


    

#    def forward(self, x):
#        x1 = self.encoder(x)
#        amp = self.decoder1(x1)
#        ph = self.decoder2(x1)*np.pi
#
#
#        support = self.get_mask(amp)
#
#        obj = self.combine_complex(amp, ph)
#
#        masked_obj = self.masked_obj_func(obj, support)
#
#        psi = self.ff_propagation(masked_obj)
#        
#
#        return psi, obj, amp, ph, support


    
    # def forward(self, x):
    #     x1 = self.encoder(x)
    #     amp = self.decoder1(x1)
    #     ph = self.decoder2(x1)
    #     print(x1.shape)
    #     print(amp.shape)
    #     print(ph.shape)

    #     # Normalize amp to max 1 before applying support
    #     amp = torch.clip(amp, min=0, max=1.0)

    #     mask = torch.tensor([0, 1], dtype=amp.dtype, device=amp.device)
    #     if self.argv.unsupervise:
    #         # Apply the support to amplitude
    #         amp = torch.where(amp < self.T, mask[0], amp)

    #     # Restore -pi to pi range
    #     # Using tanh activation (-1 to 1) for phase so multiply by pi
    #     ph = ph*np.pi

    #     # Pad the predictions to 2X
    #     #pad = nn.ConstantPad3d(int(self.H/4), 0)
    #     #amp = pad(amp)
    #     #ph = pad(ph)

    #     # get support for viz
    #     support = torch.zeros(amp.shape, device=amp.device)
    #     support = torch.where(amp < self.T, mask[0], mask[1])

    #     # Create the complex number
    #     with torch.cuda.amp.autocast(enabled=False):
    #         complex_x = torch.complex(
    #             amp.float()*torch.cos(ph.float()), amp.float()*torch.sin(ph.float()))

    #     # Compute FT, shift and take abs
    #     y = torch.fft.fftn(complex_x, dim=(-3, -2, -1))
    #     # FFT shift will move the wrong dimensions if not specified
    #     y = torch.fft.fftshift(y, dim=(-3, -2, -1))
    #     y = torch.abs(y)

    #     # Normalize to scale_I
    #     if self.argv.scale_I > 0:
    #         max_I = torch.amax(y, dim=[-1, -2, -3], keepdim=True)
    #         y = self.argv.scale_I*torch.div(y, max_I+1e-6)  # Prevent zero div

    #     return y, complex_x, amp, ph, support


torch_model = Network(argv)

# ----------------------
# 模型跑通测试：生成随机输入 + 前向推理
# ----------------------
def test_model_forward():
    # 1. 配置模型输入参数（与 argv 保持一致）
    batch_size = 1  # 测试用批量大小（可调整）
    in_channels = 1  # 模型输入通道数（从 encoder 第一层 Conv3d in_channels=1 可知）
    input_shape = (batch_size, in_channels, argv.shape, argv.shape, argv.shape)  # PyTorch 格式：(B, C, D, H, W)
    
    # 2. 生成随机测试数据（模拟真实输入的数值范围，这里用 0-1 随机数）
    # 若真实场景输入有特定范围（如CT值-1000~400），可调整生成逻辑（如 torch.randn + 缩放）
    test_input = torch.rand(input_shape, dtype=torch.float32)
    print(f"\n📌 测试输入信息：")
    print(f"  输入形状：{test_input.shape} (B, C, D, H, W)")
    print(f"  输入数据范围：[{test_input.min():.4f}, {test_input.max():.4f}]")
    
    # 3. 模型前向推理（关闭梯度计算，提升速度）
    with torch.no_grad():
        # 若有GPU，可将模型和数据移到GPU（需确保CUDA可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_model.to(device)
        test_input = test_input.to(device)
        print(f"  使用设备：{device}")
        
        # 调用模型 forward 方法（输出：远场强度、复数目标、幅度、相位、掩码）
        try:
            psi, obj, amp, ph, support = torch_model(test_input)
            print(f"\n✅ 模型前向推理成功！")
            
            # 4. 打印输出结果信息（验证输出形状和合理性）
            print(f"\n📊 输出结果详情：")
            output_info = [
                ("远场强度 (psi)", psi),       # 实数（torch.float32）
                ("复数目标 (obj)", obj),       # 复数（torch.complex64）
                ("幅度 (amp)", amp),           # 实数（torch.float32）
                ("相位 (ph)", ph),             # 实数（torch.float32，ph是Tanh输出乘pi，为实数）
                ("掩码 (support)", support)    # 实数（torch.float32）
            ]
            for name, tensor in output_info:
                # 处理复数张量：用模长的min/max表示范围
                if tensor.is_complex():
                    tensor_abs = torch.abs(tensor)  # 计算复数的模长（转为实数）
                    min_val = tensor_abs.min()
                    max_val = tensor_abs.max()
                # 处理实数张量：直接取min/max
                else:
                    min_val = tensor.min()
                    max_val = tensor.max()
                # 打印（保留4位小数）
                print(f"  {name:20s} | 形状：{tensor.shape} | 数据范围：[{min_val:.4f}, {max_val:.4f}]")
                
            # 额外验证相位范围（理论应为 -pi ~ pi，因 forward 中 ph = decoder2输出 * np.pi，且 decoder2 用 Tanh()）
            if not (ph.min() >= -np.pi - 1e-3 and ph.max() <= np.pi + 1e-3):
                print(f"⚠️  相位范围警告：实际范围 [{ph.min():.4f}, {ph.max():.4f}]（理论应接近 -pi ~ pi）")
            else:
                print(f"✅ 相位范围正常（符合 -pi ~ pi 预期）")
                
            # 额外验证掩码（应为 0/1 二值化，因 get_mask 用 torch.where 生成）
            if not torch.all((support == 0) | (support == 1)):
                print(f"⚠️  掩码值警告：存在非0/1值（理论应为二值掩码）")
            else:
                print(f"✅ 掩码正常（仅含 0/1 二值）")
                
        except Exception as e:
            print(f"\n❌ 模型前向推理失败！错误信息：")
            print(f"  {type(e).__name__}: {e}")
            return False
    return True


torch_model.eval()


pytorch_params = list(torch_model.named_parameters())
print("PyTorch模型参数名（共{}个）：".format(len(pytorch_params)))
for idx, (name, param) in enumerate(pytorch_params):
    print(f"  {idx:2d}: {name:50s} | shape: {param.shape}")



# 加载TF权重
tf_weights = np.load("tf_weights.npz", allow_pickle=True)
print(f"加载TF权重：共{len(tf_weights.keys())}个变量")


# 加载TF权重后，添加以下代码打印所有变量名
tf_param_names = sorted(tf_weights.keys())
print(f"TF权重中所有变量名（共{len(tf_param_names)}个）：")
for idx, name in enumerate(tf_param_names):
    # 重点标注BN缓冲区变量
    if "moving_mean" in name or "moving_variance" in name:
        print(f"  {idx:2d}: {name} 👉 BN缓冲区")
    else:
        print(f"  {idx:2d}: {name}")
# ----------------------
# 4. 精准映射表（TF变量名 → PyTorch参数名/缓冲区名）
# 基于你的模型实际参数编号生成，100%匹配！
# ----------------------
# 分为两部分：parameters（可训练参数）和buffers（BN的均值/方差）
param_map = {
    # ======================================
    # Encoder: Conv3D + BatchNorm (parameters)
    # ======================================
    # TF conv3d → encoder.0 (Conv3D)
    "conv3d/kernel:0": "encoder.0.weight",
    "conv3d/bias:0": "encoder.0.bias",
    # TF batch_normalization → encoder.2 (BN)
    "batch_normalization/gamma:0": "encoder.2.weight",
    "batch_normalization/beta:0": "encoder.2.bias",
    
    # TF conv3d_1 → encoder.3 (Conv3D)
    "conv3d_1/kernel:0": "encoder.3.weight",
    "conv3d_1/bias:0": "encoder.3.bias",
    # TF batch_normalization_1 → encoder.5 (BN)
    "batch_normalization_1/gamma:0": "encoder.5.weight",
    "batch_normalization_1/beta:0": "encoder.5.bias",
    
    # TF conv3d_2 → encoder.7 (Conv3D)
    "conv3d_2/kernel:0": "encoder.7.weight",
    "conv3d_2/bias:0": "encoder.7.bias",
    # TF batch_normalization_2 → encoder.9 (BN)
    "batch_normalization_2/gamma:0": "encoder.9.weight",
    "batch_normalization_2/beta:0": "encoder.9.bias",
    
    # TF conv3d_3 → encoder.10 (Conv3D)
    "conv3d_3/kernel:0": "encoder.10.weight",
    "conv3d_3/bias:0": "encoder.10.bias",
    # TF batch_normalization_3 → encoder.12 (BN)
    "batch_normalization_3/gamma:0": "encoder.12.weight",
    "batch_normalization_3/beta:0": "encoder.12.bias",
    
    # TF conv3d_4 → encoder.14 (Conv3D)
    "conv3d_4/kernel:0": "encoder.14.weight",
    "conv3d_4/bias:0": "encoder.14.bias",
    # TF batch_normalization_4 → encoder.16 (BN)
    "batch_normalization_4/gamma:0": "encoder.16.weight",
    "batch_normalization_4/beta:0": "encoder.16.bias",
    
    # TF conv3d_5 → encoder.17 (Conv3D)
    "conv3d_5/kernel:0": "encoder.17.weight",
    "conv3d_5/bias:0": "encoder.17.bias",
    # TF batch_normalization_5 → encoder.19 (BN)
    "batch_normalization_5/gamma:0": "encoder.19.weight",
    "batch_normalization_5/beta:0": "encoder.19.bias",
    
    # TF conv3d_6 → encoder.21 (Conv3D)
    "conv3d_6/kernel:0": "encoder.21.weight",
    "conv3d_6/bias:0": "encoder.21.bias",
    # TF batch_normalization_6 → encoder.23 (BN)
    "batch_normalization_6/gamma:0": "encoder.23.weight",
    "batch_normalization_6/beta:0": "encoder.23.bias",
    
    # TF conv3d_7 → encoder.24 (Conv3D)
    "conv3d_7/kernel:0": "encoder.24.weight",
    "conv3d_7/bias:0": "encoder.24.bias",
    # TF batch_normalization_7 → encoder.26 (BN)
    "batch_normalization_7/gamma:0": "encoder.26.weight",
    "batch_normalization_7/beta:0": "encoder.26.bias",
    
    # TF conv3d_8 → encoder.28 (Conv3D)
    "conv3d_8/kernel:0": "encoder.28.weight",
    "conv3d_8/bias:0": "encoder.28.bias",
    # TF batch_normalization_8 → encoder.30 (BN)
    "batch_normalization_8/gamma:0": "encoder.30.weight",
    "batch_normalization_8/beta:0": "encoder.30.bias",
    
    # TF conv3d_9 → encoder.31 (Conv3D)
    "conv3d_9/kernel:0": "encoder.31.weight",
    "conv3d_9/bias:0": "encoder.31.bias",
    # TF batch_normalization_9 → encoder.33 (BN)
    "batch_normalization_9/gamma:0": "encoder.33.weight",
    "batch_normalization_9/beta:0": "encoder.33.bias",

    # ======================================
    # Decoder1 (Amp) → parameters
    # ======================================
    # TF conv3d_10 → decoder1.0 (Conv3D)
    "conv3d_10/kernel:0": "decoder1.1.weight",
    "conv3d_10/bias:0": "decoder1.1.bias",
    # TF batch_normalization_10 → decoder1.2 (BN)
    "batch_normalization_10/gamma:0": "decoder1.3.weight",
    "batch_normalization_10/beta:0": "decoder1.3.bias",
    
    # TF conv3d_11 → decoder1.4 (Conv3D)
    "conv3d_11/kernel:0": "decoder1.4.weight",
    "conv3d_11/bias:0": "decoder1.4.bias",
    # TF batch_normalization_11 → decoder1.6 (BN)
    "batch_normalization_11/gamma:0": "decoder1.6.weight",
    "batch_normalization_11/beta:0": "decoder1.6.bias",
    
    # TF conv3d_12 → decoder1.7 (Conv3D)
    "conv3d_12/kernel:0": "decoder1.8.weight",
    "conv3d_12/bias:0": "decoder1.8.bias",
    # TF batch_normalization_12 → decoder1.9 (BN)
    "batch_normalization_12/gamma:0": "decoder1.10.weight",
    "batch_normalization_12/beta:0": "decoder1.10.bias",
    
    # TF conv3d_13 → decoder1.11 (Conv3D)
    "conv3d_13/kernel:0": "decoder1.11.weight",
    "conv3d_13/bias:0": "decoder1.11.bias",
    # TF batch_normalization_13 → decoder1.13 (BN)
    "batch_normalization_13/gamma:0": "decoder1.13.weight",
    "batch_normalization_13/beta:0": "decoder1.13.bias",
    
    # TF conv3d_14 → decoder1.14 (Conv3D)
    "conv3d_14/kernel:0": "decoder1.15.weight",
    "conv3d_14/bias:0": "decoder1.15.bias",
    # TF batch_normalization_14 → decoder1.16 (BN)
    "batch_normalization_14/gamma:0": "decoder1.17.weight",
    "batch_normalization_14/beta:0": "decoder1.17.bias",
    
    # TF conv3d_15 → decoder1.18 (Conv3D)
    "conv3d_15/kernel:0": "decoder1.18.weight",
    "conv3d_15/bias:0": "decoder1.18.bias",
    # TF batch_normalization_15 → decoder1.20 (BN)
    "batch_normalization_15/gamma:0": "decoder1.20.weight",
    "batch_normalization_15/beta:0": "decoder1.20.bias",
    
    # TF conv3d_16 → decoder1.21 (Conv3D)
    "conv3d_16/kernel:0": "decoder1.22.weight",
    "conv3d_16/bias:0": "decoder1.22.bias",
    # TF batch_normalization_16 → decoder1.23 (BN)
    "batch_normalization_16/gamma:0": "decoder1.24.weight",
    "batch_normalization_16/beta:0": "decoder1.24.bias",
    
    # TF conv3d_17 → decoder1.25 (Conv3D)
    "conv3d_17/kernel:0": "decoder1.25.weight",
    "conv3d_17/bias:0": "decoder1.25.bias",
    # TF batch_normalization_17 → decoder1.27 (BN)
    "batch_normalization_17/gamma:0": "decoder1.27.weight",
    "batch_normalization_17/beta:0": "decoder1.27.bias",
    
    # TF conv3d_18 (输出层) → decoder1.28 (Conv3D)
    "conv3d_18/kernel:0": "decoder1.28.weight",
    "conv3d_18/bias:0": "decoder1.28.bias",

    # ======================================
    # Decoder2 (Phase) → parameters
    # ======================================
    # TF conv3d_19 → decoder2.0 (Conv3D)
    "conv3d_19/kernel:0": "decoder2.1.weight",
    "conv3d_19/bias:0": "decoder2.1.bias",
    # TF batch_normalization_18 → decoder2.2 (BN)
    "batch_normalization_18/gamma:0": "decoder2.3.weight",
    "batch_normalization_18/beta:0": "decoder2.3.bias",
    
    # TF conv3d_20 → decoder2.4 (Conv3D)
    "conv3d_20/kernel:0": "decoder2.4.weight",
    "conv3d_20/bias:0": "decoder2.4.bias",
    # TF batch_normalization_19 → decoder2.6 (BN)
    "batch_normalization_19/gamma:0": "decoder2.6.weight",
    "batch_normalization_19/beta:0": "decoder2.6.bias",
    
    # TF conv3d_21 → decoder2.7 (Conv3D)
    "conv3d_21/kernel:0": "decoder2.8.weight",
    "conv3d_21/bias:0": "decoder2.8.bias",
    # TF batch_normalization_20 → decoder2.9 (BN)
    "batch_normalization_20/gamma:0": "decoder2.10.weight",
    "batch_normalization_20/beta:0": "decoder2.10.bias",
    
    # TF conv3d_22 → decoder2.11 (Conv3D)
    "conv3d_22/kernel:0": "decoder2.11.weight",
    "conv3d_22/bias:0": "decoder2.11.bias",
    # TF batch_normalization_21 → decoder2.13 (BN)
    "batch_normalization_21/gamma:0": "decoder2.13.weight",
    "batch_normalization_21/beta:0": "decoder2.13.bias",
    
    # TF conv3d_23 → decoder2.14 (Conv3D)
    "conv3d_23/kernel:0": "decoder2.15.weight",
    "conv3d_23/bias:0": "decoder2.15.bias",
    # TF batch_normalization_22 → decoder2.16 (BN)
    "batch_normalization_22/gamma:0": "decoder2.17.weight",
    "batch_normalization_22/beta:0": "decoder2.17.bias",
    
    # TF conv3d_24 → decoder2.18 (Conv3D)
    "conv3d_24/kernel:0": "decoder2.18.weight",
    "conv3d_24/bias:0": "decoder2.18.bias",
    # TF batch_normalization_23 → decoder2.20 (BN)
    "batch_normalization_23/gamma:0": "decoder2.20.weight",
    "batch_normalization_23/beta:0": "decoder2.20.bias",
    
    # TF conv3d_25 → decoder2.21 (Conv3D)
    "conv3d_25/kernel:0": "decoder2.22.weight",
    "conv3d_25/bias:0": "decoder2.22.bias",
    # TF batch_normalization_24 → decoder2.23 (BN)
    "batch_normalization_24/gamma:0": "decoder2.24.weight",
    "batch_normalization_24/beta:0": "decoder2.24.bias",
    
    # TF conv3d_26 → decoder2.25 (Conv3D)
    "conv3d_26/kernel:0": "decoder2.25.weight",
    "conv3d_26/bias:0": "decoder2.25.bias",
    # TF batch_normalization_25 → decoder2.27 (BN)
    "batch_normalization_25/gamma:0": "decoder2.27.weight",
    "batch_normalization_25/beta:0": "decoder2.27.bias",
    
    # TF conv3d_27 (输出层) → decoder2.28 (Conv3D)
    "conv3d_27/kernel:0": "decoder2.28.weight",
    "conv3d_27/bias:0": "decoder2.28.bias",
}




# BN缓冲区映射（TF moving_mean/moving_variance → PyTorch running_mean/running_var）
buffer_map = {
    # ======================================
    # Encoder BN buffers
    # ======================================
    "batch_normalization/moving_mean:0": "encoder.2.running_mean",
    "batch_normalization/moving_variance:0": "encoder.2.running_var",
    "batch_normalization_1/moving_mean:0": "encoder.5.running_mean",
    "batch_normalization_1/moving_variance:0": "encoder.5.running_var",
    "batch_normalization_2/moving_mean:0": "encoder.9.running_mean",
    "batch_normalization_2/moving_variance:0": "encoder.9.running_var",
    "batch_normalization_3/moving_mean:0": "encoder.12.running_mean",
    "batch_normalization_3/moving_variance:0": "encoder.12.running_var",
    "batch_normalization_4/moving_mean:0": "encoder.16.running_mean",
    "batch_normalization_4/moving_variance:0": "encoder.16.running_var",
    "batch_normalization_5/moving_mean:0": "encoder.19.running_mean",
    "batch_normalization_5/moving_variance:0": "encoder.19.running_var",
    "batch_normalization_6/moving_mean:0": "encoder.23.running_mean",
    "batch_normalization_6/moving_variance:0": "encoder.23.running_var",
    "batch_normalization_7/moving_mean:0": "encoder.26.running_mean",
    "batch_normalization_7/moving_variance:0": "encoder.26.running_var",
    "batch_normalization_8/moving_mean:0": "encoder.30.running_mean",
    "batch_normalization_8/moving_variance:0": "encoder.30.running_var",
    "batch_normalization_9/moving_mean:0": "encoder.33.running_mean",
    "batch_normalization_9/moving_variance:0": "encoder.33.running_var",

    # ======================================
    # Decoder1 BN buffers
    # ======================================
    "batch_normalization_10/moving_mean:0": "decoder1.3.running_mean",
    "batch_normalization_10/moving_variance:0": "decoder1.3.running_var",
    "batch_normalization_11/moving_mean:0": "decoder1.6.running_mean",
    "batch_normalization_11/moving_variance:0": "decoder1.6.running_var",
    "batch_normalization_12/moving_mean:0": "decoder1.10.running_mean",
    "batch_normalization_12/moving_variance:0": "decoder1.10.running_var",
    "batch_normalization_13/moving_mean:0": "decoder1.13.running_mean",
    "batch_normalization_13/moving_variance:0": "decoder1.13.running_var",
    "batch_normalization_14/moving_mean:0": "decoder1.17.running_mean",
    "batch_normalization_14/moving_variance:0": "decoder1.17.running_var",
    "batch_normalization_15/moving_mean:0": "decoder1.20.running_mean",
    "batch_normalization_15/moving_variance:0": "decoder1.20.running_var",
    "batch_normalization_16/moving_mean:0": "decoder1.24.running_mean",
    "batch_normalization_16/moving_variance:0": "decoder1.24.running_var",
    "batch_normalization_17/moving_mean:0": "decoder1.27.running_mean",
    "batch_normalization_17/moving_variance:0": "decoder1.27.running_var",

    # ======================================
    # Decoder2 BN buffers
    # ======================================
    "batch_normalization_18/moving_mean:0": "decoder2.3.running_mean",
    "batch_normalization_18/moving_variance:0": "decoder2.3.running_var",
    "batch_normalization_19/moving_mean:0": "decoder2.6.running_mean",
    "batch_normalization_19/moving_variance:0": "decoder2.6.running_var",
    "batch_normalization_20/moving_mean:0": "decoder2.10.running_mean",
    "batch_normalization_20/moving_variance:0": "decoder2.10.running_var",
    "batch_normalization_21/moving_mean:0": "decoder2.13.running_mean",
    "batch_normalization_21/moving_variance:0": "decoder2.13.running_var",
    "batch_normalization_22/moving_mean:0": "decoder2.17.running_mean",
    "batch_normalization_22/moving_variance:0": "decoder2.17.running_var",
    "batch_normalization_23/moving_mean:0": "decoder2.20.running_mean",
    "batch_normalization_23/moving_variance:0": "decoder2.20.running_var",
    "batch_normalization_24/moving_mean:0": "decoder2.24.running_mean",
    "batch_normalization_24/moving_variance:0": "decoder2.24.running_var",
    "batch_normalization_25/moving_mean:0": "decoder2.27.running_mean",
    "batch_normalization_25/moving_variance:0": "decoder2.27.running_var",
}

# ----------------------
# 5. 权重迁移核心逻辑（parameters + buffers）
# ----------------------
with torch.no_grad():
    # 新增：拿到模型参数的真实引用（不是副本）
    torch_params = {name: param for name, param in torch_model.named_parameters()}
    # 迁移可训练参数（Conv3D权重/偏置、BN gamma/beta）
    param_success = 0
    param_missing = []
    for tf_name, torch_name in param_map.items():
        if tf_name not in tf_weights:
            param_missing.append(tf_name)
            continue
        if torch_name not in dict(torch_model.named_parameters()):
            param_missing.append(f"TF: {tf_name} → PyTorch: {torch_name}")
            continue
        
        tf_param = tf_weights[tf_name]
        torch_param = torch_model.state_dict()[torch_name]
        
        # 3D卷积权重维度转换：TF (d,h,w,in_c,out_c) → PyTorch (out_c,in_c,d,h,w)
        if "conv" in tf_name and "weight" in torch_name:  # 正确逻辑：用TF的参数名判断
            tf_param = tf_param.transpose(4, 3, 0, 1, 2)
        
        # 赋值
        torch_param_new = torch.tensor(tf_param, dtype=torch_param.dtype)
        assert torch_param_new.shape == torch_param.shape, \
        f"\n维度不匹配！\n  TF变量名：{tf_name}\n  PyTorch变量名：{torch_name}\n  TF维度：{tf_param.shape}\n  PyTorch维度：{torch_param.shape}"
        #torch_model.state_dict()[torch_name].copy_(torch_param_new)
        # 替换：直接修改参数的真实数据
        torch_params[torch_name].data.copy_(torch_param_new)
        param_success += 1
        print(f"✅ 参数迁移：{tf_name:50s} → {torch_name}")

    # 迁移BN缓冲区（running_mean/running_var）
    buffer_success = 0
    buffer_missing = []
    torch_buffers = dict(torch_model.named_buffers())
    for tf_name, torch_name in buffer_map.items():
        if tf_name not in tf_weights:
            buffer_missing.append(tf_name)
            continue
        if torch_name not in torch_buffers:
            buffer_missing.append(f"TF: {tf_name} → PyTorch: {torch_name}")
            continue
        
        tf_buffer = tf_weights[tf_name]
        torch_buffer = torch_buffers[torch_name]
        
        # 缓冲区维度一致，直接赋值
        torch_buffer_new = torch.tensor(tf_buffer, dtype=torch_buffer.dtype)
        assert torch_buffer_new.shape == torch_buffer.shape, f"缓冲区维度不匹配：{tf_buffer.shape} vs {torch_buffer.shape}"
        # 直接修改缓冲区（buffers不在state_dict中，需通过模块访问）
        module_name, buffer_attr = torch_name.rsplit('.', 1)
        module = torch_model
        for submodule in module_name.split('.'):
            module = getattr(module, submodule)
        getattr(module, buffer_attr).data.copy_(torch_buffer_new)  # 只改这一行！
        
        buffer_success += 1
        print(f"✅ 缓冲区迁移：{tf_name:50s} → {torch_name}")

# ----------------------
# 6. 迁移统计与验证
# ----------------------
print(f"\n" + "="*60)
print(f"迁移统计：")
print(f"✅ 成功迁移参数：{param_success}/{len(param_map)}")
print(f"✅ 成功迁移缓冲区：{buffer_success}/{len(buffer_map)}")
if param_missing:
    print(f"⚠️  缺失参数：{len(param_missing)}个 → {param_missing[:5]}...")
if buffer_missing:
    print(f"⚠️  缺失缓冲区：{len(buffer_missing)}个 → {buffer_missing[:5]}...")

# ----------------------
# 7. 保存PyTorch权重（含模型结构配置）
# ----------------------
torch_model.eval()  # 这行加在所有迁移代码后面，强制用导入的均值/方差
save_path = "torch_transferred_weights_final.pth"
torch.save(
    torch_model.state_dict(),  # 直接保存模型的参数字典，不含任何多余数据
    save_path
)

print(f"\n📥 最终权重已保存到：{save_path}")
print(f"📊 模型总参数数：{sum(p.numel() for p in torch_model.parameters()):,}")
print(f"📊 模型总缓冲区数：{sum(b.numel() for b in torch_model.buffers()):,}")
