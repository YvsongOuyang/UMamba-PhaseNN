import torch
import torch.nn as nn
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
        self.support_layer = SupportLayer(self.T)   # Index 91
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
                nn.LeakyReLU(negative_slope=0.01),
                nn.BatchNorm3d(filters_out,  momentum=0.01, eps=1e-3),
                nn.Conv3d(filters_out, filters_out, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.01),
                nn.BatchNorm3d(filters_out,  momentum=0.01, eps=1e-3),
            ]

        return block
    

    def forward(self, x):
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph_raw = self.decoder2(x1)
        ph = self.phi_layer(ph_raw)

        # --- Index 91: support ---
        # 对应 TF: Lambda support mask
        # 注意：TF 图中 support 依赖 amp，虽然索引在 Obj 后面，但数据流通常是并行的
        support = self.support_layer(amp)

        # --- Index 90: Obj ---
        # 对应 TF: Lambda complex object combine
        obj = self.obj_layer(amp, ph)

        # --- Index 92: masked_obj ---
        # 对应 TF: Lambda(lambda x: x[0] * x[1])([obj, support])
        masked_obj = self.masked_obj_layer(obj, support)

        preds_amp = torch.abs(masked_obj)

        # --- Index 93: farfield_diff ---
        # 对应 TF: Lambda farfield diffraction
        psi = self.farfield_layer(masked_obj)
        

        return psi, obj, preds_amp, ph, support
    
