import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TFCompatibleAutoPhaseNN(nn.Module):
    """PyTorch AutoPhaseNN model matching the converted TF2/Keras checkpoint."""

    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold
        self.layers = nn.ModuleDict()

        conv_specs = {
            "conv3d": (1, 32),
            "conv3d_1": (32, 32),
            "conv3d_2": (32, 64),
            "conv3d_3": (64, 64),
            "conv3d_4": (64, 128),
            "conv3d_5": (128, 128),
            "conv3d_6": (128, 256),
            "conv3d_7": (256, 256),
            "conv3d_8": (256, 512),
            "conv3d_9": (512, 512),
            "conv3d_10": (512, 256),
            "conv3d_11": (256, 256),
            "conv3d_12": (256, 128),
            "conv3d_13": (128, 128),
            "conv3d_14": (128, 64),
            "conv3d_15": (64, 64),
            "conv3d_16": (64, 32),
            "conv3d_17": (32, 32),
            "conv3d_18": (32, 1),
            "conv3d_19": (512, 128),
            "conv3d_20": (128, 128),
            "conv3d_21": (128, 128),
            "conv3d_22": (128, 128),
            "conv3d_23": (128, 64),
            "conv3d_24": (64, 64),
            "conv3d_25": (64, 32),
            "conv3d_26": (32, 32),
            "conv3d_27": (32, 1),
        }
        for name, (in_channels, out_channels) in conv_specs.items():
            self.layers[name] = nn.Conv3d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

        bn_specs = {
            "batch_normalization": 32,
            "batch_normalization_1": 32,
            "batch_normalization_2": 64,
            "batch_normalization_3": 64,
            "batch_normalization_4": 128,
            "batch_normalization_5": 128,
            "batch_normalization_6": 256,
            "batch_normalization_7": 256,
            "batch_normalization_8": 512,
            "batch_normalization_9": 512,
            "batch_normalization_10": 256,
            "batch_normalization_11": 256,
            "batch_normalization_12": 128,
            "batch_normalization_13": 128,
            "batch_normalization_14": 64,
            "batch_normalization_15": 64,
            "batch_normalization_16": 32,
            "batch_normalization_17": 32,
            "batch_normalization_18": 128,
            "batch_normalization_19": 128,
            "batch_normalization_20": 128,
            "batch_normalization_21": 128,
            "batch_normalization_22": 64,
            "batch_normalization_23": 64,
            "batch_normalization_24": 32,
            "batch_normalization_25": 32,
        }
        for name, channels in bn_specs.items():
            self.layers[name] = nn.BatchNorm3d(
                channels, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True
            )

        self.zero_pad = nn.ConstantPad3d(16, 0.0)

    def _conv_lrelu_bn(self, x, conv_name, bn_name):
        x = self.layers[conv_name](x)
        x = F.leaky_relu(x, negative_slope=0.01)
        return self.layers[bn_name](x)

    def _conv_relu_bn(self, x, conv_name, bn_name):
        x = self.layers[conv_name](x)
        x = F.relu(x)
        return self.layers[bn_name](x)

    def _encoder_block(self, x, conv1, bn1, conv2, bn2, pool=True):
        x = self._conv_lrelu_bn(x, conv1, bn1)
        x = self._conv_lrelu_bn(x, conv2, bn2)
        if pool:
            x = F.max_pool3d(x, kernel_size=2, stride=2)
        return x

    def _up_lrelu_block(self, x, conv1, bn1, conv2, bn2):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self._conv_lrelu_bn(x, conv1, bn1)
        return self._conv_lrelu_bn(x, conv2, bn2)

    def _pad_relu_block(self, x, conv1, bn1, conv2, bn2):
        x = self.zero_pad(x)
        x = self._conv_relu_bn(x, conv1, bn1)
        return self._conv_relu_bn(x, conv2, bn2)

    def encode(self, x):
        x = self._encoder_block(
            x, "conv3d", "batch_normalization", "conv3d_1", "batch_normalization_1"
        )
        x = self._encoder_block(
            x, "conv3d_2", "batch_normalization_2", "conv3d_3", "batch_normalization_3"
        )
        x = self._encoder_block(
            x, "conv3d_4", "batch_normalization_4", "conv3d_5", "batch_normalization_5"
        )
        x = self._encoder_block(
            x, "conv3d_6", "batch_normalization_6", "conv3d_7", "batch_normalization_7"
        )
        return self._encoder_block(
            x,
            "conv3d_8",
            "batch_normalization_8",
            "conv3d_9",
            "batch_normalization_9",
            pool=False,
        )

    def decode_amplitude(self, encoded):
        x = self._up_lrelu_block(
            encoded, "conv3d_10", "batch_normalization_10", "conv3d_11", "batch_normalization_11"
        )
        x = self._up_lrelu_block(
            x, "conv3d_12", "batch_normalization_12", "conv3d_13", "batch_normalization_13"
        )
        x = self._up_lrelu_block(
            x, "conv3d_14", "batch_normalization_14", "conv3d_15", "batch_normalization_15"
        )
        x = self._pad_relu_block(
            x, "conv3d_16", "batch_normalization_16", "conv3d_17", "batch_normalization_17"
        )
        return torch.sigmoid(self.layers["conv3d_18"](x))

    def decode_phase(self, encoded):
        x = self._up_lrelu_block(
            encoded, "conv3d_19", "batch_normalization_18", "conv3d_20", "batch_normalization_19"
        )
        x = self._up_lrelu_block(
            x, "conv3d_21", "batch_normalization_20", "conv3d_22", "batch_normalization_21"
        )
        x = self._up_lrelu_block(
            x, "conv3d_23", "batch_normalization_22", "conv3d_24", "batch_normalization_23"
        )
        x = self._pad_relu_block(
            x, "conv3d_25", "batch_normalization_24", "conv3d_26", "batch_normalization_25"
        )
        return math.pi * torch.tanh(self.layers["conv3d_27"](x))

    def forward(self, x):
        encoded = self.encode(x)
        amp = self.decode_amplitude(encoded)
        phi = self.decode_phase(encoded)

        support = torch.where(
            amp >= self.threshold, torch.ones_like(amp), torch.zeros_like(amp)
        )
        obj = torch.complex(amp * torch.cos(phi), amp * torch.sin(phi))
        masked_obj = obj * support.to(torch.complex64)

        shifted = torch.fft.ifftshift(masked_obj, dim=(-3, -2, -1))
        farfield = torch.fft.fftn(shifted, dim=(-3, -2, -1))
        farfield = torch.fft.fftshift(farfield, dim=(-3, -2, -1))
        farfield = torch.abs(farfield).to(torch.float32)

        masked_amp = torch.abs(masked_obj).to(torch.float32)
        return farfield, masked_obj, masked_amp, phi, support


def load_weights(model, checkpoint_path, strict=True, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint

