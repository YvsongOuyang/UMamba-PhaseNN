"""Numerically compatible PyTorch implementation of the TensorFlow PhaseUNet."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple


PUBLISHED_MODEL_VARIANT = "published"
REDUCED_MODEL_VARIANT = "reduced"
REDUCED_BN_NO_OUTER_SKIP_VARIANT = "reduced_bn_no_outer_skip"
MODEL_VARIANTS = (
    REDUCED_MODEL_VARIANT,
    REDUCED_BN_NO_OUTER_SKIP_VARIANT,
    PUBLISHED_MODEL_VARIANT,
)
DEFAULT_MODEL_VARIANT = REDUCED_MODEL_VARIANT
BN_EPS = 1e-3
BN_MOMENTUM = 0.01


class TensorFlowSameConv3d(nn.Module):
    """Conv3D with TensorFlow SAME padding for stride one."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        kernel = _triple(kernel_size)
        dilation_3d = _triple(dilation)
        padding: list[int] = []
        for size, rate in reversed(tuple(zip(kernel, dilation_3d))):
            total = rate * (size - 1)
            before = total // 2
            padding.extend((before, total - before))
        self.pad = tuple(padding)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=1,
            padding=0,
            dilation=dilation_3d,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, self.pad))


class TensorFlowSameConvTranspose3d(nn.Module):
    """Conv3DTranspose matching Keras SAME size and voxel alignment."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        kernel = _triple(kernel_size)
        self.crop_before = tuple(max((size - 2) // 2, 0) for size in kernel)
        self.deconv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=2,
            padding=0,
            output_padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_shape = tuple(size * 2 for size in x.shape[-3:])
        full = self.deconv(x)
        depth, height, width = self.crop_before
        return full[
            ...,
            depth : depth + target_shape[0],
            height : height + target_shape[1],
            width : width + target_shape[2],
        ]


class HighStrainPhaseUNet(nn.Module):
    """3D U-Net that predicts reciprocal-space phase.

    Input and output use PyTorch's ``[batch, channel, depth, height, width]``
    layout. ``reduced`` removes the deepest encoder-decoder scale and caps the
    bottleneck at 1024 channels. ``reduced_bn_no_outer_skip`` additionally uses
    BatchNorm and removes the full-resolution skip. ``published`` retains the
    numerically matched six-level Keras architecture with a 2048-channel
    bottleneck.
    """

    def __init__(self, model_variant: str = DEFAULT_MODEL_VARIANT) -> None:
        super().__init__()
        if model_variant not in MODEL_VARIANTS:
            raise ValueError(
                f"Unknown model variant {model_variant!r}; expected one of {MODEL_VARIANTS}."
            )
        self.model_variant = model_variant
        self.use_batch_norm = model_variant == REDUCED_BN_NO_OUTER_SKIP_VARIANT
        self.use_outer_skip = model_variant != REDUCED_BN_NO_OUTER_SKIP_VARIANT
        self.is_published = model_variant == PUBLISHED_MODEL_VARIANT

        layers: dict[str, nn.Module] = {
            "conv3d": TensorFlowSameConv3d(1, 8, 5, dilation=8),
            "conv3d_1": TensorFlowSameConv3d(1, 8, 5, dilation=5),
            "conv3d_2": TensorFlowSameConv3d(1, 8, 5, dilation=3),
            "conv3d_3": TensorFlowSameConv3d(1, 8, 5, dilation=1),
            "conv3d_4": TensorFlowSameConv3d(33, 16, 5, dilation=6),
            "conv3d_5": TensorFlowSameConv3d(33, 16, 5, dilation=4),
            "conv3d_6": TensorFlowSameConv3d(33, 16, 5, dilation=2),
            "conv3d_7": TensorFlowSameConv3d(33, 16, 5, dilation=1),
            "conv3d_8": TensorFlowSameConv3d(97, 128, 4),
            "conv3d_9": TensorFlowSameConv3d(128, 256, 3),
            "conv3d_10": TensorFlowSameConv3d(256, 512, 3),
        }
        if self.is_published:
            layers["conv3d_11"] = TensorFlowSameConv3d(512, 1024, 3)
        if self.use_outer_skip:
            layers["conv3d_12"] = TensorFlowSameConv3d(33, 16, 3)
        layers.update(
            {
                "conv3d_13": TensorFlowSameConv3d(97, 48, 3),
                "conv3d_14": TensorFlowSameConv3d(128, 64, 3),
                "conv3d_15": TensorFlowSameConv3d(256, 128, 3),
                "conv3d_16": TensorFlowSameConv3d(512, 256, 3),
            }
        )
        if self.is_published:
            layers.update(
                {
                    "conv3d_17": TensorFlowSameConv3d(1024, 512, 3),
                    "conv3d_18": TensorFlowSameConv3d(1024, 2048, 2),
                    "conv3d_transpose": TensorFlowSameConvTranspose3d(
                        2048, 1024, 3
                    ),
                    "conv3d_transpose_1": TensorFlowSameConvTranspose3d(
                        1536, 512, 3
                    ),
                }
            )
        else:
            layers.update(
                {
                    "conv3d_18": TensorFlowSameConv3d(512, 1024, 2),
                    "conv3d_transpose": TensorFlowSameConvTranspose3d(
                        1024, 512, 3
                    ),
                }
            )
        layers.update(
            {
                "conv3d_transpose_2": TensorFlowSameConvTranspose3d(768, 256, 3),
                "conv3d_transpose_3": TensorFlowSameConvTranspose3d(384, 128, 4),
                "conv3d_transpose_4": TensorFlowSameConvTranspose3d(192, 64, 5),
                "conv3d_transpose_5": TensorFlowSameConvTranspose3d(112, 32, 5),
                "conv3d_19": TensorFlowSameConv3d(
                    48 if self.use_outer_skip else 32,
                    16,
                    5,
                ),
                "conv3d_20": TensorFlowSameConv3d(16, 1, 5),
            }
        )
        self.layers = nn.ModuleDict(layers)
        self.batch_norms = nn.ModuleDict()
        if self.use_batch_norm:
            for name, layer in self.layers.items():
                if name == "conv3d_20":
                    continue
                if isinstance(layer, TensorFlowSameConv3d):
                    channels = layer.conv.out_channels
                else:
                    channels = layer.deconv.out_channels
                self.batch_norms[name] = nn.BatchNorm3d(
                    channels,
                    eps=BN_EPS,
                    momentum=BN_MOMENTUM,
                    affine=True,
                    track_running_stats=True,
                )
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Match Keras Glorot-uniform kernels and zero biases."""

        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _modified_encoder(
        self,
        x: torch.Tensor,
        layer_names: tuple[str, str, str, str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = [x]
        for name in layer_names:
            output = self.layers[name](x)
            features.append(self._normalize(output, name))
        skip = self.activation(torch.cat(features, dim=1))
        return F.max_pool3d(skip, kernel_size=2, stride=2), skip

    def _normalize(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        if layer_name in self.batch_norms:
            return self.batch_norms[layer_name](x)
        return x

    def _hidden_layer(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        x = self.layers[layer_name](x)
        return self.activation(self._normalize(x, layer_name))

    def _encoder(
        self,
        x: torch.Tensor,
        layer_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self._hidden_layer(x, layer_name)
        return F.max_pool3d(skip, kernel_size=2, stride=2), skip

    def _skip(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        return self._hidden_layer(x, layer_name)

    def _decoder(
        self,
        x: torch.Tensor,
        layer_name: str,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        return self._hidden_layer(x, layer_name)

    def weighted_layers(self) -> Mapping[str, nn.Conv3d | nn.ConvTranspose3d]:
        """Map Keras layer names to their parameterized PyTorch modules."""

        weighted: dict[str, nn.Conv3d | nn.ConvTranspose3d] = {}
        for name, layer in self.layers.items():
            if isinstance(layer, TensorFlowSameConv3d):
                weighted[name] = layer.conv
            elif isinstance(layer, TensorFlowSameConvTranspose3d):
                weighted[name] = layer.deconv
        return weighted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[B, 1, 64, 64, 64]`` input to an equal-shaped phase volume."""

        x, s1 = self._modified_encoder(
            x,
            ("conv3d", "conv3d_1", "conv3d_2", "conv3d_3"),
        )
        x, s2 = self._modified_encoder(
            x,
            ("conv3d_4", "conv3d_5", "conv3d_6", "conv3d_7"),
        )
        x, s3 = self._encoder(x, "conv3d_8")
        x, s4 = self._encoder(x, "conv3d_9")
        x, s5 = self._encoder(x, "conv3d_10")
        if self.is_published:
            x, s6 = self._encoder(x, "conv3d_11")

        if self.use_outer_skip:
            s1 = self._skip(s1, "conv3d_12")
        s2 = self._skip(s2, "conv3d_13")
        s3 = self._skip(s3, "conv3d_14")
        s4 = self._skip(s4, "conv3d_15")
        s5 = self._skip(s5, "conv3d_16")

        x = self._hidden_layer(x, "conv3d_18")
        x = self._decoder(x, "conv3d_transpose")
        if self.is_published:
            s6 = self._skip(s6, "conv3d_17")
            x = self._decoder(x, "conv3d_transpose_1", s6)
        x = self._decoder(x, "conv3d_transpose_2", s5)
        x = self._decoder(x, "conv3d_transpose_3", s4)
        x = self._decoder(x, "conv3d_transpose_4", s3)
        x = self._decoder(x, "conv3d_transpose_5", s2)

        if self.use_outer_skip:
            x = torch.cat((x, s1), dim=1)
        x = self._hidden_layer(x, "conv3d_19")
        return self.layers["conv3d_20"](x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def infer_model_variant(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Infer the architecture from the bottleneck kernel in a state dict."""

    key = "layers.conv3d_18.conv.weight"
    if key not in state_dict:
        raise ValueError(f"Checkpoint does not contain the required tensor {key!r}.")
    channels = tuple(int(size) for size in state_dict[key].shape[:2])
    if channels == (1024, 512):
        final_hidden_key = "layers.conv3d_19.conv.weight"
        if final_hidden_key not in state_dict:
            raise ValueError(
                f"Checkpoint does not contain the required tensor {final_hidden_key!r}."
            )
        final_hidden_inputs = int(state_dict[final_hidden_key].shape[1])
        if final_hidden_inputs == 32:
            return REDUCED_BN_NO_OUTER_SKIP_VARIANT
        if final_hidden_inputs == 48:
            return REDUCED_MODEL_VARIANT
        raise ValueError(
            f"Cannot infer reduced model variant from {final_hidden_key} with "
            f"{final_hidden_inputs} input channels."
        )
    if channels == (2048, 1024):
        return PUBLISHED_MODEL_VARIANT
    raise ValueError(
        f"Cannot infer model variant from {key} with channel shape {channels}."
    )
