import torch
import torch.nn as nn

from autophasenn_training_pipeline.model_factory import create_model
from autophasenn_training_pipeline.model_residual import (
    LEAKY_RELU_SLOPE,
    ResidualAutoPhaseNN,
    ResidualBlock3D,
)


def test_residual_block_uses_projection_only_for_channel_changes():
    projected = ResidualBlock3D(32, 64)
    identity = ResidualBlock3D(64, 64)

    assert isinstance(projected.projection, nn.Conv3d)
    assert projected.projection.kernel_size == (1, 1, 1)
    assert isinstance(identity.projection, nn.Identity)
    assert projected.conv1.kernel_size == (3, 3, 3)
    assert projected.conv2.kernel_size == (3, 3, 3)
    assert projected.bn1.eps == 1e-3
    assert projected.bn1.momentum == 0.01
    assert LEAKY_RELU_SLOPE == 0.01


def test_encoder_downsampling_is_separate_strided_convolution():
    model = ResidualAutoPhaseNN()

    assert not any(isinstance(module, nn.MaxPool3d) for module in model.modules())
    assert len(model.downsample_layers) == 4
    expected_channels = (32, 64, 128, 256)
    for layer, channels in zip(model.downsample_layers, expected_channels):
        assert layer.in_channels == channels
        assert layer.out_channels == channels
        assert layer.kernel_size == (3, 3, 3)
        assert layer.stride == (2, 2, 2)
        assert layer.padding == (1, 1, 1)


def test_residual_model_preserves_baseline_forward_contract():
    with torch.device("meta"):
        model = create_model("residual", threshold=0.1).eval()
        inputs = torch.empty((2, 1, 64, 64, 64))
        outputs = model(inputs)

    assert len(outputs) == 6
    assert all(output.shape == (2, 1, 64, 64, 64) for output in outputs)
    assert outputs[1].dtype == torch.complex64
    assert all(outputs[index].dtype == torch.float32 for index in (0, 2, 3, 4, 5))


def test_decoder_channel_sequences_match_the_baseline():
    model = ResidualAutoPhaseNN()

    assert [(block.conv1.in_channels, block.conv1.out_channels) for block in model.amplitude_blocks] == [
        (512, 256),
        (256, 128),
        (128, 64),
        (64, 32),
    ]
    assert [(block.conv1.in_channels, block.conv1.out_channels) for block in model.phase_blocks] == [
        (512, 128),
        (128, 128),
        (128, 64),
        (64, 32),
    ]
