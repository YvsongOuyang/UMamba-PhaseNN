"""Tests for the BatchNorm/no-outer-skip ReLU fine-tuning variant."""

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from pytorch_autophasenn.model import (
    REDUCED_BN_NO_OUTER_SKIP_VARIANT,
    REDUCED_BN_RELU_NO_OUTER_SKIP_VARIANT,
    HighStrainPhaseUNet,
    infer_model_variant,
)
from pytorch_autophasenn.train import load_model_state, parse_args


class ReLUFineTuneVariantTest(unittest.TestCase):
    def test_only_activation_differs_from_pretrained_variant(self) -> None:
        source = HighStrainPhaseUNet(REDUCED_BN_NO_OUTER_SKIP_VARIANT)
        target = HighStrainPhaseUNet(REDUCED_BN_RELU_NO_OUTER_SKIP_VARIANT)

        self.assertIsInstance(source.activation, nn.LeakyReLU)
        self.assertIsInstance(target.activation, nn.ReLU)
        self.assertFalse(target.use_outer_skip)
        self.assertTrue(target.use_batch_norm)
        self.assertEqual(tuple(source.state_dict()), tuple(target.state_dict()))
        for key, value in source.state_dict().items():
            self.assertEqual(value.shape, target.state_dict()[key].shape)

    def test_checkpoint_metadata_distinguishes_parameter_free_variant(self) -> None:
        model = HighStrainPhaseUNet(REDUCED_BN_RELU_NO_OUTER_SKIP_VARIANT)

        self.assertEqual(
            infer_model_variant(
                model.state_dict(),
                REDUCED_BN_RELU_NO_OUTER_SKIP_VARIANT,
            ),
            REDUCED_BN_RELU_NO_OUTER_SKIP_VARIANT,
        )
        self.assertEqual(
            infer_model_variant(model.state_dict()),
            REDUCED_BN_NO_OUTER_SKIP_VARIANT,
        )

    def test_relu_variant_has_requested_finetuning_defaults(self) -> None:
        with patch(
            "sys.argv",
            [
                "train",
                "--data-format",
                "autophasenn",
                "--model-variant",
                REDUCED_BN_RELU_NO_OUTER_SKIP_VARIANT,
            ],
        ):
            args = parse_args()

        self.assertEqual(args.data_format, "autophasenn")
        self.assertEqual(args.epochs, 60)
        self.assertEqual(args.learning_rate, 5e-4)

    def test_pretrained_load_allows_only_the_activation_migration(self) -> None:
        state_dict = {
            "layers.conv3d_18.conv.weight": torch.empty(
                1024,
                512,
                1,
                1,
                1,
                device="meta",
            ),
            "layers.conv3d_19.conv.weight": torch.empty(
                16,
                32,
                1,
                1,
                1,
                device="meta",
            ),
        }
        checkpoint = {
            "model_variant": REDUCED_BN_NO_OUTER_SKIP_VARIANT,
            "model_state_dict": state_dict,
        }

        class RecordingModel:
            model_variant = REDUCED_BN_RELU_NO_OUTER_SKIP_VARIANT

            def __init__(self) -> None:
                self.loaded_strictly = False

            def load_state_dict(self, loaded_state, strict: bool) -> None:
                self.loaded_strictly = strict and loaded_state is state_dict

        model = RecordingModel()
        with patch("pytorch_autophasenn.train.torch.load", return_value=checkpoint):
            load_model_state(
                model,
                "checkpoint_best.pt",
                torch.device("cpu"),
                allow_activation_migration=True,
            )
            self.assertTrue(model.loaded_strictly)

            with self.assertRaises(ValueError):
                load_model_state(
                    model,
                    "checkpoint_best.pt",
                    torch.device("cpu"),
                )

    def test_full_model_preserves_volume_shape(self) -> None:
        with torch.device("meta"):
            model = HighStrainPhaseUNet(REDUCED_BN_RELU_NO_OUTER_SKIP_VARIANT)
            output = model(torch.empty(1, 1, 64, 64, 64, device="meta"))
        self.assertEqual(tuple(output.shape), (1, 1, 64, 64, 64))


if __name__ == "__main__":
    unittest.main()
