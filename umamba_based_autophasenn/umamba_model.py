import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_UMAMBA_FILE = SCRIPT_DIR / "UMambaEnc_3d.py"
LOCAL_UTILS_FILE = SCRIPT_DIR / "utils.py"


def _load_module_from_path(module_name, module_path, register_as=None):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[register_as or module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_local_utils():
    current = sys.modules.get("utils")
    current_file = getattr(current, "__file__", None)
    if current_file and Path(current_file).resolve() == LOCAL_UTILS_FILE:
        return current
    return _load_module_from_path(
        "umamba_based_autophasenn_local_utils",
        LOCAL_UTILS_FILE,
        register_as="utils",
    )


def _load_local_umamba_module():
    _ensure_local_utils()
    module_name = "umamba_based_autophasenn_local_UMambaEnc_3d"
    current = sys.modules.get(module_name)
    current_file = getattr(current, "__file__", None)
    if current_file and Path(current_file).resolve() == LOCAL_UMAMBA_FILE:
        return current
    return _load_module_from_path(module_name, LOCAL_UMAMBA_FILE)


get_umamba_enc_3d_from_plans = _load_local_umamba_module().get_umamba_enc_3d_from_plans


def build_umamba_plans(shape=64, batch_size=2):
    shape = int(shape)
    return {
        "dataset_name": "Diffraction3D",
        "original_median_spacing_after_transp": [1.0, 1.0, 1.0],
        "original_median_shape_after_transp": [shape, shape, shape],
        "image_reader_writer": "SimpleITKIO",
        "transpose_forward": [0, 1, 2],
        "transpose_backward": [0, 1, 2],
        "configurations": {
            "3d_fullres": {
                "data_identifier": "nnUNetPlans_3d_fullres",
                "preprocessor_name": "DefaultPreprocessor",
                "batch_size": int(batch_size),
                "patch_size": [shape, shape, shape],
                "median_image_size_in_voxels": [float(shape), float(shape), float(shape)],
                "spacing": [1.0, 1.0, 1.0],
                "normalization_schemes": ["ZScoreNormalization"],
                "use_mask_for_norm": [False],
                "UNet_class_name": "PlainConvUNet",
                "UNet_base_num_features": 32,
                "n_conv_per_stage_encoder": [2, 2, 2, 2, 2],
                "n_conv_per_stage_decoder": [2, 2, 2, 2],
                "num_pool_per_axis": [4, 4, 4, 4],
                "pool_op_kernel_sizes": [
                    [1, 1, 1],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                ],
                "conv_kernel_sizes": [
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                ],
                "unet_max_num_features": 320,
                "resampling_fn_data": "fast_resample_data_or_seg_to_shape",
                "resampling_fn_seg": "fast_resample_data_or_seg_to_shape",
                "resampling_fn_data_kwargs": {
                    "is_seg": False,
                    "order": 3,
                    "order_z": 3,
                    "force_separate_z": None,
                },
                "resampling_fn_seg_kwargs": {
                    "is_seg": True,
                    "order": 1,
                    "order_z": 1,
                    "force_separate_z": None,
                },
                "resampling_fn_probabilities": "fast_resample_data_or_seg_to_shape",
                "resampling_fn_probabilities_kwargs": {
                    "is_seg": False,
                    "order": 1,
                    "order_z": 1,
                    "force_separate_z": None,
                },
                "batch_dice": True,
            },
            "3d_cascade_fullres": {
                "inherits_from": "3d_fullres",
                "previous_stage": "3d_lowres",
            },
        },
        "experiment_planner_used": "ExperimentPlanner",
        "label_manager": "LabelManager",
        "foreground_intensity_properties_per_channel": {
            "0": {
                "max": 3071.0,
                "mean": 97.29716491699219,
                "median": 118.0,
                "min": -1024.0,
                "percentile_00_5": -958.0,
                "percentile_99_5": 270.0,
                "std": 137.8484649658203,
            }
        },
    }


class UMambaAutoPhaseNN(nn.Module):
    """Local UMamba wrapper for the AutoPhaseNN-style pipeline."""

    def __init__(self, threshold=0.1, shape=64, batch_size=2):
        super().__init__()
        self.threshold = threshold
        print(f"Using local UMamba model file: {LOCAL_UMAMBA_FILE}", flush=True)
        if abs(float(threshold) - 0.1) > 1e-12:
            print(
                "Warning: local UMambaEnc_3d.py uses threshold=0.1 internally; "
                f"received threshold={threshold}.",
                flush=True,
            )
        plans_manager = PlansManager(build_umamba_plans(shape=shape, batch_size=batch_size))
        config_manager = plans_manager.get_configuration("3d_fullres")
        dataset_json = {"labels": {"background": 0}, "num_segmentation_heads": 1}
        self.model = get_umamba_enc_3d_from_plans(
            plans_manager=plans_manager,
            dataset_json=dataset_json,
            configuration_manager=config_manager,
            num_input_channels=1,
            deep_supervision=False,
        )

    def forward(self, x):
        return self.model(x)


# Backward-compatible name for existing commands/checkpoints/scripts.
TFCompatibleAutoPhaseNN = UMambaAutoPhaseNN


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_weights(model, checkpoint_path, strict=True, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = extract_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        if not hasattr(model, "model"):
            raise
        model.model.load_state_dict(state_dict, strict=strict)
    return checkpoint


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="Print local UMamba model structure.")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--shape", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--depth", type=int, default=6)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = UMambaAutoPhaseNN(
        threshold=args.threshold,
        shape=args.shape,
        batch_size=args.batch_size,
    ).to(device)
    print(model)
    print(f"model parameters: {count_parameters(model)}")

    try:
        from torchinfo import summary

        print("===================== Local UMamba structure =====================")
        summary(
            model,
            input_size=(args.batch_size, 1, args.shape, args.shape, args.shape),
            device=str(device),
            col_width=20,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            depth=args.depth,
            row_settings=["var_names", "depth"],
        )
    except Exception as exc:
        print(f"torchinfo summary skipped: {exc}")


if __name__ == "__main__":
    main()
