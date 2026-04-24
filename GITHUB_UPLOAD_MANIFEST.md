# GitHub Upload Manifest

Remote repository:

```text
git@github.com:xinyizhiren/UMamba-PhaseNN.git
```

Only the following training-chain files should be uploaded:

```text
.gitignore
.gitattributes
README.md
requirements.txt
GITHUB_UPLOAD_MANIFEST.md
check_mamba_env.py
lcrc_run_interactive.sh
oyys_lcrc_train_multiGPU_DDP_fp16.py
data_loader.py
UMambaEnc_3d.py
utils.py
AutoPhaseNN_model_relu.py
plans_diffraction_3d.json
```

Do not upload:

```text
runs/
__pycache__/
*.ipynb
*.png
*.pt
*.pth
*.npy
CDI_simulation_upsamp_noise/
AutoPhase/
old or experimental training scripts
```

Reasoning:

- `lcrc_run_interactive.sh` is the launch entrypoint.
- `oyys_lcrc_train_multiGPU_DDP_fp16.py` owns training, validation, checkpointing, and TensorBoard logging.
- `data_loader.py` owns memory-mapped diffraction/real-space data loading.
- `UMambaEnc_3d.py`, `utils.py`, and `plans_diffraction_3d.json` are required for `MODEL_NAME=umamba`.
- `check_mamba_env.py` is the standalone server-side diagnostic for Mamba import and CUDA runtime health.
- `AutoPhaseNN_model_relu.py` is included because the same entrypoint still supports `MODEL_NAME=autophasenn`.
- Runtime outputs and datasets are intentionally excluded.
