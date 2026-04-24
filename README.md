# UMamba-PhaseNN Training Chain

这个仓库只保留从启动脚本到训练所需的核心文件，避免上传原项目里的 notebook、缓存、运行日志、数据和模型权重。

## 训练入口

```bash
bash lcrc_run_interactive.sh
```

常用覆盖方式：

```bash
DATA_PATH=/data_hdd/oyys/autophaseNN \
BATCH_SIZE=8 \
EPOCHS=100 \
MODEL_NAME=umamba \
LOSS_TYPE=comb2 \
bash lcrc_run_interactive.sh
```

## Mamba 环境诊断

在服务器上先跑下面这个脚本，可以比单纯 `import mamba_ssm` 更可靠地判断安装是否正常：

```bash
python check_mamba_env.py
```

这个脚本会依次检查：

1. Python / PyTorch / CUDA 基础环境
2. `causal_conv1d` 是否可导入
3. `mamba_ssm` 和 `selective_scan` 是否可导入
4. `causal_conv1d` 小规模 CUDA 前向和反向
5. `Mamba` 模块小规模 CUDA 前向和反向

如果它全部通过，说明 Mamba 的核心安装大概率没有问题；但这仍然不能 100% 保证完整训练一定成功，因为训练还依赖：

- `nnunetv2`
- `dynamic-network-architectures`
- 数据读取路径和 dtype
- 更大 batch / mixed precision 下的显存与数值稳定性
- 当前项目里的具体网络与训练代码路径

## 链路说明

1. `lcrc_run_interactive.sh` 设置 GPU、数据路径、超参数和输出目录。
2. `oyys_lcrc_train_multiGPU_DDP_fp16.py` 解析参数，构建数据集、模型、优化器、scheduler，并执行训练/验证/保存 checkpoint。
3. `data_loader.py` 以 mmap 方式读取 `train_diff.npy`、`train_real.npy`、`val_diff.npy`、`val_real.npy`。
4. `UMambaEnc_3d.py` 和 `utils.py` 提供 UMamba 3D 相位恢复网络及傅里叶传播层。
5. `AutoPhaseNN_model_relu.py` 提供可选的 AutoPhaseNN baseline。
6. `plans_diffraction_3d.json` 保存 UMamba 3D 的 64x64x64 diffraction plans 配置。

## 数据格式

默认读取四个文件：

- `train_diff.npy`: shape `[N, 64, 64, 64]`, dtype `float32`
- `train_real.npy`: shape `[N, 64, 64, 64]`, dtype `complex64`
- `val_diff.npy`: shape `[N, 64, 64, 64]`, dtype `float32`
- `val_real.npy`: shape `[N, 64, 64, 64]`, dtype `complex64`

`data_loader.py` 会优先按标准 `.npy` 读取；如果文件不是标准 `.npy`，会退回到 raw memmap 读取。

## 输出

训练输出默认写入：

```text
${DATA_PATH}/runs/<run_name>/
```

其中包含：

- `setting.json`
- `logging.txt`
- `tensorboard/`
- `best_model.pt`
- `training_model_*.pt`

这些运行产物不应提交到 GitHub。
