# e-CDI 复数出口波—衍射配对数据集生成器（10,000 例）

训练数据字段、划分及直接读取 HDF5 的方法见 [数据集说明](DATASET_CARD.md)和[训练使用说明](TRAINING_USAGE.md)。

这是一套**可独立部署**的生成程序，内含当前已验证的 abTEM 多切片仿真内核及四个材料 CIF；不依赖 Windows 上的原仿真目录。默认生成 512×512、六种材料均衡采样的 10,000 对模拟数据，输出到 Linux 路径 `/data_ssd/oyys/eCDI`。其中 9,000 例划为训练开发池，1,000 例划为独立测试集。程序保存复数出口波真值、理想衍射、仅泊松噪声探测器计数和元数据。这里的真值是**探针—样品作用后的出口波**，不是材料独立的静电势。

本程序只把代码上传 GitHub；大型生成数据应留在服务器 SSD，不应提交到 Git。

## 输出组织

```text
/data_ssd/oyys/eCDI/
├─ data/
│  └─ ecdi_10k.h5
├─ manifests/
│  ├─ splits_9000_1000.json         精确 train=9000、test=1000
│  └─ splits_training_with_val.json  在 9000 中再留 500 验证：8500/500/1000
├─ quality/
│  ├─ validation_report.json        傅里叶配对、完整性等质检
│  └─ sample_index.jsonl            每例材料、种子、计数等索引
└─ metadata/
   ├─ generation_plan.json          生效的仿真/划分配置
   └─ dataset_summary.json          完成后统计、实际文件大小
```

两份划分清单都只保存 HDF5 **索引**，不复制图像。第二份用于需要验证集的训练程序：其 500 例验证来自 9,000 例训练开发池，测试集始终保持同一 1,000 例且不用于选模。六种材料在各划分中按比例分层抽样。测试集与训练集仍源自相同的六种基础晶体结构，属于**同源分布测试**，不是“新材料泛化”测试。

HDF5 的主要数组形状为 `[10000, 512, 512]`：`observed_counts` 是原始模拟计数输入；`exit_wave_real`/`exit_wave_imag` 是配对标签；还有 `ideal_intensity`、`observed_log`、`support`、`metadata_json` 和 `valid`。此配置的 `support` 为全视场占位，**不是精确紧支撑**。每例由固定总种子和样本索引确定，重复运行会从 `valid=false` 的样本续跑；已有文件与配置不一致时会拒绝混写。不要同时启动两个写入同一 HDF5 的进程。

## 安装与运行（Linux）

建议 Python 3.11/3.12、GPU 环境装有匹配 CUDA 的 NVIDIA 驱动；若 CuPy wheel 与服务器 CUDA 不兼容，应按服务器 CUDA 版本调整依赖。先确认 `/data_ssd/oyys/eCDI` 是目标 SSD 上的目录且当前用户有写权限。

```bash
cd ecdi_10k_dataset_generator
python -m venv .venv
.venv/bin/python -m pip install -r requirements-gpu.txt
.venv/bin/python -m pip install -e .
.venv/bin/python build_dataset.py --dry-run
.venv/bin/python build_dataset.py
```

CPU 测试可改装 `requirements-cpu.txt`，再加 `--device cpu`。完整任务可在 `tmux` 或服务器作业系统中执行；意外中断后**原命令重跑即可续写**，完成生成后程序会做全量质检与划分。仅需重新质检/生成清单时运行 `--prepare-only`，不会重算物理图像。

运行前请根据目标实验调整 `configs/ecdi_10k.json`。默认六种材料为 Si、Li、石墨、LiFePO₄、LiCoO₂、LiMn₂O₄，固定 512² 网格、单帧相干 e-CDI、28–44 Å 高斯束宽、100–1500 e/Å² 剂量、泊松计数噪声。`configs/dataset_plan.json` 管理目标路径和划分数量。**若修改配置，必须使用新的输出目录；不要将不同模拟条件混进同一数据集。**

小规模自检，不写入正式 `/data_ssd/oyys/eCDI`：

```bash
.venv/bin/python build_dataset.py --device cpu --train-count 10 --test-count 2 --validation-count 2 --free-reserve-gib 0 --output-root ./outputs/smoke_12
```

这会得到 12 例独立 HDF5、质检报告和两份划分清单。正式运行不加这些覆盖参数。

## 空间预估

以现有同配置 24 例试验 HDF5 的实际大小 69,457,065 B 外推：**10,000 例约 28.9 GB（26.9 GiB）**。压缩率受材料、剂量和数据分布影响，不能保证每批相同。按五个 float32 图像数组加一个 bool `support` 计算，未压缩像素本体约 **55.1 GB（51.3 GiB）**，另有 HDF5 索引和元数据。训练常用的 `observed_counts` + 出口波实部/虚部三数组 NPY 导出将**额外占 31.5 GB（29.3 GiB）**。

建议 SSD 可用空间：**只保存 HDF5 至少 70 GiB；若再保存 NPY 训练副本，至少 110 GiB**。这是包含未压缩波动和操作余量的建议值，不是最终文件一定会占这么多。脚本会在开始前按较保守的未压缩估计检查剩余空间；输出 `metadata/dataset_summary.json` 会记录实际占用。请勿把数据 HDF5 或 NPY 上传 GitHub。

## 与独立训练程序对接

相邻的 `exitwave_reconstruction_training` 程序可直接读取 `manifests/splits_training_with_val.json` 对应的 HDF5（`--data-format hdf5`）；若用其 NPY 模式，先运行该训练程序里的 `export_npy.py`，输入此处的 `data/ecdi_10k.h5` 并指定 `observed_counts`。导出的 NPY 也应放在 SSD，避免占用系统盘。

本数据集仍是**模拟数据**：结构只来自六种基础晶胞/CIF，探测器仅加泊松计数噪声，尚未用目标仪器实测帧标定束形、部分相干性、背景、坏点等。10,000 例提升训练规模，但不能自行消除仿真到真实仪器的分布差异。

## 程序检查

```bash
.venv/bin/python -m pytest tests
```
