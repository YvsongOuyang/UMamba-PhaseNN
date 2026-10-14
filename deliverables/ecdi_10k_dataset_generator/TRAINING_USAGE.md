# e-CDI 数据集训练使用说明

本说明适用于生成器默认输出 `/data_ssd/oyys/eCDI/`。详细物理参数和数据限制见 [DATASET_CARD.md](DATASET_CARD.md)。**真实完成情况以服务器上的质检报告为准**；不要仅凭文件存在就认为 10,000 条均有效。

## 1. 核验与划分

在服务器上运行：

```bash
python -c 'import json; from pathlib import Path; p=Path("/data_ssd/oyys/eCDI"); q=json.loads((p/"quality/validation_report.json").read_text()); s=json.loads((p/"manifests/splits_training_with_val.json").read_text()); print("passed=",q["passed"],"samples=",q["sample_count"],"splits=",s["counts"])'
```

默认预期 `passed=True`、`samples=10000`、`train=8500`、`val=500`、`test=1000`。如果报告或清单缺失，在**生成器目录**运行 `python build_dataset.py --prepare-only`：它重新审计 HDF5 并写出清单，不重新模拟图像。不要从同一个数据集中再次随机拆分，始终使用同一份 `splits_training_with_val.json`；测试索引只用于最终评价，不参与调参。

## 2. 直接读取 HDF5（推荐起点）

数据文件为 `/data_ssd/oyys/eCDI/data/ecdi_10k.h5`，划分清单为 `/data_ssd/oyys/eCDI/manifests/splits_training_with_val.json`。清单内的 `dataset_path` 是相对清单目录的 HDF5 路径，包含互不重叠的 `train`、`val`、`test` 索引。直接读取不会复制三份图像，也不需要额外生成 NPY。

| 角色 | HDF5 字段 | 用法 |
| --- | --- | --- |
| 网络输入 | `observed_counts` | 原始非负模拟探测器计数；可在训练 transform 中逐样本计算 `sqrt(counts / max(sum(counts), 1))` |
| 监督标签 | `exit_wave_real`、`exit_wave_imag` | 两个 `float32` 通道，表示同一样本的复数出口波 |
| 前向一致性参照 | `ideal_intensity` | 无计数噪声的归一化衍射强度；不能作为有噪声输入的替代 |
| 显示 | `observed_log` | 对数显示帧；不能与原始计数的归一化公式混用 |
| 元数据 | `metadata_json` | 逐帧材料、束形、剂量、仿真参数等；已知探针可从中构造条件输入 |
| 完整性 | `valid` | 全部应为 `true` |
| 占位掩膜 | `support` | 默认全视场为真，不是真实晶体 support，勿作为边界真值 |

所有主要图像数组的形状应为 `[10000, 512, 512]`。标签是**探针与晶体作用后的出口波**，不是材料本身的静电势或原子坐标。单帧强度具有整体相位和共轭孪生歧义，训练损失/评价应按任务定义处理这些等价解。

已有的独立训练程序 `exitwave_reconstruction_training` 支持 `--data-format hdf5`，其 Dataset 会在各 DataLoader worker 中懒打开 HDF5；默认训练配置仍指向旧试验集，使用本数据时必须显式指定上述清单。如果该训练程序已部署到服务器，可在**训练程序目录**运行：

```bash
CUDA_VISIBLE_DEVICES=0 python train_exitwave.py \
  --data-format hdf5 \
  --manifest /data_ssd/oyys/eCDI/manifests/splits_training_with_val.json \
  --output-dir /data_ssd/oyys/eCDI_runs/exitwave_hdf5_v1
```

训练环境需要匹配 GPU 的 PyTorch、`h5py` 和其他训练依赖；不必与生成环境 `ecdi-gen` 相同。上述命令只在训练程序已存在且依赖已安装时可运行。生成器不包含独立训练程序；截至本说明编写时，远端仓库主分支尚无 `deliverables/exitwave_reconstruction_training/`，不能仅拉取生成器就执行训练命令。

## 3. 读取速度与是否需要 NPY

该 HDF5 使用按样本分块和 gzip 级别 1 压缩，SSD 上可随机读取，但每条样本仍需要读取并解压计数、实部、虚部三个 512² 数组。因此**可以直接读，不等于保证读取零开销**。现有训练配置的 `data.num_workers` 默认是 `0`；先用它做正确性检查，再把配置改为 `2` 或 `4` 测首轮每步耗时和 GPU 利用率，选择实测更快的值。每个 worker 应独立打开只读 HDF5，不能把父进程已打开的 `h5py.File` 句柄共享给 worker。训练程序当前的 Dataset 采用懒打开方式。

可先在已有 `h5py` 的环境中做一个只读的随机样本吞吐测试（包含三个训练字段的解压，但不包含 PyTorch 变换与 GPU 计算）：

```bash
python - <<'PY'
from collections import deque
import logging
import time
import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
path = "/data_ssd/oyys/eCDI/data/ecdi_10k.h5"
with h5py.File(path, "r") as source:
    indices = np.random.default_rng(0).choice(len(source["valid"]), size=256, replace=False)
    started = time.perf_counter()
    deque((tuple(source[key][int(index)] for key in
                 ("observed_counts", "exit_wave_real", "exit_wave_imag"))
           for index in indices), maxlen=0)
    elapsed = time.perf_counter() - started
logging.info("HDF5 random read: %.1f samples/s (256 samples, %.2f s)", len(indices) / elapsed, elapsed)
PY
```

此数值只是 I/O 参考；系统页缓存会影响重复测试。最终仍应以训练 step 时间、数据等待和 GPU 利用率来判断。

建议先跑 50–100 个训练 step，观察数据等待是否占每步显著比例，以及 GPU 是否长期空闲。仅在确认 I/O/解压成为瓶颈后，才考虑额外导出连续 NPY 内存映射；这会额外占用约 **29.3 GiB**（仅输入、实部、虚部三路），实际提速必须以同一训练配置测量。GPU 计算本身慢、batch 太小或模型尚未充分利用 GPU 时，导出 NPY 不一定有效。

不要提前把整个数据集加载到内存或 GPU：三路未压缩 `float32` 图像约 29.3 GiB，且训练时还需要模型、激活和 batch 空间。

## 4. 可复现与适用边界

训练时记录清单路径、HDF5 的 `config_sha256`、训练配置、随机种子和模型检查点；保持测试集完全留出。训练/验证/测试都包含相同六种基础材料结构，所以此划分测试的是同源分布泛化，不是新材料泛化。数据仅加入泊松计数噪声，尚未按目标真实仪器的背景、部分相干性、探测器响应与漂移做分布标定。内部傅里叶配对通过不等于对真实衍射帧的拟真性已经验证。
