# AutoPhaseNN 中间特征物理一致性实验

这是一个与主训练流程完全隔离的推理实验。它不会修改或训练 baseline，
只通过 forward hooks 读取 `TFCompatibleAutoPhaseNN` 的中间特征。

## 验证对象

按照空间尺寸和网络深度匹配四组编码器、振幅解码器和相位解码器特征：

| 实验层 | 编码器特征 | 振幅解码器特征 | 相位解码器特征 |
|---|---:|---:|---:|
| `latent_8` | `256 x 8^3` | `256 x 8^3` | `128 x 8^3` |
| `latent_16` | `128 x 16^3` | `128 x 16^3` | `128 x 16^3` |
| `latent_32` | `64 x 32^3` | `64 x 32^3` | `64 x 32^3` |
| `latent_64` | `32 x 64^3` | `32 x 64^3` | `32 x 64^3` |

四组振幅特征与编码器通道数一致；只有 `latent_8` 的相位特征为 128 通道，实验将每个相位通道
重复一次以匹配 256 通道，并在结果说明中保留这一人为假设。振幅使用 RMS 变为非负值并按通道归一化，
相位使用 `pi * tanh` 映射到 `[-pi, pi]`。随后沿三个空间维执行与模型输出端相同的
`ifftshift -> FFT -> fftshift -> abs`，并与对应编码器特征的绝对值比较。

由于中间特征没有固定物理量纲，比较前对预测值和目标值分别进行逐样本、逐通道 RMS 归一化。
因此主要误差反映空间形状差异，未经归一化的尺度差异由 `raw_rms_ratio` 单独记录。

实验同时计算三组结果：

- `paired`：原始配对的振幅、相位中间特征；
- `zero_phase`：将相位置零，检验相位特征是否提供额外信息；
- `rolled_phase`：将相位沿三个空间轴平移半个周期，作为破坏空间对应关系的对照。

只有当 `paired` 在多个层级、多个样本上稳定优于两个对照，才说明这种特定的中间物理映射
捕获到了统计关联。即便如此，也不能直接证明隐藏通道本身就是物理振幅或相位。

## 服务器运行

在仓库根目录执行：

```bash
python experiments/autophasenn_intermediate_physics/run_experiment.py \
  --checkpoint /data_ssd/oyys/autophasenn/autophasenn_pipeline_output/BASELINE_RUN/checkpoint_best.pt \
  --data-dir /data_ssd/oyys/autophasenn \
  --data-diff val_diff.npy \
  --num-samples 5000 \
  --batch-size 1 \
  --num-workers 4 \
  --device cuda
```

先运行 8 个样本检查环境：

```text
--limit 8
```

默认结果写入本实验目录下的 `outputs/<时间戳>/`：

- `report.md`：均值结果和配对相对对照的优势；
- `summary.json`：完整聚合统计；
- `per_sample_metrics.csv`：逐样本、逐层、逐对照结果；
- `resolved_args.json`：实际运行参数；
- `run.log`：控制台日志。

## 结果判断

`report.md` 中的 paired advantage 已统一成“正值有利于配对假设”：

- 误差指标：`control - paired`；
- 相关性指标：`paired - control`。

如果优势接近零或为负，不建议把这个显式 FFT 关系直接写入训练损失。此时更合理的下一步是使用
可学习的通道投影或跨分支注意力建立隐式关联，再通过消融实验判断其作用。
