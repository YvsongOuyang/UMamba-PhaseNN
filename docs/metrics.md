# Metric and Loss Guide

这个项目里目前同时有 AutoPhaseNN pipeline 和 UMamba pipeline。为了避免把训练目标、论文指标、诊断指标混在一起判断，后续统一按下面几类看结果。

## 1. 训练目标

这些数直接来自训练循环，只有 `OptTotal` 一定参与反向传播。

| 名称 | 含义 | 用途 |
| --- | --- | --- |
| `OptTotal` | 当前实际用于 `backward()` 的总优化目标 | 判断训练是否在下降 |
| `FTLoss` | `--loss-type` 选择的衍射图损失，默认通常是 L1 | diff-only 训练时它就是主要目标 |
| `SupportWeighted` | `support_weight * SupportBCE` | 只有 `--support-weight > 0` 时才影响训练 |
| `AmpL1Full` | 全体素预测 amp 和真值 amp 的 L1 | 实空间监控项，默认 diff-only 时不反传 |
| `PhaseL1PredSup` | 在预测 support 上计算的 phase L1 | 实空间监控项，默认 diff-only 时不反传 |
| `SupportBCE` | raw amp 与真值 support 的 balanced BCE | 辅助 support 监控项，单独变小不等于 support 位置正确 |

注意：AutoPhaseNN 的论文式训练默认是只拟合衍射图，所以 `AmpL1Full`、`PhaseL1PredSup`、`SupportBCE` 在默认配置下只是监控项。

## 2. 首选评估指标

验证和测试脚本会输出 `metric_groups`，推荐优先看这两组。

### Reciprocal-space primary metrics

这些指标衡量衍射图拟合质量。

| 名称 | 趋势 | 说明 |
| --- | --- | --- |
| `paper_modulus_mae` | 越低越好 | 论文 Eq. 1 风格的衍射模长 MAE |
| `relative_l1_modulus` | 越低越好 | 按样本总量归一化的相对 L1，更适合跨样本比较 |
| `chi2_modulus` | 越低越好 | 论文 Eq. 2 风格的 chi-square 误差 |
| `pearson_corr` | 越高越好 | 衍射图 Pearson 相关系数 |

### Real-space primary metrics

这些指标衡量重建出来的 amp、phase、support 是否像真实物体。

| 名称 | 趋势 | 说明 |
| --- | --- | --- |
| `real_amp_l1` | 越低越好 | 全体素 amp L1，但稀疏物体上可能偏乐观 |
| `real_amp_global_ssim` | 越高越好 | 全局 3D amp SSIM-like 分数 |
| `real_support_iou` | 越高越好 | 预测 support 与真值 support 的 IoU |
| `real_support_dice` | 越高越好 | 预测 support 与真值 support 的 Dice |
| `real_support_pred_fraction` | 接近真值更好 | 预测 support 占整个体素网格的比例 |
| `real_support_volume_ratio` | 接近 1 更好 | 预测 support 体积 / 真值 support 体积 |
| `real_phase_mae_true_support` | 越低越好 | 在真值 support 上的 wrapped phase MAE |

当前最关键的问题是：模型可能让衍射图指标下降，但 support 扩散到大面积空间，导致实空间结构很差。因此判断模型时不能只看 `FTLoss` 或 `paper_modulus_mae`。

## 3. 诊断指标

这些指标保留用于排查，不建议单独作为模型优劣结论。

| 名称 | 说明 |
| --- | --- |
| `relative_log_mse` | log 域衍射误差，适合看动态范围问题 |
| `pearson_loss` | `1 - pearson_corr`，日志里旧的 `PCC` 很多时候其实是这个 loss |
| `voxel_mse` / `voxel_rmse` | 原始尺度 MSE/RMSE，受数据是否归一化和强峰值影响很大 |
| `real_amp_mse` / `real_amp_rmse` | 实空间 amp 的 MSE/RMSE，受稀疏背景影响 |
| `real_support_true_fraction` | 真值 support 稀疏程度，用来解释 support 指标 |
| `real_phase_mae_intersection` | 在预测和真值 support 交集上的 phase 误差 |
| `real_phase_rmse_true_support` | 真值 support 上的 phase RMSE |

## 4. 推荐判断顺序

1. 先看可视化切片和 support 指标，确认模型没有用大面积 support 换取衍射图拟合。
2. 再看 `paper_modulus_mae`、`relative_l1_modulus`、`chi2_modulus`、`pearson_corr`，判断衍射图拟合质量。
3. 最后看 `real_amp_global_ssim`、`real_amp_l1`、`real_phase_mae_true_support`，判断实空间细节。
4. 如果衍射图指标变好，但 `real_support_volume_ratio` 远大于 1 或 `real_support_iou` 很低，优先判定为实空间不匹配，而不是模型已经学会了物体。

## 5. 日志命名约定

新的训练日志尽量避免使用模糊名称：

| 旧习惯名 | 新名称 | 说明 |
| --- | --- | --- |
| `Loss` / `TrainLoss` | `OptTotal` | 实际反传总目标 |
| `FT` / `LossFT` | `FTLoss` | 衍射图训练损失 |
| `L1` | `FT_L1` | 衍射图 raw L1 |
| `MAE` | `FT_RelL1` | 衍射图相对 L1 |
| `MSE` | `FT_Chi2` | 论文 chi-square 风格误差，不是 raw MSE |
| `PCC` | `FT_PearsonCorr` 或 `FT_PearsonLoss` | 相关系数越高越好，loss 越低越好，需要看清后缀 |
| `VoxelMSE` | `FT_VoxelMSE` | 原始尺度 MSE，只作诊断 |

## 6. 当前实验的重点

目前最需要确认的是：diff-only 训练能不能在小样本上同时降低衍射图误差和实空间 support 错误。如果 `FTLoss` 下降但 `real_support_volume_ratio` 长期很大，说明仅靠衍射图损失和软 support 约束仍不足以提供强空间先验，需要继续引入更硬的 support 或中心区域约束。
