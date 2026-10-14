# e-CDI 复数出口波配对模拟数据集

## 用途与位置

本数据集用于训练“单帧电子相干衍射强度 → 复数出口波”的监督模型。数据位于服务器 `/data_ssd/oyys/eCDI/`；此说明依据默认生成配置编写，**实际完成数量、材料分布和质量结果以服务器上的报告为准**。

| 项目 | 默认配置 |
| --- | --- |
| 样本数 | 10,000 个独立模拟帧 |
| 图像尺寸 | 512 × 512 |
| 训练开发池 / 独立测试 | 9,000 / 1,000 |
| 推荐训练 / 验证 / 测试 | 8,500 / 500 / 1,000；验证集从训练开发池中划出 |
| 材料 | Si、金属 Li、石墨、LiFePO₄、LiCoO₂、LiMn₂O₄；来源均衡采样 |
| 电子能量 | 180–220 keV |
| 晶体厚度 | 12–42 Å |
| 高斯照明幅值 FWHM | 28–44 Å |
| 入射剂量 | 100–1,500 e/Å² |
| 探测器噪声 | 泊松计数；默认无读出噪声、背景、坏点、饱和与 MTF 模糊 |

这些参数范围是**生成配置**，不是目标仪器的实测标定。各帧的具体抽样值在 `metadata_json` 中。

## 文件与训练输入

```text
/data_ssd/oyys/eCDI/
├── data/ecdi_10k.h5
├── manifests/splits_9000_1000.json
├── manifests/splits_training_with_val.json
├── quality/validation_report.json
├── quality/sample_index.jsonl
├── metadata/generation_plan.json
└── metadata/dataset_summary.json
```

推荐直接用 `splits_training_with_val.json` 的 `train`、`val`、`test` 索引读取同一个 HDF5；清单不会复制图像，不需要再创建三份数据。HDF5 中主要数组的形状均为 `[10000, 512, 512]`：

| 字段 | 含义 | 训练用途 |
| --- | --- | --- |
| `observed_counts` | 泊松采样后的非负探测器计数 | **推荐输入**，保留计数统计；训练时再做归一化 |
| `exit_wave_real`, `exit_wave_imag` | 同一样本的复数出口波两通道 | **监督标签** |
| `ideal_intensity` | 无探测器计数噪声、归一化的理想衍射强度 | 前向一致性检查；不是实际输入 |
| `observed_log` | 便于显示的对数归一化帧 | 可视化；不要与原始计数混用 |
| `support` | 扩展晶体配置下的全视场占位掩膜 | **不是精确物体边界**，不宜当作真实 support 标签 |
| `metadata_json` | 材料、种子、束形、剂量等逐帧参数 | 分析和可复现性 |
| `valid` | 是否完成写入 | 应全部为 `true` |

这里的“出口波”是电子束经探针照明并与晶体相互作用后的复数波场，不是独立于探针的材料静电势、原子坐标或直接拍到的实空间图像。`ideal_intensity` 是该出口波傅里叶变换模平方的归一化结果；`observed_counts` 在其基础上模拟有限剂量的计数过程。

## 完成后核验

在生成器目录执行 `python build_dataset.py --prepare-only`，可重新审计已生成 HDF5 并生成/更新划分和报告，**不重新计算模拟帧**。正式用于训练前确认：

- `quality/validation_report.json` 中 `passed=true`、`sample_count=10000`、`failures=[]`；
- `metadata/dataset_summary.json` 中 `quality_passed=true`，`training_counts` 为 `train=8500`、`val=500`、`test=1000`；
- 训练程序读取 `manifests/splits_training_with_val.json`，输入字段为 `observed_counts`，标签为 `exit_wave_real` 与 `exit_wave_imag`。

## 适用范围和限制

六种材料在训练、验证、测试三组中都会出现，测试集评估的是**同源材料与相近生成条件下的泛化**，不是未见材料泛化。此数据为合成数据，尚未用目标仪器的真实衍射帧标定部分相干性、束形、液体/支撑背景、探测器响应或采集漂移。脉宽、脉冲间隔和弛豫时间在此配置中固定，不构成动态序列建模。傅里叶配对质检通过只证明内部数学一致与数据完整，不能单独证明与真实测量分布一致。
