# AutoPhaseNN 最大池化层 / BN 层交换验证实验

本目录专门验证专利批注中的问题：在 AutoPhaseNN 推理阶段，将编码器内的
`BN -> MaxPool3d` 调整为 `MaxPool3d -> BN` 后，重建精度和输出分布变化有多大。

实验不会训练或修改权重。基线模型与交换模型是两个独立 `nn.Module` 实例，分别
严格加载同一份预训练参数，并接收同一批验证样本。两个模型都会从输入到最终
三维 FFT 远场输出完整执行 `forward()`，仅四个编码下采样块中的 BN/最大池化
执行次序不同。

## 重要的拓扑口径

当前仓库的 `autophasenn_training_pipeline/model_tf_compatible.py` 实际编码块为：

```text
Conv3d -> LeakyReLU -> BN -> MaxPool3d
```

本实验比较：

```text
基线：Conv3d -> LeakyReLU -> BN -> MaxPool3d
交换：Conv3d -> LeakyReLU -> MaxPool3d -> BN
```

专利正文将原拓扑写成“卷积、批量归一化、激活、池化”，与当前代码中激活层和
BN 层的先后次序不同。实验报告会保留这一事实，避免把实测数字错误归因到另一种
拓扑。该差异不影响本目录对 BN/MaxPool 这一对相邻算子的直接交换验证。

## 验证内容

每次正式运行会生成以下证据：

1. 四个池化相邻 BN 层的有效缩放因子
   `gamma / sqrt(running_var + eps)`，包括正、零、负通道数量和正值比例；
2. 在同一层、同一输入上直接比较 `Pool(BN(x))` 与 `BN(Pool(x))` 的局部交换误差；
3. 比较交换误差在后续网络中传播后的各编码块输出差异；
4. 基线与交换模型相对验证真值的衍射模量误差、幅值 PSNR、均匀窗口 3D-SSIM、
   support 指标和包裹相位误差；
5. 基线与交换模型的远场、幅值、相位和 support 输出一致性；
6. 最终复数物体输出的复模 MAE、相对 L2、最大复模差及实部/虚部 MAE；
7. 两个独立模型的最终六项输出形状、数据类型和完整前向调用记录；
8. 逐样本结果、均值、标准差以及配对 bootstrap 95% 置信区间。

BN 在推理模式下是逐通道仿射变换。当有效缩放因子为正时，它是严格单调递增的，
理论上 `Pool(BN(x)) = BN(Pool(x))`；若存在负缩放通道，该严格等价前提不成立，
实验仍会继续，但报告会将结论标记为经验近似而非数学等价。

## 运行前准备

编辑 `configs/default.yaml`，或直接用命令行覆盖以下路径：

- 预训练权重：`--checkpoint`
- 验证数据目录：`--data-dir`
- 输出根目录：`--output-dir`

默认数据格式沿用现有 AutoPhaseNN pipeline：

```text
val_diff.npy: float32, (N, 64, 64, 64)
val_real.npy: complex64, (N, 64, 64, 64)
```

## 正式运行

在仓库根目录执行：

```powershell
python experiments/autophasenn_bn_maxpool_swap/run_experiment.py `
  --config experiments/autophasenn_bn_maxpool_swap/configs/default.yaml `
  --checkpoint "D:\path\to\checkpoint_best.pt" `
  --data-dir "D:\path\to\validation_memmap" `
  --device cuda
```

Linux 服务器示例：

```bash
python experiments/autophasenn_bn_maxpool_swap/run_experiment.py \
  --config experiments/autophasenn_bn_maxpool_swap/configs/default.yaml \
  --checkpoint /path/to/checkpoint_best.pt \
  --data-dir /data_ssd/oyys/autophasenn \
  --device cuda
```

快速检查前 8 个样本可加：

```text
--limit 8
```

正式用于专利的数据不要使用 `--limit`，应按配置运行完整验证集。

## 服务器多数据集运行（推荐）

`run_multi_dataset.py` 会用同一个检查点依次评估多个 memmap 数据集。默认配置
`configs/server_multi_dataset.yaml` 已启用训练集前 5000 个样本和完整 5000 个验证
样本，并预留了一个默认关闭的测试集条目。可以继续在 `datasets` 列表中增加其他
数据集；每项均可独立设置文件名、样本数、形状、dtype 和 `scale_i`。

从 GitHub 拉取代码后，在仓库根目录运行：

```bash
python experiments/autophasenn_bn_maxpool_swap/run_multi_dataset.py \
  --config experiments/autophasenn_bn_maxpool_swap/configs/server_multi_dataset.yaml \
  --checkpoint /path/to/checkpoint_best.pt \
  --data-dir /data_ssd/oyys/autophasenn \
  --device cuda
```

先验证环境和路径时，建议在每个数据集只跑 2 个样本：

```bash
python experiments/autophasenn_bn_maxpool_swap/run_multi_dataset.py \
  --config experiments/autophasenn_bn_maxpool_swap/configs/server_multi_dataset.yaml \
  --checkpoint /path/to/checkpoint_best.pt \
  --data-dir /data_ssd/oyys/autophasenn \
  --device cuda \
  --limit 2 \
  --run-name smoke_2
```

多数据集读取仍直接使用
`autophasenn_training_pipeline.dataset.AutoPhaseDataset`，不是另外实现的数据预处理。
默认文件格式如下：

```text
train_diff.npy / val_diff.npy: float32, (N, 64, 64, 64)
train_real.npy / val_real.npy: complex64, (N, 64, 64, 64)
```

注意这些 `.npy` 文件按项目现有实现通过 `numpy.memmap` 作为无文件头的原始数组
读取。如果你的数据是通过 `numpy.save` 生成、带标准 `.npy` 文件头，需要先转换成
项目训练管线使用的 raw memmap 格式。

批量运行根目录会额外生成：

- `multi_dataset_comparison.md`：训练集、验证集等数据集的两模型效果总表；
- `multi_dataset_metrics.csv`：适合导入表格和绘图的长表；
- `multi_dataset_summary.json`：所有数据集的完整机器可读汇总；
- `<dataset>/`：每个数据集自己的逐样本 CSV、报告、BN 审计和环境记录。

效果评价分为两个口径：

- 相对真值的效果：直接沿用训练管线的 `paper_modulus_mae`、`chi2_modulus`、
  `relative_l1_modulus`、`pearson_corr`、实空间幅值、support 和包裹相位指标，另补充
  3D PSNR/SSIM；
- 两模型输出差异：直接比较两个完整 forward 的最终 `farfield_modulus`，报告 MAE、
  RMSE、最大绝对差、相对 L1/L2、Pearson 相关系数和直方图 JS 散度。

权重文件、数据文件和本地输出不会提交到 GitHub，需要在服务器上通过参数指定。

## 随机输入一致性预检

若暂时没有真实验证集，可使用固定随机种子的非负均匀随机输入检查数学前提和网络
输出一致性：

```powershell
python experiments/autophasenn_bn_maxpool_swap/run_experiment.py `
  --config experiments/autophasenn_bn_maxpool_swap/configs/random_input.yaml
```

随机模式默认生成 16 个 `64³`、取值范围 `[0, 1)` 的 float32 输入。它可以产生
BN 缩放审计、局部交换误差、传播误差和端到端输出差，但不能提供具有专利证明力的
PSNR、SSIM 或相位重建精度；报告生成器会自动阻止把随机输入结果写成重建精度结论。

## 输出文件

每次运行会在 `outputs/<时间戳>/` 下生成：

- `report.md`：可直接审阅的中文量化报告和结论；
- `final_output_comparison.md`：只比较两个完整模型的最终远场主输出；
- `all_bn_scale_audit.md`：模型内全部 BN 层的有效缩放因子正、零、负通道审计；
- `summary.json`：完整汇总、置信区间、BN 正值审计与判定结果；
- `per_sample.csv`：每个验证样本的全部指标；
- `resolved_config.json`：本次实际使用的配置；
- `environment.json`：Python/PyTorch/CUDA/GPU、权重哈希和数据文件信息；
- `run.log`：运行日志。

默认“影响很小”的工程判据写在 YAML 的 `acceptance` 段中，可以在正式运行前按
专利撰写口径调整。报告会逐项列出阈值和是否通过，不会只给一个笼统结论。

## 本地自检

无需权重和数据即可运行算子级测试：

```powershell
python -m unittest discover `
  -s experiments/autophasenn_bn_maxpool_swap/tests `
  -p "test_*.py" -v
```
