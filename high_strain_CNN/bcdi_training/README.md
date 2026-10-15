# BCDI 512×512 两阶段训练

本目录专门适配 `deliverables/bcdi_2d_simulator_512` 生成的二维高斯畴模拟数据。
所有命令在 `high_strain_CNN` 中运行。模型和训练核心复用当前 `ecdi_training`
及 `pytorch_autophasenn` 的实现，避免复制两套网络和优化器；请保留整个父项目。
本目录不修改模拟器，也不改变另一套 e-CDI HDF5 数据流程。

```text
bcdi_training/
  prepare_data.py       NPZ转换、seed分组、逐样本物理校验
  validate_data.py      数据哈希、集合隔离及抽样FFT检查
  model.py             已有2D模型的统一导入入口
  train.py             单独训练阶段1或阶段2，支持续训
  evaluate.py          初始/增强重建的成对评价和源样本追踪
  visualize.py         实空间幅值、相位、衍射空间、复数误差图
  run_pipeline.py      自动按顺序执行所有阶段
  configs/bcdi_2d.json  数据及训练参数模板
  scripts/train_bcdi_2d.sh  Linux后台启动
  tests/               数据物理约定与端到端接口测试
```

## 数据约定

输入是一次生成任务的**输出目录**，其中必须有 `manifest.json` 和 `synthetic_data/`。
模拟器代码目录本身不是数据目录；附带示例的实际输入路径是：

```text
D:/code/PYTHON/UMamba-PhaseNN/deliverables/bcdi_2d_simulator_512/examples/demo512
```

模拟器保存 `object_modulus`、`object_phase`、`modulus_detector`，最后一个是模长，
不是强度。按manifest记录的相位约定构造标签：

- `author`：`O = A * exp(2*pi*i*stored_phase)`；真实物理相位是 `angle(O)`。
- `radians`：`O = A * exp(i*stored_phase)`。

不能把 `author` 数据的 `stored_phase` 直接当作监督目标的物理相位，否则与衍射不配对。
转换程序保留生成器的float32计算顺序，校验每个样本的
`abs(fftshift(fft2(ifftshift(O))))` 与保存的模长的相对L2误差不超过 `2e-5`。
512为偶数，两种中心shift写法与该模拟器一致。源文件先校验生成manifest中的SHA256。

按**随机场seed**分组划分，默认约80%/10%/10%，划分随机种子42。
同一个seed下所有位置的样本只进入一个集合；至少需要3个seed。
舍入和每集合至少一个seed会影响小数据集的比例。现有示例3个seed、共12样本，
划分为4训练/4验证/4测试，仅供功能检查。正式实验需先生成更多独立seed。

转换结果：

```text
<DATA_DIR>/
  train_diff.npy, train_real.npy
  val_diff.npy, val_real.npy
  test_diff.npy, test_real.npy
  dataset_manifest.json      每个样本的源seed/文件/位置、所有文件哈希
  training_config.json       后续训练实际使用的配置，root为便于迁移的相对路径
```

数组是标准带头部NPY，形状 `[N,512,512]`。衍射为float32模长，物体为complex64，
每样本约3 MiB，10,000样本约29.3 GiB，另加少量索引。原NPZ保留则另占约48.8 GiB。
转换一次最多加载一个seed文件，训练采用只读memmap，由4个worker按需读取；
FFT、归一化、居中在选定训练设备上处理，不将整个数据集放入内存或显存。
转换目标必须是源目录以外的空目录，不会覆盖已有数据。

## 准备数据

本机示例（PowerShell，在本项目 `high_strain_CNN` 中）：

```powershell
python -m bcdi_training.prepare_data `
  --source-dir 'D:\code\PYTHON\UMamba-PhaseNN\deliverables\bcdi_2d_simulator_512\examples\demo512' `
  --output-dir 'D:\dataset\bcdi_2d_512'
```

Linux服务器，将实际生成任务目录替换为服务器位置：

```bash
python -m bcdi_training.prepare_data \
  --source-dir /data_ssd/oyys/bcdi_2d/simulation_run \
  --output-dir /data_ssd/oyys/bcdi_2d/dataset
```

运行前可编辑 `configs/bcdi_2d.json` 的训练参数；转换会将其写入实际数据目录中的
`training_config.json`。转换后需要调整学习率或epoch，可以编辑这份实际配置，
或使用下方训练参数覆盖。不要改变数据manifest、数组及相位约定。

## 训练定义

1. U-Net从头训练：`minmax(log1p(modulus_detector**2))` 输入，居中复数物体的
   FFT2相位作标签，沿用WCA损失。默认Adam、LR=1e-3、plateau调度、100 epoch。
2. 加载最佳验证U-Net并冻结权重和BN，以测量模长+预测相位经inverse FFT2构造
   初始物体，仅训练复数MoMamba。默认Adam、LR=2e-4、cosine调度、50 epoch，
   沿用RMS归一化、全局相位/twin对齐后的实部MAE+虚部MAE。

两阶段均使用float32/complex64，梯度裁剪1.0，默认不加物理loss、不做硬support投影。
U-Net是已有的BN、无最外层skip、1024瓶颈变体（13.11M参数）。
默认batch=1、worker=4；尚无本数据的CUDA显存基准，确认显存余量后再增大batch。
100/50轮是初始预算，并不代表新数据已经收敛。

正式阶段2必须使用官方Mamba，Linux CUDA环境按当前项目安装：

```bash
python -m pip install -r bcdi_training/requirements.txt
python -m pip install --no-build-isolation -r ecdi_training/requirements/mamba.txt
```

### 一键后台运行

```bash
cd /home/oyys/code/UMamba-AutoPhaseNN/high_strain_CNN
bash bcdi_training/scripts/train_bcdi_2d.sh
```

顺序为：数据哈希/seed/FFT预检 → U-Net → 冻结U-Net训练MoMamba → 独立测试集评价 → 绘图。
任一步失败就停止，训练和checkpoint选择不使用test。
已有数据目录由环境变量覆盖，batch也可分别设置：

```bash
DATA_DIR=/data_ssd/oyys/bcdi_2d/dataset \
PHASE_BATCH_SIZE=4 REFINER_BATCH_SIZE=1 NUM_WORKERS=4 GPU_ID=0 \
bash bcdi_training/scripts/train_bcdi_2d.sh
```

控制器也支持 `--phase-epochs`、`--refiner-epochs`、`--phase-lr`、`--refiner-lr`。
只规划、不启动训练：

```bash
python -m bcdi_training.run_pipeline --run-name bcdi_plan --dry-run
```

极小规模运行用于检查服务器环境：

```bash
bash bcdi_training/scripts/train_bcdi_2d.sh \
  --phase-epochs 1 --refiner-epochs 1 \
  --num-samples-train 4 --num-samples-val 4 --num-samples-test 4
```

### 单独训练与续训

```bash
python -u -m bcdi_training.train --stage phase --run-name bcdi_phase

python -u -m bcdi_training.train --stage refiner --run-name bcdi_refiner \
  --phase-checkpoint /data_ssd/oyys/bcdi_2d/checkpoints/bcdi_phase/checkpoint_best.pt

python -u -m bcdi_training.train --stage refiner --run-name bcdi_refiner \
  --resume /data_ssd/oyys/bcdi_2d/checkpoints/bcdi_refiner/checkpoint_last.pt --epochs 100
```

`--epochs` 表示最终轮数，非追加轮数。所有入口都可用 `--data-config <training_config.json>`
指定非默认数据位置。checkpoint必须来自相同的模拟生成任务与seed划分。
已有checkpoint的目录不能无提示覆盖；3D或另一数据域的checkpoint不会误用。

## 结果

自动流程的日志、配置、历史和图保存在：

```text
high_strain_CNN/artifacts/training/bcdi_2d/<RUN>/
  launcher.log, pipeline.log, pipeline.pid, pipeline_status.json
  data_validation.json, pipeline_config.json
  stages/<RUN>_phase/, stages/<RUN>_refiner/
  evaluation/metrics.json, per_sample.csv, sample_*.npz, visualizations/

/data_ssd/oyys/bcdi_2d/checkpoints/
  <RUN>_phase/checkpoint_{best,last}.pt
  <RUN>_refiner/checkpoint_{best,last}.pt
```

默认只保留best/last两份checkpoint；大文件在SSD，项目内保留轻量记录。
`tail -f <RUN>/launcher.log` 查看后台进度和每阶段ETA。
恢复自动流程第二阶段：

```bash
python -m bcdi_training.run_pipeline --run-name <原RUN> --start-stage refiner \
  --refiner-resume /data_ssd/oyys/bcdi_2d/checkpoints/<原RUN>_refiner/checkpoint_last.pt
```

单独评估/绘图：

```bash
python -m bcdi_training.evaluate \
  --checkpoint /data_ssd/oyys/bcdi_2d/checkpoints/<RUN>_refiner/checkpoint_best.pt \
  --output-dir artifacts/evaluations/bcdi_2d/test

python -m bcdi_training.visualize --evaluation-dir artifacts/evaluations/bcdi_2d/test
```

评价初始U-Net与增强后同一批样本的complex NRMSE、幅值PSNR/SSIM、实空间相位误差、
WCA、Fourier模长误差、有效support IoU/Dice，输出成对改善比例及逐样本seed/位置。
阈值默认0.3，仅用于support诊断/相位统计范围/展示，不硬裁剪模型输入或输出。
实空间指标对齐平移、全局相位、尺度和twin；衍射模长指标用完整物体，不施加真值support。
初始物体使用测量模长，故其Fourier模长误差天然接近0，不能据此宣称相位正确。
该数据为无噪声二维简化高斯畴模型，这里的结果不等同于真实三维BCDI或电子e-CDI结果。
