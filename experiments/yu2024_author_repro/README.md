# Yu et al. 2024 作者代码复现

**2026-09-11：服务器请使用 [server/README.md](server/README.md) 中的修正版命令。**
该入口直接导入 `vendor` 中的作者网络，修复已确认的数据/训练问题，提供生成、校验、训练、续训、评估命令。
此前原样运行的 500 轮实验发生数值发散；下面是历史记录，不是推荐的服务器运行方式。

## 历史：作者代码直接运行（2026-09-10）

按用户要求，本实验直接执行作者原始生成器，并复用原始网络及训练函数。与先前 `yu2024_complex_nn` 的独立实现分开。

来源：https://github.com/XFELDataScience/Complex-NNphase ，下载版本 commit `bcccdef38ca7d0c35699036ceee629fecdfc76f7`。`vendor/Complex-NNphase` 是作者源码本地副本；`runtime/model.py`、`complexNN.py`、`untils.py` 与作者文件字节一致。源码副本用于本地复现，不主张作者代码的再许可。

## 数据

原始生成器不作修改：100 个 seed，1024×1024 画布，每个输出 196 个 64×64 样本，总计 19,600。包括作者原始相位公式、域尺寸 6×8、σ=(1.5,3)、域旋转 −25°、束斑 σ=(9,30)/−35°；不作物理表达式修正。

数据路径：`E:\dataset\Yu2024_ComplexNNphase_author\synthetic_data`。总数据约 2 GiB 以内，具体取决于 NumPy FFT 输出 dtype。生成结束会验证全部数组并写入 `manifest.json`，记录各文件 SHA256、dtype、数量及耗时。

本次按论文**主文**采用 15,000 个样本、80/10/10，即 12,000 训练、1,500 验证、1,500 测试。具体操作为按 seed 数字顺序取前 15,000 个；前 13,500 构成训练/验证池，保留作者的 sklearn shuffle(random_state=0) 后 torch.random_split；最后 1,500 个为测试。random_split 使用 seed=1，清单保存为运行目录 `split_indices.npz`。全部 19,600 原始样本仍保留。

注意：作者仓库训练脚本本身是前 80% 中扣 1000 验证、后 20% 测试，与论文 80/10/10 不一致；这里仅为执行用户指定的论文划分而调整相应参数。补充材料另有 150,000 样本描述，本次采用主文 15,000，不混用。

## 最小运行适配

完整差异保存在 `runtime_changes.patch`，由 `make_runtime.py` 生成：

1. 修复未定义的 `BATCH_SIZE`，改用已有参数 `args.batch_size`。
2. Windows 下 workers=0，防止模块顶层加载逻辑被 spawn 重复执行。
3. 数据数量和划分按上述主文设置；文件顺序固定，保存实际样本索引；启用已有 seed 参数，CPU threads=4。
4. 加入每轮计时、日志输出位置、非有限损失停止检测及附加 checkpoint。作者原有裸权重仍保存。原始损失汇总公式不改，日志存在除以末批索引而非批次数的轻微偏差；最后测试另作按样本统计。
5. 作者数学和训练算法不变：保留 `sqrt(modulus)`、实虚复制、训练/测试各自全局最大值归一化、实虚 L1；Adam 初始 lr=10×0.001=0.01；cosine 的 eta_min=0.001、T_max=500，并按作者代码每 batch step；没有引入 AMP、梯度裁剪或相位对齐。

训练轮数设为论文的 500、batch size=64；网络保留源码的双卷积瓶颈（3,794,626 参数）。因为 RTX 5070 不适用作者原始 PyTorch 1.8/CUDA 11.1 环境，本次使用已有 PyTorch 2.11.0+cu128 / NumPy 2.2.6；依赖版本和输入归一化最大值会保存到 `run_config.json`。没有声称数值结果与旧硬件/旧依赖逐位一致。

## 运行与状态

```powershell
$py = 'C:\Users\30810\.conda\envs\spikingformer\python.exe'
Set-Location D:\code\PYTHON\UMamba-PhaseNN\experiments\yu2024_author_repro
& $py prepare_dataset.py --output E:\dataset\Yu2024_ComplexNNphase_author
& $py make_runtime.py
& $py run_full.py --data E:\dataset\Yu2024_ComplexNNphase_author\synthetic_data --output runs\author_full_500 --epochs 500 --batch-size 64
```

现有数据或运行目录非空时拒绝覆盖。`run_full.py` 自动设置本实验依赖搜索路径，执行作者训练，保存 `training.log`、`timings.jsonl`、`status.json`，并在正常结束后运行独立的只读测试评估。`status.json` 包含 completed_epochs、近期每轮耗时中位数和预计剩余时间。

`Real_complex_AE_6_5.pth` 是作者格式末轮权重；`resume.pt` 每 10 轮附加保存模型/优化器/调度器/RNG，当前脚本不提供自动恢复入口。`test_metrics.json` 为最终测试输出，清楚区分监督标签误差和相对保存衍射幅值的 χ²。

本次“完整跑完”的范围是**作者提供的合成数据监督 C-CNN 训练及测试**。真实 XFEL 无监督实验仍需作者提供原始数据和缺失工具，不能用合成样本替代后宣称完成。
