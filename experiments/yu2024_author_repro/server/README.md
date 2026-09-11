# Yu et al. 2024：服务器修正版

此目录提供单 GPU 的合成数据监督学习实验。直接导入作者 `Supervised training/model.py`、`complexNN.py`，不更改网络结构、Xavier 初始化、复数运算或实/虚部 L1 目标；生成器保留原作者几何及随机场函数，仅修正相位指数。新增入口将路径、训练循环和运行管理整理为可在 Linux/Windows 执行的脚本。

上游：[XFELDataScience/Complex-NNphase](https://github.com/XFELDataScience/Complex-NNphase)，commit `bcccdef38ca7d0c35699036ceee629fecdfc76f7`；论文：[npj Computational Materials 10, 34 (2024)](https://doi.org/10.1038/s41524-024-01208-7)。保留作者署名，源码来源和哈希见上级 `source_provenance.json`。这是一份修正后的复现实验，不声称就是作者训练时的私有版本，也不声称已经达到论文指标。

## 修正范围

| 项目 | 公开代码 | 本入口 |
|---|---|---|
| 相位 | 标签是弧度，前向却用 `exp(2j*pi*phase)` | 生成器使用 `exp(1j*phase)`，监督标签保持弧度 |
| 输入振幅 | 已保存 `abs(FFT)`，训练又开平方 | 直接使用保存的振幅；仍按作者方式复制到实部、虚部 |
| 归一化 | 训练、测试分别求全局最大值 | 只用 12000 训练样本拟合一个最大值；验证、测试复用，不裁剪 |
| Adam | `10 * 0.001` | 初始学习率 `0.001` |
| cosine | `T_max=500`，每 batch step | 每 epoch step，`T_max=500`，最小学习率 `0.0001` |
| loss 日志 | 除以最后一个 batch 的索引 | 按实际样本数加权平均；优化目标不变 |
| 数值检查 | 原脚本无检查 | 每步检查 loss、梯度、模型参数、Adam 两个动量；异常报错退出 |
| checkpoint | 覆盖裸模型权重 | 原子保存 `last.pt`、验证集最优 `best.pt`，含优化器、调度器、随机状态和配置 |

最小学习率 `0.0001` 是本复现的明确选择，论文主文未给出该值。没有添加 AMP、梯度裁剪、全局相位对齐或额外物理损失。实/虚复制仍与作者相同，不额外乘除 `sqrt(2)`。

## 1. 获取代码和环境

以下在 Linux Bash 执行；每一步成功后再执行下一步。已有工作目录时可另建一个克隆以避免覆盖其它实验。

```bash
git clone --branch codex/yu2024-server-fixes --single-branch \
  https://github.com/YvsongOuyang/UMamba-PhaseNN.git UMamba-PhaseNN-yu2024
cd UMamba-PhaseNN-yu2024

conda create -n yu2024 python=3.10 -y
conda activate yu2024
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r experiments/yu2024_author_repro/server/requirements.txt
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda") + 1)'
python experiments/yu2024_author_repro/server/test_pipeline.py
```

上述 PyTorch/torchvision 对应本地已检查版本；Linux wheel 可从 [PyTorch 官方索引](https://download.pytorch.org/whl/cu128/torch/) 获取。需要支持该 CUDA wheel 的 NVIDIA 驱动。若服务器已有可用的匹配 torch/torchvision 环境，可直接安装 requirements 并运行检查。旧卡如 V100 可使用 [官方历史版本](https://pytorch.org/get-started/previous-versions/) 中的 `torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118`，该组合未在本机实测，务必先通过下述预跑。不要仅依据 `nvidia-smi` 顶部的 CUDA 字样混装不同版本。

## 2. 生成数据并检查

将 DATA 改成服务器有写权限的数据盘目录。Linux 没有 `E:` 盘；Windows 可以传 `--data-dir 'E:\dataset\Yu2024_ComplexNNphase_corrected'`。

```bash
export DATA=/data/dataset/Yu2024_ComplexNNphase_corrected
export RUN="$PWD/experiments/yu2024_author_repro/runs/server_corrected_500"
export ENTRY="$PWD/experiments/yu2024_author_repro/server"

python "$ENTRY/data.py" generate --data-dir "$DATA" --seeds 100 --samples 15000 --seed 1
```

100 个 seed 生成 19600 个 64×64 原始样本，约 1.5 GiB。数字文件顺序取前 15000 个参与实验，剩余 4600 个保留但不使用。前 13500 个先按 sklearn seed=0 打乱、再按 torch seed=1 分为 12000 训练与 1500 验证；最后 1500 个测试。划分比例依据主文，补充材料的 150000 样本描述与主文不一致。本划分明确按样本进行，未额外改成按生成 seed 分组划分。

生成命令检查剩余空间，并自动校验全部 19600 个样本的傅里叶对应关系、数组有限性、文件哈希和划分。成功后目录含 `data0.npz`…`data99.npz`、`manifest.json`、`split_indices.npz`、`verification.json`。建议数据盘预留至少 2 GiB，运行盘至少 1 GiB，主机内存至少 8 GiB（16 GiB 更宽裕），workers 默认 0 避免复制大数组。

**旧的原始数据不能直接用于这个修正版。** 必须生成到新空目录。程序拒绝混入旧数据或覆盖已有目录。若生成被中断，换一个空目录重新生成；本入口没有生成断点恢复功能。

如数据已生成，只需再次校验，不要重新运行 generate：

```bash
python "$ENTRY/data.py" verify --data-dir "$DATA"
```

## 3. 先预跑 5 轮，检查硬件并估时

```bash
python "$ENTRY/train.py" --data-dir "$DATA" --run-dir "$RUN" \
  --epochs 500 --stop-after 5 --batch-size 64 --lr 0.001 --min-lr 0.0001 \
  --seed 1 --device cuda:0 --num-workers 0
cat "$RUN/status.json"
```

此时应显示 `paused`、`completed_epochs: 5`。`--epochs 500` 决定完整学习率日程，`--stop-after 5` 仅暂停；不要改成 `--epochs 5` 再接续 500 轮。

估算 500 轮总训练时间（预跑前一轮预热不计）：

```bash
python - "$RUN" <<'PY'
import json, pathlib, statistics, sys
rows = [json.loads(s) for s in (pathlib.Path(sys.argv[1]) / 'metrics.jsonl').read_text().splitlines()]
seconds = statistics.median(r['seconds'] for r in rows[1:] or rows)
print(f'Median: {seconds:.2f} s/epoch; 500 epochs: {seconds*500/3600:.2f} h; remaining: {seconds*(500-rows[-1]["epoch"])/3600:.2f} h')
PY
```

此前 RTX 5070 原始代码完整运行约 2 小时 33 分钟。修正版新增每步检查和 checkpoint I/O，服务器性能也不同，以上预跑估时更可靠，不把此前耗时作为保证。

## 4. 续跑到 500 轮

```bash
nohup python -u "$ENTRY/train.py" --data-dir "$DATA" --run-dir "$RUN" \
  --epochs 500 --batch-size 64 --lr 0.001 --min-lr 0.0001 \
  --seed 1 --device cuda:0 --num-workers 0 \
  --resume "$RUN/last.pt" > "$RUN/console.log" 2>&1 &
echo $! > "$RUN/train.pid"
tail -f "$RUN/console.log"
```

Ctrl+C 退出 `tail` 不会停止后台训练。中断后再次执行同一个续训命令即可；先确认旧训练进程已退出，避免两个进程写同一目录。`--resume` 恢复至最近完成的 epoch，不恢复一轮中间的 batch。恢复时 epochs、lr、min-lr、batch-size、seed、workers、数据哈希必须一致；不要载入旧发散模型。不同 GPU/CUDA 版本之间不保证逐位一致。

查看进度：

```bash
cat "$RUN/status.json"
tail -n 5 "$RUN/metrics.jsonl"
```

`state: complete` 表示 500 轮计算结束；`failed` 表示异常；训练中记录 `epoch`。日志 `val` 应与 `zero`（全零预测的 MAE）一起看，`amp_ratio` 是预测/真实振幅均值比。持续接近全零基线、振幅比很小，代表重建仍可能失败，不能把有限 loss 或完成 500 轮当作复现成功。

## 5. 完成后执行测试集评估

训练脚本不自动消费测试集；训练完成后运行：

```bash
python "$ENTRY/evaluate.py" --data-dir "$DATA" --checkpoint "$RUN/best.pt" \
  --output-dir "$RUN/test_best" --device cuda:0
cat "$RUN/test_best/test_metrics.json"
```

输出 `test_metrics.json` 和 `test_predictions.npz`（1500 个预测复数图、样本 ID、逐样本 MAE/χ²）。χ² 使用 float64/complex128 计算，避免此前的 FP32 平方溢出；定义为衍射**振幅**的归一化平方误差。没有做全局相位、平移或共轭对齐，指标不应直接当作论文全部指标。

论文保存末轮模型；若要同时报告严格末轮结果，再运行：

```bash
python "$ENTRY/evaluate.py" --data-dir "$DATA" --checkpoint "$RUN/last.pt" \
  --output-dir "$RUN/test_last" --device cuda:0
```

## 可选：8 样本拟合诊断

```bash
python "$ENTRY/train.py" --data-dir "$DATA" --run-dir "${RUN}_overfit8" \
  --epochs 500 --overfit-samples 8 --device cuda:0
```

仅使用训练集前 8 个样本，同时在相同样本上检查拟合能力；不是验证集成绩，不进入正式实验。评估入口会拒绝把此诊断 checkpoint 用作测试模型。正式运行无需先执行这一项。

## 还需要哪些东西

合成数据实验只需要上述环境、生成/校验、训练和评估步骤，不需要下载作者预训练模型。本仓库没有找到作者提供的预训练权重或原始实验数据下载。真实 XFEL 实验微调需要先向作者获取真实数据并确认预处理，当前服务器入口没有复现那一部分；也没有实现完整 R-CNN 对照、噪声鲁棒性及论文全部 SSIM 图表。

本地验收结果见 [VALIDATION.md](VALIDATION.md)。
