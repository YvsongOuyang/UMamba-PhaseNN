# Au5nm 当前训练代码整理

服务器目录：

`/home/oyys/code/UMamba-AutoPhaseNN/high_strain_CNN/ecdi_training/au5nm_bundle_20261028`

数据目录：

`/data_ssd/oyys/eCDI/external_test/au5nm_ready_512`

其中包含严格配对的 `realspace.npy` 和 `diffraction.npy`，形状为 `[1,512,512]`。

## 文件用途

- `prepare_au5nm_ready_server.py`：将原始 579×596 复数场缩放为 512×512，并重新计算配对衍射强度。
- `fix_au5nm_shapes_server.py`：补齐 NPY 的样本维度，变成 `[1,512,512]`。
- `config_au5nm_ready_512.json`：当前数据配置。
- `train_au5nm_single.py`：只训练 refiner 的单样本微调程序。
- `fit_au5nm_phasecal.py`：phase+refiner 联合拟合程序；当前已改为直接优化最终复数 MAE。
- `inspect_au5nm_npz.py`：只读检查模型输出范围。

## 当前实验权重

- `au5nm_ready_ft_20261028_b2`：refiner 微调。
- `au5nm_direct_complex_fit_20261028_b9`：当前复数 MAE 最好的实验，但尚未达到 PSNR>20 dB、MAE<0.1。

## 当前结论

当前模型在这组 Au5nm 数据上的瓶颈是空间结构适配，不是简单训练轮数不足。原始六材料模型和 Au5nm 适配实验的权重均保留，没有覆盖。
