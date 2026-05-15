import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchinfo import summary

# 假设这些是你本地的文件
from PyTorch.data_loader1 import Dataset
from AutoPhaseNN_model_relu import Network

# --- 全局配置与损失函数定义 ---
# （保持你定义的 loss_log, loss_sq, loss_mae, loss_paper, loss_pcc, loss_comb 不变）
# 这里省略重复的函数定义，实际运行时请保留在脚本中
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  

import torch
import torch.nn as nn

def loss_log(Y_true, Y_pred):
    # 使用 log10(x + 1)
    pred = torch.log10(Y_pred + 1.0)
    true = torch.log10(Y_true + 1.0)

    top = torch.sum(torch.pow(pred - true, 2))
    bottom = torch.sum(torch.pow(true, 2))
    
    loss_value = top / (bottom + 1e-8) # 防止除零
    return loss_value

def loss_sq(Y_true, Y_pred):
    # 针对 3D 数据 (Batch, Channel, H, W, D)，对空间维度求和
    # 如果 Channel 只有 1，dim=(1,2,3,4)；如果输入是 (B, H, W, D)，dim=(1,2,3)
    dims = tuple(range(1, Y_true.ndim)) 
    
    top = torch.sum(torch.pow(Y_pred - Y_true, 2), dim=dims, keepdim=True)
    bottom = torch.sum(torch.pow(Y_true, 2), dim=dims, keepdim=True)

    loss_value = torch.sum(top / (bottom + 1e-8))
    return loss_value

def loss_mae(Y_true, Y_pred):
    dims = tuple(range(1, Y_true.ndim))
    
    top = torch.sum(torch.abs(Y_pred - Y_true), dim=dims, keepdim=True)
    bottom = torch.sum(torch.abs(Y_true), dim=dims, keepdim=True)
    
    loss_value = torch.sum(top / (bottom + 1e-8))
    return loss_value

def loss_paper(Y_true, Y_pred):
    # Y_true: 测量衍射强度 (Im)，Y_pred: 估计衍射强度 (Ie)
    # 根据论文公式：Sum|sqrt(Ie) - sqrt(Im)| / N^3 [cite: 192, 194]
    sqrt_true = torch.sqrt(torch.clamp(Y_true, min=0.0))
    sqrt_pred = torch.sqrt(torch.clamp(Y_pred, min=0.0))
    
    abs_error = torch.abs(sqrt_pred - sqrt_true)
    total_error = torch.sum(abs_error)
    
    # N^3 = 64x64x64 = 262144 [cite: 194]
    loss_value = total_error / 262144.0
    return loss_value

def loss_pcc(Y_true, Y_pred):
    dims = tuple(range(1, Y_true.ndim))
    
    # 计算均值
    pred_mean = torch.mean(Y_pred, dim=dims, keepdim=True)
    true_mean = torch.mean(Y_true, dim=dims, keepdim=True)
    
    # 中心化
    pred_centered = Y_pred - pred_mean
    true_centered = Y_true - true_mean
    
    top = torch.sum(pred_centered * true_centered, dim=dims, keepdim=True)
    
    pred_var_sum = torch.sum(torch.pow(pred_centered, 2), dim=dims, keepdim=True)
    true_var_sum = torch.sum(torch.pow(true_centered, 2), dim=dims, keepdim=True)
    
    bottom = torch.sqrt(pred_var_sum * true_var_sum + 1e-8)
    
    # PCC 损失 = Sum(1 - Pearson相关系数)
    loss_value = torch.sum(1.0 - (top / bottom))
    return loss_value

def loss_comb(Y_true, Y_pred):
    l1 = loss_sq(Y_true, Y_pred)
    l2 = loss_pcc(Y_true, Y_pred)
    return (l1 + l2) / 2.0

def loss_comb2(Y_true, Y_pred):
    l1 = torch.sqrt(loss_sq(Y_true, Y_pred) + 1e-8)
    l2 = loss_pcc(Y_true, Y_pred)
    return (l1 + l2) / 2.0

def loss_comb_log(Y_true, Y_pred):
    l1 = loss_sq(Y_true, Y_pred)
    l2 = loss_pcc(Y_true, Y_pred)
    l3 = loss_log(Y_true, Y_pred)
    
    a1, a2, a3 = 50.0, 50.0, 1.0
    return (a1*l1 + a2*l2 + a3*l3) / (a1 + a2 + a3)

def main():
    # 1. 环境初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 24
    scale_I = 1
    MODEL_SAVE_PATH = './AutoPhase/SN/'
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    NGPUS = torch.cuda.device_count()
    BATCH_SIZE = 16 
    LR = NGPUS * 1e-5
    
    # 2. 数据准备
    data_path = '/data_hdd/oyys/autophaseNN/CDI_simulation_upsamp_noise/'
    dataname_list = os.path.join(data_path, '3D_upsamp.txt')
    
    with open(dataname_list, 'r') as f:
        filelist = [line.strip().split('/')[-1] for line in f.readlines()]

    TRAIN_ratio = 0.9
    total_files = len(filelist)
    split_idx = int(total_files * TRAIN_ratio)
    
    # 3. 物理切分列表 (关键步骤：互不重叠)
    #train_filelist = filelist[:split_idx]  # 前 90%
    #val_filelist = filelist[split_idx:]    # 后 10%

    # num_workers 建议设为 NGPUS*4，设置 64 往往会导致系统卡死
    train_dataset = Dataset(filelist, data_path, load_all=False, ratio=TRAIN_ratio, dataset='train', scale_I=scale_I)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NGPUS*4, drop_last=True)

    validation_dataset = Dataset(filelist, data_path, load_all=False, ratio=TRAIN_ratio, dataset='validation', scale_I=scale_I)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NGPUS*4, drop_last=False)


    print(f"总文件数: {total_files}")
    print(f"训练集数量: {len(train_dataset)}")
    print(f"验证集数量: {len(validation_dataset)}")

    # 3. 模型配置
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', type=int, default=64)
    parser.add_argument('--T', type=float, default=0.1)
    parser.add_argument('--nconv', type=int, default=32)
    parser.add_argument('--use_down_stride', type=bool, default=False)
    parser.add_argument('--use_up_stride', type=bool, default=False)
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--unsupervise', type=bool, default=True) 
    parser.add_argument('--scale_I', type=int, default=1) 
    argv = parser.parse_args([])

    model = Network(argv).to(device)

    model_path = "./torch_transferred_weights_final.pth"

    params = torch.load(model_path, map_location=torch.device("cpu"))

    model.load_state_dict(params)
    
    # 4. 优化器与调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    iterations_per_epoch = len(train_loader)
    step_size = 6 * iterations_per_epoch
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LR, 
                                                  max_lr=LR*2, step_size_up=step_size,
                                                  cycle_momentum=False, mode='triangular2')

    # 5. Shrink-wrap 参数
    INIT_SW = 0.07
    FINAL_SW = 0.1
    CONST_EPOCHS = 0
    SW_INCREMENT = (FINAL_SW - INIT_SW) / (EPOCHS - 1) if EPOCHS > 1 else 0
    model.sw_thresh = INIT_SW

    # 6. 训练循环
    metrics = {'losses':[], 'val_losses':[], 'lrs':[], 'best_val_loss': np.inf}

    print("Starting Training...")
    for epoch in range(EPOCHS):
        # 训练
        train(train_loader, metrics, model, optimizer, scheduler, device)
        
        # 验证
        validate(validation_loader, metrics, model, device, MODEL_SAVE_PATH)
        
        l = metrics['losses'][-1]
        lv = metrics['val_losses'][-1]

        print(f'Epoch: {epoch} | FT Train/Val: {l[0]:.5f}/{lv[0]:.5f} | SW: {model.sw_thresh:.4f} | LR: {metrics["lrs"][-1][0]:.6f}')

        # 更新 Shrink-wrap 阈值
        if epoch >= (CONST_EPOCHS - 1):
            model.sw_thresh += SW_INCREMENT

    print("Training Complete. Starting Visualization...")
    # 这里可以添加你最后的绘图逻辑...

# --- 核心训练/验证函数（修改了参数传递） ---

from tqdm import tqdm  # 确保头部导入了 tqdm

def train(trainloader, metrics, model, optimizer, scheduler, device):
    model.train()
    loss_ft_cum, loss_amp_cum, loss_ph_cum = 0.0, 0.0, 0.0
    criterion = nn.L1Loss() # 辅助监控用

    # 1. 使用 tqdm 包装 trainloader
    # desc: 进度条前缀文字
    # leave: False 表示跑完后清除进度条，防止输出刷屏（如果希望保留设为 True）
    pbar = tqdm(trainloader, desc="Training", leave=False)

    for ft_images, amps, phs in pbar:
        ft_images, amps, phs = ft_images.to(device).float(), amps.to(device).float(), phs.to(device).float()

        optimizer.zero_grad()
        psi, obj, pred_amps, pred_phs, support = model(ft_images)
        
        # 能量对齐
        scale = torch.sum(ft_images) / (torch.sum(psi) + 1e-10)
        psi_scaled = psi * scale

        # 计算 Loss
        loss = loss_comb(ft_images, psi_scaled)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        metrics['lrs'].append(scheduler.get_last_lr())
        
        # 记录数值
        current_loss = loss.item()
        loss_ft_cum += current_loss
        loss_amp_cum += criterion(pred_amps, amps).item()
        loss_ph_cum += criterion(pred_phs * support, phs).item()
        
        # 2. 实时更新进度条后缀，显示当前 Batch 的 Loss
        pbar.set_postfix({'loss': f'{current_loss:.4f}'})
        
    n = len(trainloader)
    metrics['losses'].append([loss_ft_cum/n, loss_amp_cum/n, loss_ph_cum/n])

def validate(validloader, metrics, model, device, save_path):
    model.eval()
    val_ft, val_amp, val_ph = 0.0, 0.0, 0.0
    criterion = nn.L1Loss()
    
    with torch.no_grad():
        for ft_images, amps, phs in validloader:
            ft_images, amps, phs = ft_images.to(device).float(), amps.to(device).float(), phs.to(device).float()
            psi, _, pred_amps, pred_phs, support = model(ft_images)
            
            scale = torch.sum(ft_images) / (torch.sum(psi) + 1e-10)
            psi_scaled = psi * scale
            
            val_ft += loss_comb(ft_images, psi_scaled).item()
            val_amp += criterion(pred_amps, amps).item()
            val_ph += criterion(pred_phs * support, phs).item()
            
    n = len(validloader)
    avg_val_loss = val_ft / n
    metrics['val_losses'].append([avg_val_loss, val_amp/n, val_ph/n])
    
    if avg_val_loss < metrics['best_val_loss']:
        metrics['best_val_loss'] = avg_val_loss
        torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))

# --- 运行入口 ---
if __name__ == '__main__':
    main()