from __future__ import print_function
import os
import argparse
import time
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch.utils.tensorboard import SummaryWriter
import json

from UMambaEnc_3d import get_umamba_enc_3d_from_plans
from AutoPhaseNN_model_relu import Network
from data_loader import *
from utils import CombinedDiffractionLoss, get_criterion

from datetime import timedelta

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")

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

    loss_value = torch.mean(top / (bottom + 1e-6))
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

def loss_pcc_old(Y_true, Y_pred):
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
    loss_value = torch.mean(1.0 - (top / bottom))
    return loss_value
def loss_pcc(Y_true, Y_pred):
    dims = tuple(range(1, Y_true.ndim))
    
    pred_mean = torch.mean(Y_pred, dim=dims, keepdim=True)
    true_mean = torch.mean(Y_true, dim=dims, keepdim=True)
    
    pred_centered = Y_pred - pred_mean
    true_centered = Y_true - true_mean
    
    top = torch.sum(pred_centered * true_centered, dim=dims, keepdim=True)
    
    pred_var_sum = torch.sum(pred_centered ** 2, dim=dims, keepdim=True)
    true_var_sum = torch.sum(true_centered ** 2, dim=dims, keepdim=True)
    
    # ✅ 分别给两个方差加 epsilon，避免单独一个为 0 时梯度爆炸
    bottom = torch.sqrt(pred_var_sum + 1e-8) * torch.sqrt(true_var_sum + 1e-8)
    
    pcc = top / bottom
    # ✅ clamp 防止数值误差超出 [-1, 1]，同时截断极端梯度
    pcc = torch.clamp(pcc, -1.0 + 1e-6, 1.0 - 1e-6)
    
    loss_value = torch.mean(1.0 - pcc)
    return loss_value

def loss_comb(Y_true, Y_pred):
    l1 = loss_sq(Y_true, Y_pred)
    l2 = loss_pcc_old(Y_true, Y_pred)
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


def train(args, model, criterion, trainloader, optimizer, scheduler, epoch, scaler, writer=None, global_step=0):
    start_time = time.time()
    model.train()
    num_batches = len(trainloader)
    loss_total = 0.0
    use_amp = args.fp16 and args.device == 'cuda'

    # 在循环开始前记录起始时间
    epoch_start_time = time.time()

    for i, (ft_images, amps, phs) in enumerate(trainloader):

        if args.device == 'cuda':
            ft_images, amps, phs = ft_images.cuda(), amps.cuda(), phs.cuda()

        optimizer.zero_grad(set_to_none=True)

        # --- 前向传播开始 ---
        with torch.cuda.amp.autocast(enabled=use_amp):
            y, _, pred_amps, pred_phs, support = model(ft_images)
            loss = criterion(y, ft_images)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 梯度检查逻辑
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().max() > 0:
                has_grad = True
                break

        # --- 优化器更新开始 ---
        before = next(model.parameters()).data.flatten()[0].item()
        if scaler.is_enabled():
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= scale_before
        else:
            optimizer.step()
            optimizer_stepped = True
        after = next(model.parameters()).data.flatten()[0].item()

        # --- 记录与输出 ---
        if optimizer_stepped:
            global_step += 1
        if optimizer_stepped and args.lr_type == 'clr':
            scheduler.step()
        if optimizer_stepped and writer is not None:
            writer.add_scalar("Lr", optimizer.param_groups[0]['lr'], global_step)

        loss_total += loss.detach().item()

        # --- 时间计算逻辑 ---
        current_iter = i + 1
        # 已过去的时间 (秒)
        elapsed_seconds = time.time() - epoch_start_time
        # 每一迭代的平均时间
        avg_time_per_iter = elapsed_seconds / current_iter
        # 剩余迭代次数
        iters_left = num_batches - current_iter
        # 预估剩余时间 (秒)
        eta_seconds = iters_left * avg_time_per_iter

        # 转化为易读格式 (HH:MM:SS)
        elapsed_str = str(timedelta(seconds=int(elapsed_seconds)))
        eta_str = str(timedelta(seconds=int(eta_seconds)))

        # 打印当前 Batch 的各项耗时
        if i % 100 == 0:
            print(f"Epoch[{epoch}] Batch[{current_iter}/{num_batches}] | "
                f"Loss: {loss.item():.4e} | "
                f"LR: {optimizer.param_groups[0]['lr']:.3e} | "
                f"Grad: {str(has_grad):5s} | Update: {str(before!=after):5s} | "
                f"Elapsed: {elapsed_str} | ETA: {eta_str}",
                flush=True)

    # 原有的收尾逻辑
    loss_total /= num_batches
    time_cost = time.time() - start_time
    print("\n" + "="*80)
    print(f'✅ Epoch {epoch} 完成 | 总耗时: {time_cost:.2f}s | 平均 Loss: {loss_total:.4e}')
    print("="*80 + "\n")

    return loss_total, global_step


def validation(args, model, criterion, validloader, epoch):
    model.eval()
    num_batches = len(validloader)

    details_total = {
        'loss_l1':0.0,
        'loss_mae': 0.0,
        'loss_mse': 0.0,
        'loss_huber': 0.0,
        'loss_pcc': 0.0,
        'loss_comb': 0.0,
        'loss_comb2': 0.0,
    }

    with torch.no_grad():
        # 1. 移除 tqdm，使用标准 enumerate
        for i, (ft_images, amps, phs) in enumerate(validloader):
            if args.device == 'cuda':
                ft_images, amps, phs = ft_images.cuda(), amps.cuda(), phs.cuda()

            details = {}
            # 注意：validation 阶段通常建议开启 autocast（如果训练时也开启了）以保持精度一致
            with torch.cuda.amp.autocast(enabled=args.fp16 and args.device == 'cuda'):
                y, _, pred_amps, pred_phs, support = model(ft_images)  # 前向传播

                # Keep validation on the same raw diffraction scale as training; no scale_raw normalization is applied.
                
                # 计算各项指标
                details['loss_l1'] = criterion(ft_images, y)
                details['loss_mae'] = loss_mae(ft_images, y)
                details['loss_mse'] = loss_sq(ft_images, y)
                details['loss_huber'] = nn.SmoothL1Loss()(y, ft_images)
                details['loss_pcc'] = loss_pcc(ft_images, y)
                details['loss_comb'] = loss_comb(ft_images, y)
                details['loss_comb2'] = loss_comb2(ft_images, y)

            # 累积分项损失以计算平均值
            for key in details_total:
                details_total[key] += details[key].item() # 使用 .item() 避免显存堆积

            # 2. 每 100 个 batch 输出一次，或者在最后一个 batch 输出
            if (i + 1) % 100 == 0 or (i + 1) == num_batches:
                print(f"Validation Epoch [{epoch}] | Batch [{i+1}/{num_batches}] | "
                      f"L1: {details['loss_l1']:.4e} | "
                      f"MAE: {details['loss_mae']:.4e} | "
                      f"MSE: {details['loss_mse']:.4e} | "
                      f"PCC: {details['loss_pcc']:.4e}", flush=True)

    # 计算平均值
    for key in details_total:
        details_total[key] /= num_batches

    print("\n" + "="*80)
    print(f'✅ Epoch {epoch} 验证完成')
    print(f'📉 平均损失:')
    print(f'   MAE:   {details_total["loss_mae"]:.4e} | MSE:   {details_total["loss_mse"]:.4e}')
    print(f'   Huber: {details_total["loss_huber"]:.4e} | PCC:   {details_total["loss_pcc"]:.4e}')
    print(f'   Comb:  {details_total["loss_comb"]:.4e} | Comb2: {details_total["loss_comb2"]:.4e}')
    print("="*80 + "\n")

    return details_total

# def validation(args, model, criterion, validloader, epoch):
#     model.eval()
#     num_batches = len(validloader)

#     details_total = {
#             'loss_l1':0.0,
#             'loss_mae': 0.0,
#             'loss_mse': 0.0,
#             'loss_huber': 0.0,
#             'loss_pcc': 0.0,
#             'loss_comb': 0.0,
#             'loss_comb2': 0.0,
#         }
    

#     with torch.no_grad():
#         pbar = tqdm(enumerate(validloader), total=num_batches, desc=f"Epoch {epoch}")
#         for i, (ft_images, amps, phs) in pbar:
#             if args.device == 'cuda':
#                 ft_images, amps, phs = ft_images.cuda(), amps.cuda(), phs.cuda()

#             details = {}
#             with torch.cuda.amp.autocast(enabled=args.fp16):
#                 y, _, pred_amps, pred_phs, support = model(ft_images)  # Forward pass
#                 #loss, details = criterion(
#                 #    y, ft_images, pred_amps, amps, pred_phs, phs, support,
#                 #    supervised=not args.unsupervise
#                 #)
#                 scale_raw = torch.sum(ft_images) / (torch.sum(y) + 1e-10)
#                 y = y * scale_raw
                
#                 # details['loss_mae'] = get_criterion('mae')(y, ft_images)
#                 # details['loss_mse'] = get_criterion('mse')(y, ft_images)
#                 # details['loss_huber'] = get_criterion('huber')(y, ft_images)
#                 # details['loss_pcc'] = get_criterion('pcc')(y, ft_images)
#                 # details['loss_comb'] = get_criterion('comb')(y, ft_images)
#                 # details['loss_comb2'] = get_criterion('comb2')(y, ft_images)
#                 details['loss_l1'] = criterion(ft_images, y)
#                 details['loss_mae'] = loss_mae(ft_images, y)
#                 details['loss_mse'] = loss_sq(ft_images, y)
#                 details['loss_huber'] = nn.SmoothL1Loss()(y, ft_images)
#                 details['loss_pcc'] = loss_pcc(ft_images, y)
#                 details['loss_comb'] = loss_comb(ft_images, y)
#                 details['loss_comb2'] = loss_comb2(ft_images, y)

#                 # if args.unsupervise:
#                 #     loss = loss_f  # Use only FT loss for gradients
#                 # else:
#                 #     loss = loss_a + loss_p + loss_f

#             # 累积分项损失以计算平均值
#             for key in details_total:
#                 details_total[key] += details[key]

#             # 动态更新tqdm进度条，显示当前Batch的损失（不刷屏，实时刷新）
#             pbar.set_postfix({
#                'loss_l1': f'{details["loss_l1"]:.4e}',
#                'loss_mae': f'{details["loss_mae"]:.4e}',
#                'loss_mse': f'{details["loss_mse"]:.4e}',
#                'loss_huber': f'{details["loss_huber"]:.4e}',
#                'loss_pcc': f'{details["loss_pcc"]:.4e}',
#                'loss_comb': f'{details["loss_comb"]:.4e}',
#                'loss_comb2': f'{details["loss_comb2"]:.4e}'
#             })

#     for key in details_total:
#         details_total[key] /= num_batches

#     print("\n" + "="*80)
#     print(f'📉 平均损失 | loss_l1: {details_total["loss_l1"]:.4e} | loss_mae: {details_total["loss_mae"]:.4e} | loss_mse: {details_total["loss_mse"]:.4e} | loss_huber: {details_total["loss_huber"]:.4e} | loss_pcc: {details_total["loss_pcc"]:.4e} | loss_comb: {details_total["loss_comb"]:.4e} | loss_comb2: {details_total["loss_comb2"]:.4e}')
#     print("="*80 + "\n")

#     return details_total

if __name__ == "__main__":
    print('Starting script \n pytorch version: {}'.format(torch.__version__))
    torch.set_num_threads(1)
    t0 = time.time()
    
    # Training settings
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # shared args
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--OutputFolder', type=str, default='/lcrc/project/AutoPhase/test_pytorch/')
    parser.add_argument('--model_name', type=str, default='autophasenn', help='model name: autophasenn or umamba')
    parser.add_argument('--checkpoint', type=str, default='')
    #parser.add_argument('--unsupervise', action='store_true', default=False)
    parser.add_argument('--DataFolder', type=str, default='/data_hdd/oyys/autophaseNN/CDI_simulation_upsamp_noise/')
    parser.add_argument('--data_train_diff', type=str, default='train_diff.npy')
    parser.add_argument('--data_train_real', type=str, default='train_real.npy')
    parser.add_argument('--data_val_diff', type=str, default='val_diff.npy')
    parser.add_argument('--num_samples_train', type=int, default=25000)
    parser.add_argument('--num_samples_val', type=int, default=5000)
    parser.add_argument('--data_val_real', type=str, default='val_real.npy')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--epoch', default=5, type=int, help='training epochs')
    parser.add_argument('--train_size', type=int, default=60000, help='training data size')
    parser.add_argument('--train_perc', type=float, default=0.9)
    parser.add_argument('--loss_type', type=str, default='mae', help='loss type')
    parser.add_argument('--Initlr', type=float, default=1e-5, help='initial lr')
    parser.add_argument('--lr_type', type=str, default='clr', help='lr type')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum lr for cosine/plateau schedulers')
    parser.add_argument('--clr_step_epochs', type=float, default=6.0, help='epochs for one CLR warm-up half-cycle')
    parser.add_argument('--optim_type', type=str, default='adam', help='lr optim_type')
    #parser.add_argument('--use_down_stride', action='store_true', default=False)
    #parser.add_argument('--use_up_stride', action='store_true', default=False)
    #parser.add_argument('--n_blocks', type=int, default=4)
    #parser.add_argument('--scale_I', type=int, default=0)
    parser.add_argument('--num_workers', default=8, type=int, help='num of workers') # 降低了默认worker数以适应单机
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--num_threads', type=int, default=0)
    parser.add_argument('--fp16', action='store_true', default=False, help='enable mixed precision training; disabled by default')
    parser.add_argument('--reset_optimizer', action='store_true',
                        help='load checkpoint weights but restart optimizer/scheduler/scaler with the requested lr settings')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
    parser.add_argument('--notes', type=str, default='test')

    parser.add_argument('--shape', type=int, default=64)
    parser.add_argument('--T', type=float, default=0.1)
    parser.add_argument('--nconv', type=int, default=32)
    parser.add_argument('--use_down_stride', type=str2bool, default=False)
    parser.add_argument('--use_up_stride', type=str2bool, default=False)
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--unsupervise', type=str2bool, default=True)
    parser.add_argument('--scale_I', type=int, default=1) 

    args = parser.parse_args()

    arguments_strOut = args.OutputFolder
    if not os.path.exists(arguments_strOut):
        os.makedirs(arguments_strOut)

    total_train_size = args.train_size
    batch_size = args.batch_size
    data_path = args.DataFolder
    result_path = args.OutputFolder
    scale_I = args.scale_I

    data_train_diff = args.data_train_diff
    data_train_real = args.data_train_real
    data_val_diff = args.data_val_diff
    data_val_real = args.data_val_real

    num_samples_train = args.num_samples_train
    num_samples_val = args.num_samples_val

    model_name = args.model_name

    for key, value in args.__dict__.items():
        print('{}: {}'.format(key, value))
    print('use device: {}'.format(args.device))

    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(0) # 强制使用第一张卡

    if (args.num_threads != 0):
        torch.set_num_threads(args.num_threads)

    print("Torch Thread setup: ")
    print(" Number of threads: ", torch.get_num_threads())

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.device == 'cuda'}
    if args.num_workers > 0:
        kwargs.update({'prefetch_factor': 2, 'persistent_workers': True})

    print(kwargs)

    with open(os.path.join(args.OutputFolder, 'setting.json'), 'w') as f:
        f.write(json.dumps(args.__dict__, indent=4))

    layout = {
        "": {
            "Loss_ft": ["Multiline", ["Loss_ft/train", "Loss_ft/validation"]],
            "Loss_amp": ["Multiline", ["Loss_amp/train", "Loss_amp/validation"]],
            "Loss_ph": ["Multiline", ["Loss_ph/train", "Loss_ph/validation"]],
            'LR': ["Multiline", ["Lr", "Lr_epoch"]]
        },
    }
    writer = SummaryWriter(comment=os.path.basename(os.path.dirname(result_path)))
    writer.add_custom_scalars(layout)

    plans = {
        "dataset_name": "Diffraction3D",
        "original_median_spacing_after_transp": [1.0, 1.0, 1.0],
        "original_median_shape_after_transp": [64, 64, 64],
        "image_reader_writer": "SimpleITKIO",
        "transpose_forward": [0, 1, 2],
        "transpose_backward": [0, 1, 2],

        "configurations": {
            "3d_fullres": {
                "data_identifier": "nnUNetPlans_3d_fullres",
                "preprocessor_name": "DefaultPreprocessor",
                "batch_size": args.batch_size,
                "patch_size": [64, 64, 64],
                "median_image_size_in_voxels": [64.0, 64.0, 64.0],
                "spacing": [1.0, 1.0, 1.0],
                "normalization_schemes": ["ZScoreNormalization"],
                "use_mask_for_norm": [False],
                "UNet_class_name": "PlainConvUNet",
                "UNet_base_num_features": 32,
                "n_conv_per_stage_encoder": [2, 2, 2, 2], 
                "n_conv_per_stage_decoder": [2, 2, 2],
                "num_pool_per_axis": [4, 4, 4],
                "pool_op_kernel_sizes": [[1, 1, 1],[2, 2, 2],[2, 2, 2],[2, 2, 2]],
                "conv_kernel_sizes": [[3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3]],
                "unet_max_num_features": 320,
                "resampling_fn_data": "fast_resample_data_or_seg_to_shape",
                "resampling_fn_seg": "fast_resample_data_or_seg_to_shape",
                "resampling_fn_data_kwargs": {
                    "is_seg": False,
                    "order": 3,
                    "order_z": 3,
                    "force_separate_z": None
                },
                "resampling_fn_seg_kwargs": {
                    "is_seg": True,
                    "order": 1,
                    "order_z": 1,
                    "force_separate_z": None
                },
                "resampling_fn_probabilities": "fast_resample_data_or_seg_to_shape",
                "resampling_fn_probabilities_kwargs": {
                    "is_seg": False,
                    "order": 1,
                    "order_z": 1,
                    "force_separate_z": None
                },
                "batch_dice": True,
            },
            "3d_cascade_fullres": {
                "inherits_from": "3d_fullres",
                "previous_stage": "3d_lowres"
            }
        },
        "experiment_planner_used": "ExperimentPlanner",
        "label_manager": "LabelManager",

        "foreground_intensity_properties_per_channel": {
            "0": {
                "max": 3071.0,
                "mean": 97.29716491699219,
                "median": 118.0,
                "min": -1024.0,
                "percentile_00_5": -958.0,
                "percentile_99_5": 270.0,
                "std": 137.8484649658203
            }
        }
    }
    # 模拟dataset.json（标签配置）
    dataset_json = {"labels": {"background":0,}, "num_segmentation_heads":1}
    
    # 初始化PlansManager/ConfigurationManager（函数必需入参）
    plans_manager = PlansManager(plans)
    config_manager = plans_manager.get_configuration("3d_fullres")

    if model_name == 'umamba':
        model = get_umamba_enc_3d_from_plans(
            plans_manager=plans_manager,
            dataset_json=dataset_json,
            configuration_manager=config_manager,
            num_input_channels=1,  # 单模态输入（如CT）
            deep_supervision=False
        )
    elif model_name == 'autophasenn':
        model = Network(args).to(args.device)

    checkpoint_path = args.checkpoint
    device = torch.device("cuda" if args.device == 'cuda' else "cpu")
    load_success = False  # 加载状态标志
    checkpoint = None

    # 【核心容错】捕获所有加载异常：文件不存在、文件损坏、解析失败等
    try:
        # 1. 加载检查点文件
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"✅ 成功加载检查点文件：{checkpoint_path}")

        # 2. 兼容多种权重存储格式
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint  # 兼容直接保存模型权重

        # 3. 清洗权重key前缀（model. / net.）
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                cleaned_state_dict[k.replace("model.", "")] = v
            elif k.startswith("net."):
                cleaned_state_dict[k.replace("net.", "")] = v
            else:
                cleaned_state_dict[k] = v

        # 4. 加载权重到模型
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        print(f"✅ 模型权重加载完成！缺失层: {len(missing_keys)}, 多余层: {len(unexpected_keys)}")
        load_success = True

    # 捕获所有异常，失败则从零开始训练
    except Exception as e:
        print(f"\n⚠️  检查点加载失败：{str(e)}")
        print(f"⚠️  未找到有效检查点文件，将**从零开始训练**模型！\n")

    # ===================== 统一后续逻辑（无论是否加载权重都执行） =====================
    # 模型迁移到指定设备
    if args.device == 'cuda':
        model.cuda()

        # 可选：打印最终状态
    if load_success:
        print("🔍 训练模式：断点续训")
    else:
        print("🔍 训练模式：从头训练")

    print('model parameters: {}'.format(sum([param.nelement() for param in model.parameters()])))

    # load data
    data_train_diff = os.path.join(data_path, data_train_diff)
    data_train_real = os.path.join(data_path, data_train_real)
    data_val_diff = os.path.join(data_path, data_val_diff)
    data_val_real = os.path.join(data_path, data_val_real)
    # filelist_train = []
    # filelist_val = []

    # with open(dataname_list_train, 'r') as f:
    #     txtfile = f.readlines()
    # for i in range(len(txtfile)):
    #     tmp = str(txtfile[i]).split('/')[-1]
    #     tmp = tmp.split('\n')[0]
    #     filelist_train.append(tmp)

    # with open(dataname_list_val, 'r') as f:
    #     txtfile = f.readlines()
    # for i in range(len(txtfile)):
    #     tmp = str(txtfile[i]).split('/')[-1]
    #     tmp = tmp.split('\n')[0]
    #     filelist_val.append(tmp)

    # give training data size and filelist
    # print('number of training:%d' % len(filelist_train))
    # print('number of validation:%d' % len(filelist_val))

    # Single GPU DataLoaders (Removed DistributedSampler)
    train_dataset = Dataset(data_train_diff, data_train_real, num_samples_train, 
                            dtype_diff='float32', dtype_real='complex64', scale_I=scale_I, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    validation_dataset = Dataset(data_val_diff, data_val_real, num_samples_val, 
                                 dtype_diff='float32', dtype_real='complex64', scale_I=scale_I, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    # Setup optimizer and learning rate
    LR = args.Initlr 
    #criterion = CombinedDiffractionLoss().to(args.device)
    #criterion = get_criterion(args.loss_type)
    criterion = nn.L1Loss().to(args.device)  # 训练时使用 L1 损失，验证时计算多种指标

    if args.optim_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    elif args.optim_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    else:
        raise ValueError(f"Unsupported optim_type: {args.optim_type}")

    training_losses = []
    validation_losses = []
    epochs = 0
    best_val_loss = float('inf')
    global_step = 0

    if load_success and isinstance(checkpoint, dict):
        training_losses = checkpoint.get("training_losses", training_losses)
        validation_losses = checkpoint.get("validation_losses", validation_losses)
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        epochs = int(checkpoint.get("epoch", epochs))
        global_step = int(checkpoint.get("global_step", epochs * len(train_loader)))
        if args.reset_optimizer:
            training_losses = []
            validation_losses = []
            best_val_loss = float('inf')
        print(f"Resume metadata loaded: next_epoch={epochs + 1} | global_step={global_step}")

    if args.lr_type == 'clr':
        iterations_per_epoch = len(train_loader)
        step_size = max(1, int(round(args.clr_step_epochs * iterations_per_epoch)))
        print("LR step size is:", step_size, "which is every %.2f epochs" % (step_size/iterations_per_epoch))
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LR/10,
                                                      max_lr=LR, step_size_up=step_size,
                                                      cycle_momentum=False, mode='triangular2')
    elif args.lr_type == 'cosine':
        remaining_epochs = max(1, args.epoch - epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining_epochs, eta_min=args.min_lr)
    elif args.lr_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    elif args.lr_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=args.min_lr)
    else:
        raise ValueError(f"Unsupported lr_type: {args.lr_type}")
    print(f"LR scheduler: {args.lr_type} | init_lr={LR:.3e} | min_lr={args.min_lr:.3e}", flush=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and args.device == 'cuda'))

    t_start = time.time()

    # 最优模型固定保存路径（只会保留这一个最新最优文件）
    best_model_path = arguments_strOut + '/best_model.pt'
    if load_success and isinstance(checkpoint, dict):
        if args.reset_optimizer:
            print("Reset optimizer/scheduler/scaler; checkpoint weights and resume position were kept.")
        else:
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("Optimizer state restored.")
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
                print("Scheduler state restored.")
            if "scaler" in checkpoint and scaler.is_enabled():
                scaler.load_state_dict(checkpoint["scaler"])
                print("GradScaler state restored.")

    for epoch in range(epochs + 1, args.epoch+1):
        
        train_loss_details, global_step = train(
            args, model, criterion, train_loader, optimizer, scheduler, epoch, scaler,
            writer=writer, global_step=global_step
        )

        training_losses.append(train_loss_details)

        val_loss_details = validation(args, model, criterion, validation_loader, epoch)
        
        validation_losses.append(val_loss_details)

        if args.lr_type == 'step':
            scheduler.step()
        elif args.lr_type == 'cosine':
            scheduler.step()
        elif args.lr_type == 'plateau':
            scheduler.step(val_loss_details['loss_l1'])

        # ===================== 修复2：记录学习率（兼容所有调度器） =====================
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Lr_epoch", current_lr, epoch)
        print(f"Epoch {epoch} scheduler lr: {current_lr:.3e}", flush=True)

        # ===================== 自动保存【最优模型】 =====================

        current_val_loss = val_loss_details['loss_l1']
        # 对比：当前模型比历史最优更好（损失更小）
        if current_val_loss < best_val_loss:
            # 1. 删除旧的最优模型（如果存在）
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
                print(f"🗑️  已删除旧最优模型 | 历史最优损失: {best_val_loss:.6f}")
            # 2. 更新最优损失值
            best_val_loss = current_val_loss
            # 3. 保存新的最优模型（完整断点信息）
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'training_losses': training_losses,
                    'validation_losses': validation_losses,
                    'best_val_loss': best_val_loss,  # 额外记录最优损失
                    'global_step': global_step,
                }, best_model_path)
            print(f"✅ 已保存新最优模型 | 当前最优损失: {best_val_loss:.6f} | Epoch: {epoch}")
        # ======================================================================

        if epoch % args.save_model == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'training_losses': training_losses,
                    'validation_losses': validation_losses,
                    'best_val_loss': best_val_loss,
                    'global_step': global_step,
                }, arguments_strOut + '/training_model_{:06d}'.format(epoch) + '.pt')
            print('checkpoint saved!')

        writer.add_scalar('loss-coarse', train_loss_details, epoch)
        writer.add_scalars('loss-coarse-val', val_loss_details, epoch)

    t1 = time.time()
    print("Total running time: %s seconds" % (t1 - t0))
    print("Average time per epoch: {:.2f}s".format((t1-t_start)/(args.epoch-epochs)))

    writer.flush()
    writer.close()
