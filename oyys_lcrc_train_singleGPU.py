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

def safe_path_name(value):
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in str(value))
    return safe.strip('._') or 'experiment'

def to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)

def tensor_stats(tensor):
    data = tensor.detach().float()
    finite = data[torch.isfinite(data)]
    if finite.numel() == 0:
        return {
            'mean': float('nan'),
            'std': float('nan'),
            'min': float('nan'),
            'max': float('nan'),
            'sum': float('nan'),
            'abs_mean': float('nan'),
        }
    return {
        'mean': to_float(finite.mean()),
        'std': to_float(finite.std(unbiased=False)),
        'min': to_float(finite.min()),
        'max': to_float(finite.max()),
        'sum': to_float(finite.sum()),
        'abs_mean': to_float(finite.abs().mean()),
    }

def scaled_l1(pred, target, criterion):
    scale = target.detach().float().sum() / (pred.detach().float().sum() + 1e-8)
    return criterion(pred * scale, target), scale

def support_threshold(model, default=0.1):
    layer = getattr(model, 'support_layer', None)
    return float(getattr(layer, 'threshold', default))

def hard_support(amp, threshold):
    return torch.where(amp >= threshold, torch.ones_like(amp), torch.zeros_like(amp))

def sigmoid_support(amp, threshold, steepness=50.0):
    return torch.sigmoid(steepness * (amp - threshold))

def fft_amplitude(obj, pre_shift=False, post_shift=True):
    spatial_dims = (-3, -2, -1)
    x = torch.fft.ifftshift(obj, dim=spatial_dims) if pre_shift else obj
    x = torch.fft.fftn(x, dim=spatial_dims, norm=None)
    x = torch.fft.fftshift(x, dim=spatial_dims) if post_shift else x
    return torch.abs(x).to(torch.float32)

def pearson_corr(pred, target):
    pred = pred.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = torch.sqrt(torch.sum(pred * pred) * torch.sum(target * target)) + 1e-8
    return torch.sum(pred * target) / denom

def log_debug_diagnostics(stage, epoch, batch_idx, model, criterion, ft_images, y,
                          pred_amps, pred_phs, support, amps, phs, writer=None, step=None):
    with torch.no_grad():
        target = ft_images.detach().float()
        pred = y.detach().float()
        l1 = criterion(pred, target)
        zero_l1 = criterion(torch.zeros_like(target), target)
        rel_l1 = torch.sum(torch.abs(pred - target)) / (torch.sum(torch.abs(target)) + 1e-8)
        pred_scaled_l1, pred_scale = scaled_l1(pred, target, criterion)

        threshold = support_threshold(model)
        gt_support = model.support_layer(amps)
        gt_obj = model.obj_layer(amps, phs)
        gt_unmasked_y = model.farfield_layer(gt_obj).detach().float()

        gt_hard_support = hard_support(amps, threshold)
        gt_hard_masked_obj = model.masked_obj_layer(gt_obj, gt_hard_support)
        gt_hard_y = model.farfield_layer(gt_hard_masked_obj).detach().float()

        gt_sigmoid_support = sigmoid_support(amps, threshold)
        gt_sigmoid_masked_obj = model.masked_obj_layer(gt_obj, gt_sigmoid_support)
        gt_sigmoid_y = model.farfield_layer(gt_sigmoid_masked_obj).detach().float()

        fft_variants = {
            'no_pre_shift': fft_amplitude(gt_obj, pre_shift=False, post_shift=True),
            'no_post_shift': fft_amplitude(gt_obj, pre_shift=True, post_shift=False),
            'no_shifts': fft_amplitude(gt_obj, pre_shift=False, post_shift=False),
        }
        fft_variant_l1 = {
            name: criterion(value, target) for name, value in fft_variants.items()
        }
        fft_variant_pcc = {
            name: pearson_corr(value, target) for name, value in fft_variants.items()
        }

        gt_masked_obj = model.masked_obj_layer(gt_obj, gt_support)
        gt_y = model.farfield_layer(gt_masked_obj).detach().float()
        gt_l1 = criterion(gt_y, target)
        gt_scaled_l1, gt_scale = scaled_l1(gt_y, target, criterion)
        gt_unmasked_l1 = criterion(gt_unmasked_y, target)
        gt_hard_l1 = criterion(gt_hard_y, target)
        gt_sigmoid_l1 = criterion(gt_sigmoid_y, target)
        gt_unmasked_scaled_l1, gt_unmasked_scale = scaled_l1(gt_unmasked_y, target, criterion)
        gt_hard_scaled_l1, gt_hard_scale = scaled_l1(gt_hard_y, target, criterion)
        gt_sigmoid_scaled_l1, gt_sigmoid_scale = scaled_l1(gt_sigmoid_y, target, criterion)
        gt_pcc = pearson_corr(gt_y, target)
        gt_unmasked_pcc = pearson_corr(gt_unmasked_y, target)
        target_fraction = torch.abs(target - torch.round(target))

        target_stats = tensor_stats(target)
        pred_stats = tensor_stats(pred)
        gt_stats = tensor_stats(gt_y)
        gt_unmasked_stats = tensor_stats(gt_unmasked_y)
        gt_hard_stats = tensor_stats(gt_hard_y)
        gt_sigmoid_stats = tensor_stats(gt_sigmoid_y)
        amp_stats = tensor_stats(amps)
        debug_values = {
            'l1': to_float(l1),
            'zero_l1': to_float(zero_l1),
            'relative_l1': to_float(rel_l1),
            'scaled_l1': to_float(pred_scaled_l1),
            'scale_pred_to_target': to_float(pred_scale),
            'gt_forward_l1': to_float(gt_l1),
            'gt_forward_scaled_l1': to_float(gt_scaled_l1),
            'scale_gt_to_target': to_float(gt_scale),
            'gt_unmasked_l1': to_float(gt_unmasked_l1),
            'gt_unmasked_scaled_l1': to_float(gt_unmasked_scaled_l1),
            'scale_gt_unmasked_to_target': to_float(gt_unmasked_scale),
            'gt_hard_support_l1': to_float(gt_hard_l1),
            'gt_hard_support_scaled_l1': to_float(gt_hard_scaled_l1),
            'scale_gt_hard_to_target': to_float(gt_hard_scale),
            'gt_sigmoid_support_l1': to_float(gt_sigmoid_l1),
            'gt_sigmoid_support_scaled_l1': to_float(gt_sigmoid_scaled_l1),
            'scale_gt_sigmoid_to_target': to_float(gt_sigmoid_scale),
            'gt_forward_pcc': to_float(gt_pcc),
            'gt_unmasked_pcc': to_float(gt_unmasked_pcc),
            'fft_no_pre_shift_l1': to_float(fft_variant_l1['no_pre_shift']),
            'fft_no_post_shift_l1': to_float(fft_variant_l1['no_post_shift']),
            'fft_no_shifts_l1': to_float(fft_variant_l1['no_shifts']),
            'fft_no_pre_shift_pcc': to_float(fft_variant_pcc['no_pre_shift']),
            'fft_no_post_shift_pcc': to_float(fft_variant_pcc['no_post_shift']),
            'fft_no_shifts_pcc': to_float(fft_variant_pcc['no_shifts']),
            'target_mean': target_stats['mean'],
            'target_max': target_stats['max'],
            'target_sum': target_stats['sum'],
            'target_fraction_mean': to_float(target_fraction.mean()),
            'target_integer_fraction': to_float((target_fraction < 1e-6).float().mean()),
            'pred_mean': pred_stats['mean'],
            'pred_max': pred_stats['max'],
            'pred_sum': pred_stats['sum'],
            'gt_mean': gt_stats['mean'],
            'gt_max': gt_stats['max'],
            'gt_sum': gt_stats['sum'],
            'gt_unmasked_mean': gt_unmasked_stats['mean'],
            'gt_unmasked_max': gt_unmasked_stats['max'],
            'gt_unmasked_sum': gt_unmasked_stats['sum'],
            'gt_hard_mean': gt_hard_stats['mean'],
            'gt_hard_max': gt_hard_stats['max'],
            'gt_hard_sum': gt_hard_stats['sum'],
            'gt_sigmoid_mean': gt_sigmoid_stats['mean'],
            'gt_sigmoid_max': gt_sigmoid_stats['max'],
            'gt_sigmoid_sum': gt_sigmoid_stats['sum'],
            'amp_mean': amp_stats['mean'],
            'amp_max': amp_stats['max'],
            'support_threshold': threshold,
            'support_mean': to_float(support.detach().float().mean()),
            'gt_support_mean': to_float(gt_support.detach().float().mean()),
            'gt_hard_support_mean': to_float(gt_hard_support.detach().float().mean()),
            'gt_sigmoid_support_mean': to_float(gt_sigmoid_support.detach().float().mean()),
        }

    print(
        f"[DEBUG][{stage}] Epoch[{epoch}] Batch[{batch_idx}] | "
        f"L1={debug_values['l1']:.4e} | ZeroL1={debug_values['zero_l1']:.4e} | "
        f"RelL1={debug_values['relative_l1']:.4e} | "
        f"ScaledL1={debug_values['scaled_l1']:.4e} scale={debug_values['scale_pred_to_target']:.4e} | "
        f"GT_L1={debug_values['gt_forward_l1']:.4e} | "
        f"GT_ScaledL1={debug_values['gt_forward_scaled_l1']:.4e} gt_scale={debug_values['scale_gt_to_target']:.4e} | "
        f"GT_unmasked={debug_values['gt_unmasked_l1']:.4e} scale={debug_values['scale_gt_unmasked_to_target']:.4e} | "
        f"GT_hard={debug_values['gt_hard_support_l1']:.4e} scale={debug_values['scale_gt_hard_to_target']:.4e} | "
        f"GT_sigmoid={debug_values['gt_sigmoid_support_l1']:.4e} scale={debug_values['scale_gt_sigmoid_to_target']:.4e} | "
        f"GT_PCC={debug_values['gt_forward_pcc']:.4e} | "
        f"FFT(no_pre/no_post/no_shift)="
        f"{debug_values['fft_no_pre_shift_l1']:.4e}/"
        f"{debug_values['fft_no_post_shift_l1']:.4e}/"
        f"{debug_values['fft_no_shifts_l1']:.4e} | "
        f"target(mean/max/sum)={debug_values['target_mean']:.4e}/{debug_values['target_max']:.4e}/{debug_values['target_sum']:.4e} | "
        f"target_frac_mean={debug_values['target_fraction_mean']:.4e} "
        f"target_int_frac={debug_values['target_integer_fraction']:.4e} | "
        f"pred(mean/max/sum)={debug_values['pred_mean']:.4e}/{debug_values['pred_max']:.4e}/{debug_values['pred_sum']:.4e} | "
        f"gt(mean/max/sum)={debug_values['gt_mean']:.4e}/{debug_values['gt_max']:.4e}/{debug_values['gt_sum']:.4e} | "
        f"amp(mean/max)={debug_values['amp_mean']:.4e}/{debug_values['amp_max']:.4e} | "
        f"support={debug_values['support_mean']:.4e} gt_support={debug_values['gt_support_mean']:.4e} | "
        f"hard_support={debug_values['gt_hard_support_mean']:.4e} "
        f"sigmoid_support={debug_values['gt_sigmoid_support_mean']:.4e} "
        f"T={debug_values['support_threshold']:.4e}",
        flush=True
    )

    if writer is not None and step is not None:
        for key, value in debug_values.items():
            writer.add_scalar(f"debug/{stage}/{key}", value, step)

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
    if args.debug_max_train_batches > 0:
        num_batches = min(num_batches, args.debug_max_train_batches)
    loss_total = 0.0
    use_amp = args.fp16 and args.device == 'cuda'

    # 在循环开始前记录起始时间
    epoch_start_time = time.time()

    for i, (ft_images, amps, phs) in enumerate(trainloader):
        if i >= num_batches:
            break

        if args.device == 'cuda':
            ft_images, amps, phs = ft_images.cuda(), amps.cuda(), phs.cuda()

        optimizer.zero_grad(set_to_none=True)

        # --- 前向传播开始 ---
        with torch.cuda.amp.autocast(enabled=use_amp):
            y, _, pred_amps, pred_phs, support = model(ft_images)
            loss = criterion(y, ft_images)

        if args.debug_diagnostics and i < args.debug_diagnostic_batches:
            debug_step = global_step if global_step > 0 else (epoch - 1) * len(trainloader) + i + 1
            log_debug_diagnostics(
                'train', epoch, i + 1, model, criterion, ft_images, y,
                pred_amps, pred_phs, support, amps, phs, writer=writer, step=debug_step
            )

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
        current_batch_lr = optimizer.param_groups[0]['lr']
        if optimizer_stepped and writer is not None:
            writer.add_scalar("Lr", current_batch_lr, global_step)

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
                f"BatchLR: {current_batch_lr:.3e} | "
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
    if args.debug_max_val_batches > 0:
        num_batches = min(num_batches, args.debug_max_val_batches)

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
            if i >= num_batches:
                break
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
            if args.debug_diagnostics and i < args.debug_diagnostic_batches:
                log_debug_diagnostics(
                    'val', epoch, i + 1, model, criterion, ft_images, y,
                    pred_amps, pred_phs, support, amps, phs
                )

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
    parser.add_argument('--tensorboard_dir', type=str, default='runs',
                        help='root directory for TensorBoard logs')
    parser.add_argument('--debug_diagnostics', action='store_true',
                        help='print scale and physics-forward diagnostics for early batches')
    parser.add_argument('--debug_diagnostic_batches', type=int, default=1,
                        help='number of early batches per epoch to print diagnostics for')
    parser.add_argument('--debug_max_train_batches', type=int, default=0,
                        help='limit train batches per epoch when debugging; 0 means full epoch')
    parser.add_argument('--debug_max_val_batches', type=int, default=0,
                        help='limit validation batches per epoch when debugging; 0 means full validation')
    parser.add_argument('--debug_overfit_samples', type=int, default=0,
                        help='train and validate on the same first N train samples for overfit debugging')
    parser.add_argument('--debug_skip_scheduler', action='store_true',
                        help='keep the learning rate fixed during debug runs')
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
            "Loss_l1": ["Multiline", ["loss-coarse/train_loss_l1", "loss-coarse-val/loss_l1"]],
            'LR': ["Multiline", ["Lr", "Lr_epoch"]]
        },
    }
    experiment_name = safe_path_name(os.path.basename(os.path.normpath(result_path)))
    tb_run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{experiment_name}"
    tb_log_dir = os.path.join(args.tensorboard_dir, tb_run_name)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard log dir: {os.path.abspath(tb_log_dir)}", flush=True)
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
                "n_conv_per_stage_encoder": [2, 2, 2, 2, 2], 
                "n_conv_per_stage_decoder": [2, 2, 2, 2],
                "num_pool_per_axis": [4, 4, 4, 4],
                "pool_op_kernel_sizes": [[1, 1, 1],[2, 2, 2],[2, 2, 2],[2, 2, 2],[2, 2, 2]],
                "conv_kernel_sizes": [[3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3]],
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
    # Keep Dataset indices in natural order and let DataLoader reshuffle train samples every epoch.
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_dataset = Dataset(data_train_diff, data_train_real, num_samples_train, 
                            dtype_diff='float32', dtype_real='complex64', scale_I=scale_I, shuffle=False)
    validation_dataset = Dataset(data_val_diff, data_val_real, num_samples_val, 
                                 dtype_diff='float32', dtype_real='complex64', scale_I=scale_I, shuffle=False)

    train_shuffle = True
    if args.debug_overfit_samples > 0:
        debug_sample_count = min(args.debug_overfit_samples, len(train_dataset))
        debug_indices = list(range(debug_sample_count))
        debug_subset = torch.utils.data.Subset(train_dataset, debug_indices)
        train_dataset = debug_subset
        validation_dataset = debug_subset
        train_shuffle = False
        print(
            f"[DEBUG] Overfit mode enabled: using the same {debug_sample_count} train samples "
            "for both train and validation.",
            flush=True
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=train_shuffle, generator=train_generator, **kwargs)
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

        if args.debug_skip_scheduler:
            pass
        elif args.lr_type == 'step':
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

        writer.add_scalar('loss-coarse/train_loss_l1', train_loss_details, epoch)
        for key, value in val_loss_details.items():
            writer.add_scalar(f'loss-coarse-val/{key}', value, epoch)

    t1 = time.time()
    print("Total running time: %s seconds" % (t1 - t0))
    print("Average time per epoch: {:.2f}s".format((t1-t_start)/(args.epoch-epochs)))

    writer.flush()
    writer.close()
