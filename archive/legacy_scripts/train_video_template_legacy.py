import datetime
import os
import time
import warnings

import PyTorch.data_loader1 as data_loader1
import torch
import torch.utils.data
import torchvision
import torchvision.datasets.video_utils
import utils
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.samplers import DistributedSampler, RandomClipSampler, UniformClipSampler
from AutoPhaseNN_model import Network
from spike_utils import *
from PyTorch.data_loader1 import *
from torchinfo import summary
from utils import LossComb2


def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("clips/s", utils.SmoothedValue(window_size=10, fmt="{value:.3f}"))

    header = f"Epoch: [{epoch}]"
    for video, target, _ in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        video, target = video.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(video)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = video.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["clips/s"].update(batch_size / (time.time() - start_time))
        lr_scheduler.step()

import torch
import torch.nn.functional as F
import utils  # 假设你沿用了原有的工具库

def evaluate_spike(model, criterion, data_loader, device, delay):
    """
    适配 SNN 的衍射成像评估函数
    核心机制：在时间步 (timesteps) 上累积模型的输出振幅，取平均后与真值对比
    """
    model.eval()
    
    # 1. 初始化指标记录器 (改为记录重建指标)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    
    # 2. 设置 SNN 模式 (保持原逻辑)
    # 确保所有脉冲神经元处于 SNN 模式而不是 ANN 模式
    for module in model.modules():
        # 你的代码里原本是检查 SpikingNeuron 等类，这里保持不变
        # 建议检查类名字符串或者 hasattr，以防 import 问题
        class_name = module.__class__.__name__
        if "SpikingNeuron" in class_name and hasattr(module, "mode"):
            module.mode = "snn"

    # 3. 处理时间步和延迟
    timesteps = int(args.timesteps)
    # 自动修正 delay，防止 delay 设置得比总时间步还长
    if timesteps < delay + 4:
        delay = max(0, timesteps - 4)

    with torch.inference_mode():
        # 遍历验证集                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        for batch in metric_logger.log_every(data_loader, 100, header):
            
            # --- 数据解包 ---
            # 适配你的 DataLoader: (ft_images, amps, phs)
            ft_images = batch[0].to(device, non_blocking=True)
            target_amps = batch[1].to(device, non_blocking=True)
            # target_phs = batch[2].to(device, non_blocking=True) # 如果需要评估相位可以解包
            
            # --- SNN 时间步循环 ---
            # 初始化累加器 (Accumulator)
            # 我们主要累积 Amp (振幅) 用于计算 Loss 和指标
            sum_amp = 0
            valid_steps = 0
            
            for t in range(timesteps):
                # SNN 前向传播
                # 注意：对于静态图像重建，通常每个时间步输入相同的 ft_images
                output_tuple = model(ft_images) 
                
                # output_tuple = (psi, obj, amp, ph, support)
                # 我们只提取 amp (索引 2) 进行累积
                curr_amp = output_tuple[2]

                # 处理延迟 (Delay): 只有超过 delay 的时间步才计入结果
                if t >= delay:
                    sum_amp += curr_amp
                    valid_steps += 1
            
            # --- 计算平均输出 (Mean Firing Rate / Potential) ---
            # 避免除以 0
            if valid_steps > 0:
                avg_amp = sum_amp / valid_steps
            else:
                avg_amp = sum_amp # Fallback

            # --- 计算 Loss 和 指标 ---
            # 使用这里的 criterion (LossComb2) 计算 Loss
            loss = criterion(avg_amp, target_amps)

            # 计算物理指标 (MSE, PCC)
            batch_size = ft_images.shape[0]
            mse = F.mse_loss(avg_amp, target_amps)
            
            # 计算 PCC (Pearson Correlation Coefficient)
            # 1. 减均值
            pred_mean = avg_amp.mean(dim=(1, 2, 3), keepdim=True)
            target_mean = target_amps.mean(dim=(1, 2, 3), keepdim=True)
            pred_centered = avg_amp - pred_mean
            target_centered = target_amps - target_mean
            # 2. 计算相关性
            numerator = (pred_centered * target_centered).sum(dim=(1, 2, 3))
            denominator = torch.sqrt(
                (pred_centered ** 2).sum(dim=(1, 2, 3)) * (target_centered ** 2).sum(dim=(1, 2, 3))
            ) + 1e-8
            pcc = (numerator / denominator).mean()

            # --- 更新日志 ---
            metric_logger.update(loss=loss.item())
            metric_logger.meters["mse"].update(mse.item(), n=batch_size)
            metric_logger.meters["pcc"].update(pcc.item(), n=batch_size)

            # --- 🔴 重置 SNN 状态 (非常重要！) ---
            # 每个 Batch 结束后，必须重置神经元的膜电位，否则会通过隐状态泄漏到下一个 Batch
            # 假设你有一个全局函数 reset_model，或者 model.reset()
            reset_model(model)

    # --- 同步与打印 ---
    metric_logger.synchronize_between_processes()
    
    print(
        " * [SNN Test] Loss: {loss.global_avg:.4f}  "
        "MSE: {mse.global_avg:.4f}  "
        "PCC: {pcc.global_avg:.4f}".format(
            loss=metric_logger.loss,
            mse=metric_logger.meters["mse"],
            pcc=metric_logger.meters["pcc"]
        )
    )

    return metric_logger.loss.global_avg

import torch
import torch.nn.functional as F
import utils

def evaluate_ann(model, criterion, data_loader, device):
    """
    适配 ANN (模拟模式) 的衍射成像评估函数
    核心机制：单次前向传播，直接对比输出振幅与真值 (无时间步循环)
    """
    model.eval()
    
    # 1. 初始化指标记录器
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test (ANN):"
    
    # 2. 强制设置 ANN 模式
    # SNN 模型在训练或评估 ANN 性能时，需要将脉冲神经元切换为模拟激活模式
    for module in model.modules():
        class_name = module.__class__.__name__
        # 检查是否是脉冲神经元层
        if "SpikingNeuron" in class_name and hasattr(module, "mode"):
            module.mode = "ann" 

    with torch.inference_mode():
        # 遍历验证集
        for batch in metric_logger.log_every(data_loader, 100, header):
            
            # --- 数据解包 ---
            # 适配你的 DataLoader: (ft_images, amps, phs)
            ft_images = batch[0].to(device, non_blocking=True)
            target_amps = batch[1].to(device, non_blocking=True)
            # target_phs = batch[2].to(device, non_blocking=True) # 如需评估相位可解包
            
            # --- ANN 前向传播 ---
            # 这里的 model 已经是 ANN 模式，直接输出模拟值
            output_tuple = model(ft_images)
            
            # output_tuple = (psi, obj, amp, ph, support)
            # 我们提取 amp (索引 2) 进行对比
            pred_amp = output_tuple[2]

            # --- 计算 Loss 和 指标 ---
            # 使用传入的 criterion (如 LossComb2)
            loss = criterion(pred_amp, target_amps)

            # --- 计算物理指标 (MSE, PCC) ---
            batch_size = ft_images.shape[0]
            
            # 1. MSE
            mse = F.mse_loss(pred_amp, target_amps)
            
            # 2. PCC (Pearson Correlation Coefficient)
            # 减均值
            pred_mean = pred_amp.mean(dim=(1, 2, 3), keepdim=True)
            target_mean = target_amps.mean(dim=(1, 2, 3), keepdim=True)
            pred_centered = pred_amp - pred_mean
            target_centered = target_amps - target_mean
            
            # 计算相关性分子分母
            numerator = (pred_centered * target_centered).sum(dim=(1, 2, 3))
            denominator = torch.sqrt(
                (pred_centered ** 2).sum(dim=(1, 2, 3)) * (target_centered ** 2).sum(dim=(1, 2, 3))
            ) + 1e-8
            pcc = (numerator / denominator).mean()

            # --- 更新日志 ---
            metric_logger.update(loss=loss.item())
            metric_logger.meters["mse"].update(mse.item(), n=batch_size)
            metric_logger.meters["pcc"].update(pcc.item(), n=batch_size)

    # --- 同步与打印 ---
    # 汇总所有 GPU 的结果
    metric_logger.synchronize_between_processes()
    
    print(
        " * [ANN Test] Loss: {loss.global_avg:.4f}  "
        "MSE: {mse.global_avg:.4f}  "
        "PCC: {pcc.global_avg:.4f}".format(
            loss=metric_logger.loss,
            mse=metric_logger.meters["mse"],
            pcc=metric_logger.meters["pcc"]
        )
    )

    # 返回 Loss 用于 ModelCheckpoint
    return metric_logger.loss.global_avg

def _get_cache_path(filepath, args):
    import hashlib

    value = f"{filepath}"
    h = hashlib.sha1(value.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def collate_fn(batch):
    # remove audio from the batch
    batch = [(d[0], d[2], d[3]) for d in batch]
    return default_collate(batch)


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # Data loading code
    print("Loading data")
    data_size = tuple(args.data_size)

    dataname_list = os.path.join(args.data_path, '3D_upsamp.txt')
    filelist = []

    with open(dataname_list, 'r') as f:
        txtfile = f.readlines()
    for i in range(len(txtfile)):
        tmp = str(txtfile[i]).split('/')[-1]
        tmp = tmp.split('\n')[0]

        filelist.append(tmp)
    f.close()

    print('number of available file:%d' % len(filelist))
    train_filelist = filelist[:int(len(filelist)*args.train_ratio)]
    print('number of training:%d' % len(train_filelist))

    train_dataset = Dataset(
    train_filelist, args.data_path, load_all=False, ratio=args.train_ratio, dataset='train',scale_I=args.scale_I)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    validation_dataset = Dataset(
        train_filelist, args.data_path, load_all=False, ratio=args.train_ratio, dataset='validation',scale_I=args.scale_I)
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers*2)



    print("Creating model")

    model = Network(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = LossComb2().to(device)

    model_path = r"D:\code\PYTHON\AutoPhaseNN-main\torch_transferred_weights_final.pth"
    params = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(params)


    if args.ann:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        #criterion = nn.CrossEntropyLoss()
        evaluate_ann(model, criterion, validation_loader, device=device)
        #return

    model, cnt = convert_ann_to_snn(model)
    model.to(device)
    h = w = t = 64
    summary(model, (1, 1, h, w, t), device=device)

    save_path = "snn_autophase.pth"
    torch.save(
        model.state_dict(),  # 直接保存模型的参数字典，不含任何多余数据
        save_path
    )

    if args.load_path is not None:
        sd = torch.load(args.load_path +".pth")
        model.load_state_dict(sd, strict=True)
    else:
        weight_scaling_iter_new(train_loader, model, device, 1000, criterion)
    delay = cal_delay_time(train_loader, model, device)
    print(delay)
    # torch.save(model.state_dict(), './model_r3d.pth')

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)


    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        evaluate_spike(model, criterion, validation_loader, device, delay)
        return
    
    save_path = "snn_autophase.pth"
    torch.save(
        model.state_dict(),  # 直接保存模型的参数字典，不含任何多余数据
        save_path
    )

    print(delay)



def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Video Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default=r"D:\code\PYTHON\AutoPhaseNN-main\TF2\CDI_simulation_upsamp_noise\\", type=str, help="dataset path")
    parser.add_argument("--train_ratio", default=0.9, type=float, help="train_ratio")
    parser.add_argument("--scale_I", default=1, type=float, help="scale_I normalize diff or not")
    parser.add_argument('--load_path', type=str, default=None) 
    parser.add_argument('--shape', type=int, default=64)  # 输入维度64x64x64
    parser.add_argument('--T', type=float, default=0.1)       # 时间维度（假设）
    parser.add_argument('--nconv', type=int, default=32)   # 初始卷积数（对应TF的32）
    parser.add_argument('--use_down_stride', type=bool, default=False)  # 按你的实际配置
    parser.add_argument('--use_up_stride', type=bool, default=False)    # 按你的实际配置
    parser.add_argument('--n_blocks', type=int, default=4)  # Encoder 4个block（对应TF 4个MaxPool）
    parser.add_argument('--unsupervise', type=bool, default=True) 
    parser.add_argument("--output_dir", default="./output", type=str, help="output dir")
    parser.add_argument("--model", default="AutoPhaseNN", type=str, help="model name")
    parser.add_argument("--device", default="cuda:0", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=12, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=45, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=12, type=int, metavar="N", help="number of data loading workers (default: 10)"
    )
    parser.add_argument("--lr", default=0.64, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--lr-milestones", nargs="+", default=[20, 30, 40], type=int, help="decrease lr on milestones")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-warmup-epochs", default=10, type=int, help="the number of epochs to warmup (default: 10)")
    parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)")
    parser.add_argument("--lr-warmup-decay", default=0.001, type=float, help="the decay for lr")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        default=True,
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        default=True,
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    parser.add_argument(
        "--data-size",
        default=(64, 64),
        nargs="+",
        type=int,
        help="the resize size used for validation (default: (64,64))",
    )

    parser.add_argument("--weights", default="R3D_18_Weights.DEFAULT", type=str, help="the weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--ann",  type=bool, default=True)
    parser.add_argument("--timesteps", default=256)

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)




if __name__ == "__main__":
    # ===================== 1. 模拟必要的配置（无需真实数据集） =====================
    # 模拟plans.json核心参数（和UMambaEnc 3D匹配）
    plans = {
        "dataset_name": "Diffraction3D",
        "UNet_base_num_features": 32,
        "unet_max_num_features": 256,

        # N=3次下采样，所以这里3个元素
        "pool_op_kernel_sizes":      [[2,2,2], [2,2,2], [2,2,2]],

        # N+1=4个stage（3个encoder + 1个bottleneck）
        "conv_kernel_sizes":         [[3,3,3], [3,3,3], [3,3,3], [3,3,3]],
        "n_conv_per_stage_encoder":  [2, 2, 2, 4],

        # N=3个decoder stage，与下采样次数对称
        "n_conv_per_stage_decoder":  [2, 2, 2],

        "configurations": {
            "3d_fullres": {
                "patch_size": [64, 64, 64],
                "pool_op_kernel_sizes":     [[2,2,2], [2,2,2], [2,2,2]],
                "conv_kernel_sizes":        [[3,3,3], [3,3,3], [3,3,3], [3,3,3]],
                "n_conv_per_stage_encoder": [2, 2, 2, 4],
                "n_conv_per_stage_decoder": [2, 2, 2],
                "UNet_base_num_features": 32,
                "unet_max_num_features": 256,
                # 关键：显式加这一行
                
            }
        }
    }
    # 模拟dataset.json（标签配置）
    dataset_json = {"labels": {"background":0, "tumor":1}, "num_segmentation_heads":1}
    
    # 初始化PlansManager/ConfigurationManager（函数必需入参）
    plans_manager = PlansManager(plans)
    config_manager = plans_manager.get_configuration("3d_fullres")

    # ===================== 2. 调用目标函数初始化模型 =====================
    model = get_umamba_enc_3d_from_plans(
        plans_manager=plans_manager,
        dataset_json=dataset_json,
        configuration_manager=config_manager,
        num_input_channels=1,  # 单模态输入（如CT）
        deep_supervision=True
    )
    model = model.cuda() if torch.cuda.is_available() else model.cpu()

    # ===================== 3. 一键输出清晰的模型结构 =====================
    print("===================== UMambaEnc 3D 模型结构（分层+参数量） =====================")
    summary(
        model,
        input_size=(1, 1, 64, 64, 64),  # (batch_size, channels, D, H, W)
        device="cuda" if torch.cuda.is_available() else "cpu",
        col_width=20,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth = 4, 
        #verbose = 2,
        row_settings=["var_names", "depth"] # 显示每层变量名，更易读
    )