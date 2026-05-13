import torch
from tqdm import tqdm
from slayers import *
import torch.nn as nn
import numpy as np
import random
import os

def snn_inference(dataloader, model, device, timesteps, st=16):
    for module in model.modules():
        if isinstance(module, (SpikingNeuron, SpikingNeuron2d, SpikingNeuron4d, SpikingNeuron5d)):
            module.mode = "snn"
    tot = np.zeros(timesteps)
    sops = np.zeros(timesteps)
    model.eval()
    model.to(device)
    length = 0
    length2 = 0
    with torch.no_grad():
        for img, label in tqdm(dataloader):
            img = img.to(device)
            label = label.to(device)
            outs = []
            for t in range(timesteps):
                outs.append(model(img))

            out_spike = torch.stack(outs, 0)
            out_spike[:st] = torch.cumsum(out_spike[:st], dim=0)
            out_spike[st:] = torch.cumsum(out_spike[st:], dim=0)
            reset_model(model)
            length += len(label)
            for i in range(timesteps):
                tot[i] += (label==out_spike[i].max(1)[1]).sum().data
    return tot/length

def evaluate(test_dataloader, model, device):
    tot = 0.
    model.eval()
    model.to(device)
    length = 0
    with torch.no_grad():
        for img, label in tqdm(test_dataloader):
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            # loss = loss_fn(out, label)
            length += len(label)    
            tot += (label==out.max(1)[1]).sum().data
    return tot/length

def convert_ann_to_snn(model, cnt=0):
    prev_module = None
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name], cnt = convert_ann_to_snn(module, cnt)
        if isinstance(module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU)):
            if isinstance(model._modules[prev_module], nn.BatchNorm2d):
                print("Convert Layer {}, ReLU to SpkingNeuron4d".format(cnt))
                model._modules[name] = SpikingNeuron4d(num_features=model._modules[prev_module].num_features)
            elif isinstance(model._modules[prev_module], nn.Linear):
                print("Convert Layer {}, ReLU to SpkingNeuron2d".format(cnt))
                model._modules[name] = SpikingNeuron2d(num_features=model._modules[prev_module].out_features)
            elif isinstance(model._modules[prev_module], nn.Sequential):
                print("Convert Layer {}, ReLU to SpkingNeuron5d".format(cnt))
                model._modules[name] = SpikingNeuron5d(num_features=model._modules[prev_module][-1].num_features)
            elif isinstance(model._modules[prev_module], nn.BatchNorm3d):
                print("Convert Layer {}, ReLU to SpkingNeuron5d".format(cnt))
                model._modules[name] = SpikingNeuron5d(num_features=model._modules[prev_module].num_features)
            elif isinstance(model._modules[prev_module], nn.Conv3d):
                print(f"Convert Layer {cnt}, Act to SpikingNeuron5d (prev: Conv3d)".format(cnt))
                model._modules[name] = SpikingNeuron5d(num_features=model._modules[prev_module].out_channels)
            else:
                raise AssertionError(prev_module)
            cnt += 1
        if isinstance(module, nn.MaxPool2d) and isinstance(model._modules[prev_module], SpikingNeuron4d):
            print("Using prev spike maxpooling")
            prev_feature = model._modules[prev_module].num_features
            model._modules[prev_module] = module
            model._modules[name] = SpikingNeuron4d(num_features=prev_feature)
        elif isinstance(module, nn.MaxPool2d) and isinstance(model._modules[prev_module], SpikingNeuron2d):
            print("Using prev spike maxpooling")
            prev_feature = model._modules[prev_module].num_features
            model._modules[prev_module] = module
            model._modules[name] = SpikingNeuron2d(num_features=prev_feature)
        
        prev_module = name
    return model, cnt

def convert_newmaxpool(model, lat):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = convert_newmaxpool(module, lat)
        if isinstance(module, (nn.MaxPool2d)):
            model._modules[name] = NewMaxPool2d(module, lat=lat)
    return model

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def reset_model(model):
    for module in model.modules():
        if isinstance(module, (SpikingNeuron, SpikingNeuron2d, SpikingNeuron4d, SpikingNeuron5d)):
            module.reset()
        elif isinstance(module, NewMaxPool2d):
            module.reset()
    return

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm # 推荐加上进度条

def weight_scaling_iter(dataloader, model, device, iter_steps, criterion):
    """
    SNN 阈值校准函数 (适配 3D 衍射成像任务 & 论文局部阈值平衡策略)
    
    Args:
        dataloader: 包含 (ft_images, amps, phs) 的数据加载器
        model: 已进行 BN 融合的模型 (处于 optimize/ann 模式)
        device: 计算设备
        iter_steps: 校准的迭代次数 (Paper 建议几百次即可收敛 )
    """
    
    # 2. 滑动窗口记录 Loss (用于监测成像质量)
    maxlen = 10
    loss_buffer = torch.zeros(maxlen) 
    
    # 3. 记录 Delta (用于监测 SNN 转换误差 [cite: 117])
    # 论文中定义的 e^l = ||SNN - ANN|| [cite: 118]
    losses_delta = [] 
    
    model.eval()
    model.to(device)
    
    print(f"🚀 Start Local Threshold Balancing for {iter_steps} iterations...")
    
    with torch.no_grad():
        # tqdm 用于显示进度
        loop = tqdm(enumerate(dataloader), total=iter_steps, leave=False)
        
        for i, (ft_images, amps, phs) in loop:
            # --- A. 数据加载 ---
            ft_images = ft_images.to(device)
            amps = amps.to(device)
            # amps, phs 虽然 ANN 训练用了，但在纯阈值校准阶段
            # 实际上只需要 ft_images 来驱动每一层的激活即可。
            # 但为了监测最终输出质量，我们还是保留 ground truth。
            
            # --- B. 前向传播 (驱动 SpikingNeuron.optimize) ---
            # 数据流过网络，触发 optimize 内部的 eq.14/15 更新规则 [cite: 161]
            y, _, pred_amps, pred_phs, support = model(ft_images)
            
            # --- C. 统计 SNN 转换误差 (Delta) ---
            ss = 0.
            layers = 0
            for module in model.modules():
                # 必须包含 SpikingNeuron5d 以支持 3D 卷积层
                # 论文提到 Channel-wise Balancing ，5d Neuron 天然支持这个
                if isinstance(module, (SpikingNeuron, SpikingNeuron2d, SpikingNeuron4d, SpikingNeuron5d)):

                    delta_value = module.delta # 获取 delta 属性/方法
                    
                    # 1. 检查是否是一个方法/函数（builtin_function_or_method）
                    if callable(delta_value):
                         # 如果是，调用它来获取数值
                         delta_value = delta_value()
                    
                    # 2. 检查是否是 PyTorch Tensor
                    if isinstance(delta_value, torch.Tensor):
                         # 如果是 Tensor，使用 .item() 提取 Python float
                         delta_value = delta_value.item()
                         
                    # 3. 累加（现在 ss 和 delta_value 都是 float 了）
                    ss += delta_value
                    layers += 1
            
            # 计算当前 Batch 所有层的平均转换误差
            avg_delta = ss / layers if layers > 0 else 0
            losses_delta.append(avg_delta)

            # --- D. 计算成像任务 Loss (验收标准) ---
            # 重点关注傅里叶域 Loss (Loss_FT)，这是 ANN 优化的主目标
            loss_f = criterion(pred_amps, amps)
            
            # --- E. 滑动平均记录 ---
            loss_buffer[i % maxlen] = loss_f.item()
            if i < maxlen:
                curr_loss = loss_buffer[:i+1].mean().item()
            else:
                curr_loss = loss_buffer.mean().item()
            
            # --- F. 进度显示 ---
            loop.set_description(f"Iter {i}/{iter_steps}")
            loop.set_postfix(Loss_FT=f"{curr_loss:.5f}", Delta=f"{avg_delta:.5f}")
            
            # 达到指定步数停止 (论文显示 1000 步以内通常已足够 )
            if i >= iter_steps:
                break
                
    print(f"✅ Calibration Finished. Final Loss_FT: {curr_loss:.5f}, Final Delta: {losses_delta[-1]:.5f}")
    return losses_delta



def weight_scaling_iter_new(dataloader, model, device, iter_steps, criterion):
    
    # 2. 滑动窗口记录 Loss (用于监测成像质量)
    maxlen = 10
    loss_buffer = torch.zeros(maxlen) 
    
    # 3. 记录 Delta (用于监测 SNN 转换误差)
    losses_delta = [] 
    
    model.eval()
    model.to(device)
    
    print(f"🚀 Start Local Threshold Balancing for {iter_steps} iterations...")
    
    # --- 修改点 1: 初始化全局步数计数器和进度条 ---
    global_step = 0
    pbar = tqdm(total=iter_steps, desc="Calibrating", leave=False)
    
    with torch.no_grad():
        # --- 修改点 2: 使用 while 循环确保跑满 iter_steps ---
        while global_step < iter_steps:
            
            # --- 修改点 3: 遍历 dataloader ---
            # 当这个 for 循环结束（数据集跑完一轮），while 会再次进入，
            # 从而重新从头开始遍历 dataloader
            for i, (ft_images, amps, phs) in enumerate(dataloader):
                
                # 如果达到了指定的总步数，强制退出内层循环
                if global_step >= iter_steps:
                    break

                # --- A. 数据加载 ---
                ft_images = ft_images.to(device)
                amps = amps.to(device)
                
                # --- B. 前向传播 (驱动 SpikingNeuron.optimize) ---
                y, _, pred_amps, pred_phs, support = model(ft_images)
                
                # --- C. 统计 SNN 转换误差 (Delta) ---
                ss = 0.
                layers = 0
                for module in model.modules():
                    # 这里保持原有逻辑，适配 3D/5D SNN
                    if isinstance(module, (SpikingNeuron, SpikingNeuron2d, SpikingNeuron4d, SpikingNeuron5d)):
                        delta_value = module.delta
                        if callable(delta_value):
                            delta_value = delta_value()
                        if isinstance(delta_value, torch.Tensor):
                            delta_value = delta_value.item()
                        
                        ss += delta_value
                        layers += 1
                
                avg_delta = ss / layers if layers > 0 else 0
                losses_delta.append(avg_delta)

                # --- D. 计算成像任务 Loss ---
                loss_f = criterion(pred_amps, amps)
                
                # --- E. 滑动平均记录 ---
                # 注意：这里使用 global_step 作为索引，保证跨 Epoch 时的连续性
                loss_buffer[global_step % maxlen] = loss_f.item()
                
                if global_step < maxlen:
                    curr_loss = loss_buffer[:global_step+1].mean().item()
                else:
                    curr_loss = loss_buffer.mean().item()
                
                # --- F. 进度显示与更新 ---
                global_step += 1
                pbar.update(1) # 手动更新进度条
                pbar.set_postfix(Loss_FT=f"{curr_loss:.5f}", Delta=f"{avg_delta:.5f}")

    pbar.close()
    
    # 防止 iter_steps 为 0 或其他异常导致 losses_delta 为空
    final_delta = losses_delta[-1] if len(losses_delta) > 0 else 0.0
    
    print(f"✅ Calibration Finished. Final Loss_FT: {curr_loss:.5f}, Final Delta: {final_delta:.5f}")
    return losses_delta

def _fold_bn(conv_module, bn_module, avg=False):
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def fold_bn_into_conv(conv_module, bn_module, avg=False):
    w, b = _fold_bn(conv_module, bn_module, avg)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w
    # set bn running stats
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data ** 2


def is_absorbing(m):
    return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear)


def search_fold_and_remove_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if isinstance(m, nn.BatchNorm2d) and is_absorbing(prev):
            fold_bn_into_conv(prev, m)
            setattr(model, n, Dummy())
        elif is_absorbing(m):
            prev = m
        else:
            prev = search_fold_and_remove_bn(m)
    return prev


def snn_inference_delay(dataloader, model, device, timesteps):
    for module in model.modules():
        if isinstance(module, (SpikingNeuron, SpikingNeuron2d, SpikingNeuron4d)):
            module.mode = "snn"

    tot = np.zeros((timesteps, timesteps))
    sops = np.zeros(timesteps)
    model.eval()
    model = model.to(device)
    length = 0
    length2 = 0
    cnt = 0
    with torch.no_grad():
        for img, label in tqdm(dataloader):
            cnt += 1
            img = img.to(device)
            label = label.to(device)
            outs = []
            for t in range(timesteps):
                outs.append(model(img))
                # sops[t] += get_energy(model)
            out_spike = torch.stack(outs, 0)
            # start from i-th timestep
            for i in range(timesteps):
                # out = torch.zeros_like(out_spike)
                # end at j-th timestep
                for j in range(i, timesteps):
                    tmp = out_spike[i:j+1,...].sum(0)
                    tot[i,j] += (label==tmp.max(1)[1]).sum().data
                # for k in range(timesteps):
                    # tot[k] += (label==out[k].max(1)[1]).sum().data
            
            reset_model(model)
            length += len(label)
            length2 += 1
            # if cnt>10:
            #     break
    return tot/length

import torch
from tqdm import tqdm

def cal_delay_time(dataloader, model, device):
    """
    计算 SNN 的脉冲传播延迟 (Delayed Step t0)。
    适配 3D 衍射成像任务 & SpikingNeuron5d。
    """
    model.eval()
    model = model.to(device)
    
    # 1. 切换模式为 "clip"
    # 在这个模式下，SpikingNeuron 会模拟 ANN 的行为，并计算 r 值 (Equation 19 中的分母部分)
    for module in model.modules():
        if isinstance(module, (SpikingNeuron, SpikingNeuron2d, SpikingNeuron4d, SpikingNeuron5d)):
            module.mode = "clip"
            
    print("🚀 Calculating estimated delay time (t0)...")
    
    with torch.no_grad():
        # 2. 只需要跑一个 Batch 的数据即可计算出延迟
        # 修改解包逻辑以匹配你的 DataLoader: (ft_images, amps, phs)
        for i, (ft_images, amps, phs) in enumerate(dataloader):
            ft_images = ft_images.to(device, non_blocking=True)
            
            # 触发前向传播，让 SpikingNeuron 内部计算 r = theta / max(mean(ReLU(z)))
            _ = model(ft_images)
            break  # 只跑一次
            
    # 3. 累加每层的延迟贡献
    # t0 ≈ sum( (theta - v(0)) / max_activation )
    # 代码中 r 存储的就是这一项
    total_delay = 0.
    layer_count = 0
    
    for name, module in model.named_modules():
        # 务必包含 SpikingNeuron5d
        if isinstance(module, (SpikingNeuron, SpikingNeuron2d, SpikingNeuron4d, SpikingNeuron5d)):
            
            # 鲁棒地获取 r 值
            if hasattr(module, 'r'):
                r_val = module.r
                if isinstance(r_val, torch.Tensor):
                    r_val = r_val.item()
                
                # 打印每层的延迟贡献，方便调试
                # print(f"Layer {name}: delay contribution = {r_val:.4f}")
                total_delay += r_val
                layer_count += 1
            else:
                print(f"Warning: Layer {name} has no attribute 'r'")

    estimated_t0 = int(total_delay)
    print(f"✅ Estimated Delay Time (t0): {estimated_t0} steps (over {layer_count} layers)")
    
    return estimated_t0