import datetime
import errno
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if not isinstance(v, (float, int)):
                raise TypeError(
                    f"This method expects the value of the input arguments to be of type float or int, instead  got {type(v)}"
                )
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def reduce_across_processes(val, op=dist.ReduceOp.SUM):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t, op=op)
    return t


class CombinedDiffractionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def loss_log(self, Y_true, Y_pred):
        pred = torch.log10(Y_pred + 1)
        true = torch.log10(Y_true + 1)
        top    = torch.sum((pred - true) ** 2)
        bottom = torch.sum(true ** 2)
        return top / bottom

    def loss_sq(self, Y_true, Y_pred):
        top    = torch.sum((Y_pred - Y_true) ** 2, dim=(1, 2, 3), keepdim=True)
        bottom = torch.sum(Y_true ** 2,             dim=(1, 2, 3), keepdim=True)
        return torch.mean(top / bottom)

    def loss_mae(self, Y_true, Y_pred):
        top    = torch.sum(torch.abs(Y_pred - Y_true), dim=(1, 2, 3), keepdim=True)
        bottom = torch.sum(torch.abs(Y_true),           dim=(1, 2, 3), keepdim=True)
        return torch.mean(top / bottom)

    def loss_pcc(self, Y_true, Y_pred):
        pred = Y_pred - Y_pred.mean(dim=(1, 2, 3), keepdim=True)
        true = Y_true - Y_true.mean(dim=(1, 2, 3), keepdim=True)
        top      = torch.sum(pred * true,  dim=(1, 2, 3), keepdim=True)
        pred_sum = torch.sum(pred ** 2,    dim=(1, 2, 3), keepdim=True)
        true_sum = torch.sum(true ** 2,    dim=(1, 2, 3), keepdim=True)
        bottom   = torch.sqrt(pred_sum * true_sum)
        return torch.mean(1 - top / bottom)

    def loss_comb(self, Y_true, Y_pred):
        return (self.loss_sq(Y_true, Y_pred) + self.loss_pcc(Y_true, Y_pred)) / 2

    def loss_comb2(self, Y_true, Y_pred):
        return (torch.sqrt(self.loss_sq(Y_true, Y_pred)) + self.loss_pcc(Y_true, Y_pred)) / 2

    def loss_comb_log(self, Y_true, Y_pred):
        l1 = self.loss_sq(Y_true, Y_pred)
        l2 = self.loss_pcc(Y_true, Y_pred)
        l3 = self.loss_log(Y_true, Y_pred)
        return (50 * l1 + 50 * l2 + 1 * l3) / 101

    def forward(self, y, ft_images, pred_amps, amps, pred_phs, phs, support,
                supervised=True, w_f=2.0, w_a=1.0, w_p=1.0):

        y          = y.float()
        ft_images  = ft_images.float()
        pred_amps  = pred_amps.float()
        amps       = amps.float()
        pred_phs   = pred_phs.float()
        phs        = phs.float()

        loss_f = self.loss_log(ft_images, y)
        loss_a = self.loss_sq(amps, pred_amps)
        loss_p = self.loss_comb2(phs * support, pred_phs * support)

        def _bad(t):
            return torch.isnan(t) or torch.isinf(t)

        if _bad(loss_f) or _bad(loss_a) or _bad(loss_p):
            print(f"[WARN] bad loss — f={loss_f.item():.4e}, "
                  f"a={loss_a.item():.4e}, p={loss_p.item():.4e}")
            return None, None

        total = (w_f * loss_f + w_a * loss_a + w_p * loss_p) if supervised else loss_f

        details = {
            'loss_f':     loss_f.item(),
            'loss_a':     loss_a.item(),
            'loss_p':     loss_p.item(),
            'loss_total': total.item(),
        }
        return total, details


class LossComb2(nn.Module):
    def __init__(self):
        super(LossComb2, self).__init__()
        # 对应 TF 隐式处理：防止分母为 0
        self.epsilon = 1e-8

    def loss_sq(self, y_true, y_pred):
        """
        对应 TF:
        top = tf.reduce_sum(tf.math.square(Y_pred - Y_true), axis=(1,2,3), keepdims=True)
        bottom = tf.reduce_sum(tf.math.square(Y_true), axis=(1,2,3), keepdims=True)
        loss_value = tf.reduce_sum(top / bottom)
        """
        # 自动识别维度：对 Batch (dim 0) 以外的所有维度求和
        # PyTorch 格式通常是 (B, C, D, H, W)，所以这里对 (1, 2, 3, 4) 求和
        # 这涵盖了 TF 的 axis=(1, 2, 3) 空间维度
        dims = tuple(range(1, y_pred.ndim))
        
        # 分子：(pred - true)^2
        top = torch.sum((y_pred - y_true) ** 2, dim=dims)
        
        # 分母：true^2
        bottom = torch.sum(y_true ** 2, dim=dims) + self.epsilon
        
        # ⚠️ 关键修改：TF 使用 reduce_sum (总和)，而不是 mean (平均)
        return torch.sum(top / bottom)

    def loss_pcc(self, y_true, y_pred):
        """
        对应 TF:
        pred = Y_pred - tf.reduce_mean(...)
        ...
        loss_value = tf.reduce_sum(1 - top / bottom)
        """
        dims = tuple(range(1, y_pred.ndim))
        
        # 1. 减去均值 (Center the data)
        pred_mean = torch.mean(y_pred, dim=dims, keepdim=True)
        true_mean = torch.mean(y_true, dim=dims, keepdim=True)
        
        pred_centered = y_pred - pred_mean
        true_centered = y_true - true_mean
        
        # 2. 分子：协方差
        numerator = torch.sum(pred_centered * true_centered, dim=dims)
        
        # 3. 分母：标准差乘积
        pred_sq_sum = torch.sum(pred_centered ** 2, dim=dims)
        true_sq_sum = torch.sum(true_centered ** 2, dim=dims)
        denominator = torch.sqrt(pred_sq_sum * true_sq_sum) + self.epsilon
        
        # PCC
        pcc = numerator / denominator
        
        # ⚠️ 关键修改：TF 使用 reduce_sum (总和)
        return torch.sum(1.0 - pcc)

    def forward(self, y_pred, y_true):
        """
        对应 TF loss_comb2:
        loss_value = (a1*loss_1 + a2*loss_2) / (a1+a2)
        其中 loss_1 是 sqrt(loss_sq)
        """
        # 计算两个子损失
        # 注意参数顺序：虽然 PyTorch 习惯 (pred, true)，但为了不混淆，
        # 内部逻辑要保证减法方向一致 (平方后没区别，但好习惯保持一致)
        l_sq = self.loss_sq(y_true, y_pred)
        l_pcc = self.loss_pcc(y_true, y_pred)
        
        # 组合
        # TF: a1=1, a2=1 -> (sqrt(l_sq) + l_pcc) / 2
        # 注意：这里 l_sq 和 l_pcc 已经是 Batch 的总和了
        # 直接开根号可能会改变 Sum 的物理意义，但我们需要严格复刻 TF 的公式逻辑：
        # TF loss_comb2 里: loss_1 = tf.math.sqrt(loss_sq(...))
        # ⚠️ 注意 TF 的执行顺序：
        # TF loss_sq 返回的是一个标量 (Sum over batch)。
        # 所以 tf.math.sqrt(loss_sq) 是对“整个Batch的总误差”开根号。
        
        # 🚨 再次确认 TF 代码逻辑：
        # loss_sq 函数内部直接返回了 reduce_sum(top/bottom)。
        # 所以 loss_1 = sqrt(整个Batch的SQ损失之和)。
        
        loss = (torch.sqrt(l_sq) + l_pcc) / 2.0
        
        return loss



# --- 对应 TF 索引 89: phi (Lambda) ---
class PhiLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # 注册 PI 为常量
        #self.register_buffer('pi', torch.tensor(np.pi, dtype=torch.float32))

    def forward(self, x):
        # 假设输入 x 已经是 decoder2 的输出 (Tanh)
        return x * torch.pi 

#--- 对应 TF 索引 91: support (Lambda) ---
class SupportLayer(nn.Module):
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

    def forward(self, amp):
        # 生成 0/1 Mask
        return torch.where(
            amp >= self.threshold,
            torch.ones_like(amp),
            torch.zeros_like(amp)
        )

# class SupportLayer(nn.Module):
#     def __init__(self, threshold=0.1):
#         super().__init__()
#         self.threshold = threshold

#     def forward(self, amp):
#         # 用 sigmoid 近似阶跃函数，处处可微
#         # steepness 控制陡峭程度，越大越接近原来的硬阈值
#         steepness = 50.0
#         return torch.sigmoid(steepness * (amp - self.threshold))

# --- 对应 TF 索引 90: Obj (Lambda) ---
class ObjLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, amp, phi):
        # 合成复数场: Amp * e^(j*Phi)
        return torch.polar(amp, phi)

# --- 对应 TF 索引 92: masked_obj (Lambda) ---
class MaskedObjLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, obj, support):
        # 复数 Obj * 实数 Support
        # 需要把 support 转为 complex 才能相乘 (或者利用广播)
        return obj * support.type_as(obj)

# --- 对应 TF 索引 93: farfield_diff (Lambda) ---
class FarfieldDiffLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, masked_obj):
        # 对应 TF: ifftshift -> fft3d -> fftshift -> abs
        # 针对 PT 格式 (B, C, D, H, W)，对最后三维操作
        spatial_dims = (-3, -2, -1)
        
        x = torch.fft.ifftshift(masked_obj, dim=spatial_dims)
        x = torch.fft.fftn(x, dim=spatial_dims, norm=None)
        x = torch.fft.fftshift(x, dim=spatial_dims)
        
        #intensity = torch.abs(x).to(torch.float32)
        # ✅ 数值稳定的 abs，物理含义完全不变
        intensity = torch.sqrt(x.real**2 + x.imag**2 + 1e-8).to(torch.float32)
        #intensity = (x.real ** 2 + x.imag ** 2).to(torch.float32)
        return intensity




def loss_log(y_true, y_pred):
    pred = torch.log10(y_pred + 1)
    true = torch.log10(y_true + 1)
    top = torch.sum((pred - true) ** 2)
    bottom = torch.sum(true ** 2)
    return top / bottom


def loss_sq(y_true, y_pred):
    top    = torch.sum((y_pred - y_true) ** 2, dim=(1, 2, 3), keepdim=True)
    bottom = torch.sum(y_true ** 2,             dim=(1, 2, 3), keepdim=True)
    return torch.sum(top / bottom)


def loss_mae(y_true, y_pred):
    top    = torch.sum(torch.abs(y_pred - y_true), dim=(1, 2, 3), keepdim=True)
    bottom = torch.sum(torch.abs(y_true),           dim=(1, 2, 3), keepdim=True)
    return torch.sum(top / bottom)


def loss_paper(y_true, y_pred):
    sqrt_true  = torch.sqrt(y_true)
    sqrt_pred  = torch.sqrt(y_pred)
    abs_error  = torch.abs(sqrt_pred - sqrt_true)
    total_error = torch.sum(abs_error)
    return total_error / (64 * 64 * 64)


def loss_pcc(y_true, y_pred):
    pred = y_pred - y_pred.mean(dim=(1, 2, 3), keepdim=True)
    true = y_true - y_true.mean(dim=(1, 2, 3), keepdim=True)

    top      = torch.sum(pred * true,  dim=(1, 2, 3), keepdim=True)
    pred_sum = torch.sum(pred ** 2,    dim=(1, 2, 3), keepdim=True)
    true_sum = torch.sum(true ** 2,    dim=(1, 2, 3), keepdim=True)
    bottom   = torch.sqrt(pred_sum * true_sum)

    return torch.sum(1 - top / bottom)


def loss_comb(y_true, y_pred):
    a1, a2 = 1, 1
    return (a1 * loss_sq(y_true, y_pred) + a2 * loss_pcc(y_true, y_pred)) / (a1 + a2)


def loss_comb2(y_true, y_pred):
    a1, a2 = 1, 1
    return (a1 * torch.sqrt(loss_sq(y_true, y_pred)) + a2 * loss_pcc(y_true, y_pred)) / (a1 + a2)


def loss_comb_log(y_true, y_pred):
    a1, a2, a3 = 50, 50, 1
    return (
        a1 * loss_sq(y_true, y_pred)  +
        a2 * loss_pcc(y_true, y_pred) +
        a3 * loss_log(y_true, y_pred)
    ) / (a1 + a2 + a3)


# ───────────────────────── 损失函数选择（替换 compile） ─────────────────────────

def get_criterion(loss_type: str):
    """
    返回对应的损失函数，签名统一为 criterion(y_pred, y_true)。
    在训练循环中直接调用：loss = criterion(output, target)
    """
    _map = {
        'mae':      nn.L1Loss(),                        # 内置 MAE
        'mae_cus':  loss_mae,                           # 自定义归一化 MAE
        'mse':      loss_sq,                            # 自定义归一化 MSE
        'huber':    nn.HuberLoss(),                     # 内置 Huber
        'pcc':      loss_pcc,                           # Pearson 相关系数
        'comb':     loss_comb,                          # sq + pcc
        'comb2':    loss_comb2,                         # sqrt(sq) + pcc
    }
    # 未匹配时退回到 MAE
    return _map.get(loss_type, nn.L1Loss())


# ───────────────────────────── 训练循环示例 ─────────────────────────────

# criterion = get_criterion(args.loss_type)
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#
# for x, y in dataloader:
#     optimizer.zero_grad()
#     output = model(x)
#     loss   = criterion(y, output)   # 注意：自定义函数签名是 (y_true, y_pred)
#     loss.backward()
#     optimizer.step()