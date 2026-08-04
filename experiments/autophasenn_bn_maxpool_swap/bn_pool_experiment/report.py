"""Chinese Markdown report generation for the patent experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .model import AFFECTED_BN_LAYERS


DISPLAY_NAMES = {
    "paper_modulus_mae": "远场衍射模量 MAE",
    "real_amp_psnr": "实空间幅值 PSNR (dB)",
    "real_amp_ssim3d": "实空间幅值 3D-SSIM",
    "real_phase_mae_true_support": "真值 support 内包裹相位 MAE (rad)",
    "real_support_iou": "Support IoU",
    "real_support_dice": "Support Dice",
}


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "通过" if value else "未通过"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}g}"
    return str(value)


def _ci_text(summary: dict[str, Any]) -> str:
    low = summary.get("ci_low")
    high = summary.get("ci_high")
    if low is None or high is None:
        return "N/A"
    return f"[{_fmt(low)}, {_fmt(high)}]"


def _primary_table(summary: dict[str, Any], primary_metrics: list[str]) -> list[str]:
    lines = [
        "| 指标 | 方向 | 基线均值 | 交换后均值 | 交换-基线 | 配对差值95%CI | 相对退化 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    reconstruction = summary["reconstruction"]
    for key in primary_metrics:
        item = reconstruction.get(key)
        if item is None:
            continue
        direction = {"lower": "越低越好", "higher": "越高越好"}.get(
            item["direction"], "仅诊断"
        )
        lines.append(
            "| {name} | {direction} | {base} | {swap} | {delta} | {ci} | {deg}% |".format(
                name=DISPLAY_NAMES.get(key, key),
                direction=direction,
                base=_fmt(item["baseline"]["mean"]),
                swap=_fmt(item["swapped"]["mean"]),
                delta=_fmt(item["delta"]["mean"]),
                ci=_ci_text(item["delta"]),
                deg=_fmt(item["degradation_percent"]),
            )
        )
    return lines


def _bn_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| 池化相邻 BN | 通道数 | 正值 | 零值 | 负值 | 正值比例 | 有效缩放最小值 | 最大值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary["bn_scale_audit"].items():
        lines.append(
            f"| {name} | {item['channels']} | {item['positive']} | {item['zero']} | "
            f"{item['negative']} | {_fmt(100 * item['positive_fraction'])}% | "
            f"{_fmt(item['min_effective_scale'])} | {_fmt(item['max_effective_scale'])} |"
        )
    return lines


def _consistency_table(summary: dict[str, Any]) -> list[str]:
    requested = [
        ("farfield.mae", "远场输出 MAE"),
        ("farfield.relative_l1", "远场输出相对 L1"),
        ("farfield.pearson_corr", "远场输出 Pearson"),
        ("farfield.histogram_js_divergence", "远场直方图 JS 散度"),
        ("amplitude.mae", "幅值输出 MAE"),
        ("amplitude.relative_l1", "幅值输出相对 L1"),
        ("amplitude.histogram_js_divergence", "幅值直方图 JS 散度"),
        ("complex_object.mae", "复数物体输出复模差 MAE"),
        ("complex_object.relative_l2", "复数物体输出相对 L2"),
        ("complex_object.max_abs", "复数物体输出最大复模差"),
        ("phase.wrapped_mae", "包裹相位输出差 MAE (rad)"),
        ("support.disagreement_fraction", "Support 不一致体素比例"),
    ]
    lines = [
        "| 一致性指标 | 均值 | 标准差 | 95%CI |",
        "|---|---:|---:|---:|",
    ]
    for key, label in requested:
        item = summary["output_consistency"].get(key)
        if item is None:
            continue
        lines.append(
            f"| {label} | {_fmt(item['mean'])} | {_fmt(item['std'])} | {_ci_text(item)} |"
        )
    return lines


def _forward_output_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| 完整 forward 最终输出 | 基线形状 | 交换模型形状 | 数据类型 |",
        "|---|---:|---:|---:|",
    ]
    metadata = summary["model_comparison"]["output_metadata"]
    for name in summary["model_comparison"]["outputs_compared"]:
        baseline = metadata["baseline"][name]
        swapped = metadata["swapped"][name]
        lines.append(
            f"| {name} | `{baseline['shape']}` | `{swapped['shape']}` | "
            f"`{baseline['dtype']}` |"
        )
    return lines


def _layer_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| 编码块（末端 BN） | 同输入局部交换 MAE | 同输入局部交换相对L2 | 传播后块输出 MAE |",
        "|---|---:|---:|---:|",
    ]
    intrinsic = summary["layer_intrinsic"]
    propagated = summary["layer_propagated"]
    layer_names = sorted({key.split(".", 1)[0] for key in intrinsic})
    for layer in layer_names:
        local_mae = intrinsic.get(f"{layer}.mae", {}).get("mean")
        local_l2 = intrinsic.get(f"{layer}.relative_l2", {}).get("mean")
        propagated_mae = propagated.get(f"{layer}.mae", {}).get("mean")
        lines.append(
            f"| {layer} | {_fmt(local_mae)} | {_fmt(local_l2)} | {_fmt(propagated_mae)} |"
        )
    return lines


def _patent_sentence(summary: dict[str, Any]) -> str:
    if not summary.get("reconstruction_claim_supported", False):
        return (
            "本次使用随机生成输入，仅能验证层序交换的数学前提和网络输出一致性，"
            "不能形成 PSNR、SSIM 或相位重建精度的专利实验表述。"
        )
    reconstruction = summary["reconstruction"]
    n = summary["num_samples"]
    parts = []
    for metric in (
        "paper_modulus_mae",
        "real_amp_psnr",
        "real_amp_ssim3d",
        "real_phase_mae_true_support",
    ):
        item = reconstruction.get(metric)
        if item is None:
            continue
        parts.append(
            f"{DISPLAY_NAMES.get(metric, metric)}由{_fmt(item['baseline']['mean'])}变为"
            f"{_fmt(item['swapped']['mean'])}（相对退化{_fmt(item['degradation_percent'])}%）"
        )
    if not parts:
        return "本次运行未提供实空间真值，不能形成完整的重建精度专利表述。"
    status = "满足" if summary["acceptance"]["overall_pass"] else "未满足"
    return (
        f"在包含 {n} 个样本的固定验证集上，仅交换四处最大池化层与批量归一化层的"
        f"先后次序并保持其余网络参数不变，" + "；".join(parts) + "。"
        f"按本次预设工程阈值，结果{status}‘影响很小’的判据。"
    )


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    """Write a self-contained Chinese experiment report."""

    acceptance = summary["acceptance"]
    status = "通过" if acceptance["overall_pass"] else "未通过"
    monotonic = "满足" if acceptance["bn_scale_precondition_pass"] else "不满足"
    if summary.get("reconstruction_claim_supported", False):
        conclusion = f"本次实验判定为 **{status}**"
    else:
        conclusion = (
            f"本次随机输入一致性验证为 **{status}**；该结果不构成真实验证集上的"
            "重建精度证据"
        )
    if summary.get("reconstruction_claim_supported", False):
        accuracy_heading = "## 重建精度对照"
        accuracy_note = (
            "其中 3D-SSIM 使用配置指定的均匀三维滑动窗口；相位误差采用真值 "
            "support 内的包裹相位差，单位为弧度。相对退化统一定义为正值表示"
            "交换后更差。"
        )
    else:
        accuracy_heading = "## 随机输入拟合诊断（非重建精度证据）"
        accuracy_note = (
            "这里的远场 MAE 只是预训练模型对非物理随机输入的拟合诊断。由于没有"
            "真实物体幅值和相位真值，本节不能解释为重建精度。"
        )
    lines = [
        "# AutoPhaseNN 最大池化层与 BN 层交换量化验证报告",
        "",
        f"结论：{conclusion}。BN 单调递增前提 **{monotonic}**。",
        "",
        "## 实验对象与口径",
        "",
        f"- 验证样本数：{summary['num_samples']}；",
        f"- 输入数据模式：`{summary['data_mode']}`；",
        f"- 预训练权重：`{summary['checkpoint']['path']}`；",
        f"- 权重 SHA-256：`{summary['checkpoint'].get('sha256', '未计算')}`；",
        f"- 支持阈值：{_fmt(summary['threshold'])}；",
        "- 基线拓扑：`Conv3d -> LeakyReLU -> BN -> MaxPool3d`；",
        "- 交换拓扑：`Conv3d -> LeakyReLU -> MaxPool3d -> BN`；",
        "- 四个交换位置的权重、BN 运行均值/方差及其他网络参数完全相同；",
        "- 说明：专利正文写成 `Conv -> BN -> 激活 -> Pool`，与当前 AutoPhaseNN "
        "代码中的激活/BN 次序不同。本报告数字对应当前代码的真实拓扑。",
        "",
        "## 完整前向链路证明",
        "",
        f"- 独立模型实例：{_fmt(summary['model_comparison']['separate_model_instances'])}；",
        f"- 两模型 state_dict 逐张量完全一致：{_fmt(summary['model_comparison']['state_dicts_identical'])}；",
        f"- 两模型均独立调用完整 `forward()`：{_fmt(summary['model_comparison']['full_forward_called_for_each_model'])}；",
        "- 前向链路：输入衍射模量 -> 完整编码器 -> 幅值/相位双解码器 -> support -> "
        "复数物体 -> 三维 FFT -> 远场衍射模量；",
        f"- 输入统计：min={_fmt(summary['input_statistics']['minimum'])}，"
        f"max={_fmt(summary['input_statistics']['maximum'])}，"
        f"mean={_fmt(summary['input_statistics']['mean'])}，"
        f"std={_fmt(summary['input_statistics']['std'])}；",
        "",
        *_forward_output_table(summary),
        "",
        "## BN 有效缩放因子审计",
        "",
        "有效缩放因子定义为 `gamma / sqrt(running_var + eps)`。只有其为正时，"
        "BN 才在该通道内严格单调递增，最大池化交换才具有严格等价前提。",
        "",
        *_bn_table(summary),
        "",
        accuracy_heading,
        "",
        *_primary_table(summary, acceptance["primary_metrics"]),
        "",
        accuracy_note,
        "",
        "## 端到端输出与分布一致性",
        "",
        *_consistency_table(summary),
        "",
        "## 分层交换误差",
        "",
        *_layer_table(summary),
        "",
        "“同输入局部交换”在同一层的同一输入上直接比较两种顺序；“传播后块输出”"
        "包含前面交换误差向后传播造成的累计差异。",
        "",
        "## 预设判据",
        "",
        f"- 所有主指标最大允许相对退化：{_fmt(acceptance['criteria']['max_primary_metric_degradation_percent'])}%；",
        f"- 远场输出相对 L1 最大允许值：{_fmt(acceptance['criteria']['max_farfield_relative_l1'])}；",
        f"- 每个相邻 BN 的最小正缩放比例：{_fmt(100 * acceptance['criteria']['min_positive_bn_scale_fraction'])}%；",
        f"- 主指标判定：{_fmt(acceptance['primary_metrics_pass'])}；",
        f"- 输出一致性判定：{_fmt(acceptance['output_consistency_pass'])}；",
        f"- BN 单调性前提判定：{_fmt(acceptance['bn_scale_precondition_pass'])}；",
        "",
        "## 可用于说明书的量化表述草案",
        "",
        _patent_sentence(summary),
        "",
        "只有使用固定真实验证集及对应真值完成运行后，才能把重建精度数字写入专利。"
        "若真实验证实验未通过，不应在说明书中继续使用‘影响很小’这一无条件表述。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_final_output_comparison_report(path: Path, summary: dict[str, Any]) -> None:
    """Write a focused report comparing only the two models' primary outputs."""

    comparison = summary["final_output_comparison"]
    metrics = comparison["metrics"]
    identical = "完全一致" if comparison["exactly_identical"] else "存在差异"
    lines = [
        "# 两个 AutoPhaseNN 模型最终输出对比",
        "",
        "## 对比口径",
        "",
        "- 使用两个彼此独立的模型实例；",
        "- 两个模型严格加载同一预训练权重，state_dict 逐张量完全一致；",
        "- 模型A层序：`BN -> MaxPool3d`；",
        "- 模型B层序：`MaxPool3d -> BN`；",
        "- 两个模型均从随机输入开始完整执行编码器、双解码器、support、复数物体"
        "构建及三维 FFT；",
        "- 只比较两个模型的主输出 `outputs[0]`：最终远场衍射模量。",
        "",
        "## 输入与输出",
        "",
        f"- 随机输入样本数：{comparison['num_samples']}；",
        f"- 输入体素总数：{summary['input_statistics']['count']}；",
        f"- 单个最终输出形状：`{comparison['shape']}`；",
        f"- 输出数据类型：`{comparison['dtype']}`；",
        f"- 共比较输出元素数：{comparison['compared_element_count']}；",
        "",
        "## 数值差异",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| MAE | {_fmt(metrics['mae']['mean'])} |",
        f"| RMSE | {_fmt(metrics['rmse']['mean'])} |",
        f"| 全部样本最大绝对差 | {_fmt(comparison['maximum_absolute_difference_over_all_samples'])} |",
        f"| 相对 L1 | {_fmt(metrics['relative_l1']['mean'])} |",
        f"| 相对 L2 | {_fmt(metrics['relative_l2']['mean'])} |",
        f"| Pearson 相关系数 | {_fmt(metrics['pearson_corr']['mean'])} |",
        f"| 输出直方图 JS 散度 | {_fmt(metrics['histogram_js_divergence']['mean'])} |",
        "",
        f"结论：两个独立模型的最终主输出 **{identical}**。",
        "",
        "该结果只证明当前预训练权重和随机输入下的最终输出一致性；随机输入不提供"
        "真实重建 PSNR、SSIM 或相位误差证据。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_all_bn_scale_audit_report(path: Path, summary: dict[str, Any]) -> None:
    """Write a complete effective-scale audit for every BN layer."""

    aggregate = summary["all_bn_scale_summary"]
    lines = [
        "# AutoPhaseNN 全部 BN 层有效缩放因子审计",
        "",
        r"审计量：$a=\gamma/\sqrt{\mathrm{running\_var}+\epsilon}$。".replace("\\\\", "\\"),
        "",
        f"- BN 层数：{aggregate['layer_count']}；",
        f"- BN 通道总数：{aggregate['channel_count']}；",
        f"- 正值通道：{aggregate['positive']}；",
        f"- 零值通道：{aggregate['zero']}；",
        f"- 负值通道：{aggregate['negative']}；",
        f"- 是否全部大于 0：{_fmt(aggregate['all_positive'])}；",
        f"- 全局最小值：{_fmt(aggregate['global_min_effective_scale'])}；",
        f"- 全局最大值：{_fmt(aggregate['global_max_effective_scale'])}；",
        "",
        "| BN 层 | 参与池化交换 | 通道数 | 正值 | 零值 | 负值 | 正值比例 | 最小 a | 最大 a |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary["all_bn_scale_audit"].items():
        affected = "是" if name in AFFECTED_BN_LAYERS else "否"
        lines.append(
            f"| {name} | {affected} | {item['channels']} | {item['positive']} | "
            f"{item['zero']} | {item['negative']} | "
            f"{_fmt(100 * item['positive_fraction'])}% | "
            f"{_fmt(item['min_effective_scale'])} | {_fmt(item['max_effective_scale'])} |"
        )
    lines.extend(
        [
            "",
            "只有标记为“参与池化交换”的四个 BN 层决定本实验中 BN/MaxPool 的严格"
            "等价条件；其他 BN 的符号用于完整模型审计，但不直接参与该层序交换。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
