from __future__ import annotations

import importlib
import importlib.metadata
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass
from typing import Optional


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


def line(char: str = "=") -> None:
    print(char * 72)


def section(title: str) -> None:
    line()
    print(title)
    line()


def safe_package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
    except Exception:
        return "unavailable"


def format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else ("FAIL" if result.required else "WARN")
    print(f"[{status}] {result.name}: {result.detail}")


def run_nvidia_smi() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def import_torch():
    try:
        return importlib.import_module("torch"), None
    except Exception as exc:
        return None, exc


def import_module(module_name: str):
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:
        return None, exc


def check_causal_conv1d_runtime(torch, device) -> CheckResult:
    module, error = import_module("causal_conv1d")
    if error is not None:
        return CheckResult("causal_conv1d import", False, format_exception(error))

    try:
        causal_conv1d_fn = getattr(module, "causal_conv1d_fn")
        batch, dim, seqlen, width = 2, 8, 32, 4
        x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32, requires_grad=True)
        weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
        out = causal_conv1d_fn(x, weight, bias, activation="silu")
        if tuple(out.shape) != (batch, dim, seqlen):
            return CheckResult("causal_conv1d runtime", False, f"unexpected output shape: {tuple(out.shape)}")
        out.sum().backward()
        torch.cuda.synchronize(device)
    except Exception as exc:
        return CheckResult("causal_conv1d runtime", False, format_exception(exc))

    version = getattr(module, "__version__", safe_package_version("causal-conv1d"))
    return CheckResult("causal_conv1d runtime", True, f"version={version}")


def check_selective_scan_runtime(torch, device) -> CheckResult:
    module, error = import_module("mamba_ssm.ops.selective_scan_interface")
    if error is not None:
        return CheckResult("selective_scan import", False, format_exception(error))

    try:
        selective_scan_fn = getattr(module, "selective_scan_fn")
        batch, dim, seqlen, dstate = 2, 8, 16, 4
        u = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32, requires_grad=True)
        delta = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32, requires_grad=True)
        A = (-torch.rand(dim, dstate, device=device, dtype=torch.float32)).requires_grad_()
        B = torch.randn(batch, dstate, seqlen, device=device, dtype=torch.float32, requires_grad=True)
        C = torch.randn(batch, dstate, seqlen, device=device, dtype=torch.float32, requires_grad=True)
        D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
        delta_bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
        out = selective_scan_fn(
            u,
            delta,
            A,
            B,
            C,
            D=D,
            delta_bias=delta_bias,
            delta_softplus=True,
        )
        if tuple(out.shape) != (batch, dim, seqlen):
            return CheckResult("selective_scan runtime", False, f"unexpected output shape: {tuple(out.shape)}")
        out.mean().backward()
        torch.cuda.synchronize(device)
    except Exception as exc:
        return CheckResult("selective_scan runtime", False, format_exception(exc))

    version = safe_package_version("mamba-ssm")
    return CheckResult("selective_scan runtime", True, f"mamba-ssm version={version}")


def check_mamba_module_runtime(torch, device) -> CheckResult:
    try:
        from mamba_ssm import Mamba

        model = Mamba(d_model=16, d_state=8, d_conv=4, expand=2).to(device)
        model.train()
        x = torch.randn(2, 32, 16, device=device, dtype=torch.float32, requires_grad=True)
        y = model(x)
        if tuple(y.shape) != tuple(x.shape):
            return CheckResult("Mamba module runtime", False, f"unexpected output shape: {tuple(y.shape)}")
        y.square().mean().backward()
        torch.cuda.synchronize(device)
    except Exception as exc:
        return CheckResult("Mamba module runtime", False, format_exception(exc))

    return CheckResult("Mamba module runtime", True, "forward/backward passed")


def main() -> int:
    section("Environment summary")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")

    smi_output = run_nvidia_smi()
    if smi_output:
        print("nvidia-smi:")
        print(smi_output)
    else:
        print("nvidia-smi: unavailable")

    torch, torch_error = import_torch()
    if torch_error is not None:
        print_result(CheckResult("PyTorch import", False, format_exception(torch_error)))
        return 2

    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    results: list[CheckResult] = []

    causal_module, causal_error = import_module("causal_conv1d")
    if causal_error is None:
        version = getattr(causal_module, "__version__", safe_package_version("causal-conv1d"))
        results.append(CheckResult("causal_conv1d import", True, f"version={version}"))
    else:
        results.append(CheckResult("causal_conv1d import", False, format_exception(causal_error)))

    mamba_module, mamba_error = import_module("mamba_ssm")
    if mamba_error is None:
        version = getattr(mamba_module, "__version__", safe_package_version("mamba-ssm"))
        results.append(CheckResult("mamba_ssm import", True, f"version={version}"))
    else:
        results.append(CheckResult("mamba_ssm import", False, format_exception(mamba_error)))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Active GPU: {torch.cuda.get_device_name(0)}")
        results.append(check_causal_conv1d_runtime(torch, device))
        results.append(check_selective_scan_runtime(torch, device))
        results.append(check_mamba_module_runtime(torch, device))
    else:
        results.append(
            CheckResult(
                "CUDA runtime checks",
                False,
                "GPU not visible to PyTorch; import checks ran, runtime checks skipped",
            )
        )

    section("Check results")
    for result in results:
        print_result(result)

    failed_required = [result for result in results if result.required and not result.ok]

    section("Interpretation")
    if failed_required:
        print("Mamba is not healthy enough for training yet.")
        print("Fix the failed items above before trusting server-side training.")
        return 1

    print("Mamba core installation looks healthy for this environment.")
    print("This is strong evidence, not an absolute proof, that full training will work.")
    print("Remaining risks are project-specific code paths, mixed-precision issues, and data-dependent failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
