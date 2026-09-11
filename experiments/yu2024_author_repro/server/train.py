"""Server entry point using the unchanged upstream complex U-Net (FP32)."""
# DOMAIN: custom. Author-faithful reproduction: no AMP, clipping or architecture changes.
import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data import ROOT, load_arrays, sha256

sys.path.insert(0, str(ROOT / "vendor/Complex-NNphase/Supervised training"))
from model import Unet

LOG = logging.getLogger(__name__)


@dataclass
class Config:
    data_dir: str
    run_dir: str
    epochs: int = 500
    lr: float = 0.001
    min_lr: float = 0.0001
    batch_size: int = 64
    num_workers: int = 0
    seed: int = 1
    device: str = "cuda:0"
    threads: int = 4
    stop_after: int = 0
    overfit_samples: int = 0
    resume: str = ""


class Samples(Dataset):
    def __init__(self, arrays, ids, scale):
        self.arrays, self.ids, self.scale = arrays, ids, scale

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        idx = self.ids[index]
        # Stored detector data are already amplitude. Keep upstream's equal Re/Im input.
        amplitude = self.arrays["modulus_detector"][idx] / self.scale
        x = np.stack((amplitude, amplitude), axis=-1)[None]
        amp = self.arrays["object_modulus"][idx]
        phase = self.arrays["object_phase"][idx]
        target = np.stack((amp * np.cos(phase), amp * np.sin(phase)), axis=-1)[None]
        return torch.from_numpy(x), torch.from_numpy(target)


def finite_tensors(tensors, label):
    checks = [torch.isfinite(value).all() for value in tensors if value is not None]
    if checks and not torch.stack(checks).all().item():
        raise FloatingPointError(f"Non-finite {label}; stopping before saving an invalid checkpoint")


def check_optimizer(model, optimizer):
    finite_tensors(list(model.parameters()), "model parameters")
    values = [state[key] for state in optimizer.state.values()
              for key in ("exp_avg", "exp_avg_sq") if key in state]
    finite_tensors(values, "Adam moments")


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total = zero = amp_pred = amp_true = 0.0
    count = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(inputs)
            # Same author objective: L1(Re) + L1(Im).
            loss = (outputs - targets).abs().sum(dim=-1).mean()
            finite_tensors([loss], "loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                finite_tensors([parameter.grad for parameter in model.parameters()], "gradients")
                optimizer.step()
                check_optimizer(model, optimizer)
        n = len(inputs)
        count += n
        total += float(loss.detach()) * n
        zero += float(targets.abs().sum(dim=-1).mean()) * n
        amp_pred += float(torch.linalg.vector_norm(outputs.detach(), dim=-1).mean()) * n
        amp_true += float(torch.linalg.vector_norm(targets, dim=-1).mean()) * n
    return dict(mae=total / count, zero_mae=zero / count,
                predicted_amplitude_mean=amp_pred / count, target_amplitude_mean=amp_true / count)


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def write_status(directory, **state):
    (directory / "status.json").write_text(json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")


def train(config):
    out = Path(config.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if config.resume and Path(config.resume).resolve() != out / "last.pt":
        raise ValueError("Resume with --run-dir RUN --resume RUN/last.pt; preserve the complete run directory")
    if not config.resume and any(out.iterdir()):
        raise FileExistsError("Use an empty run directory or --resume RUN/last.pt")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "training.log", encoding="utf-8")])
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; install matching GPU PyTorch or explicitly use --device cpu")
    if config.epochs < 1 or config.batch_size < 1 or not 0 <= config.min_lr <= config.lr:
        raise ValueError("Invalid epochs, batch size or learning rates")
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    arrays, splits, manifest = load_arrays(Path(config.data_dir))
    if config.seed != manifest["seed"]:
        raise ValueError("Training seed must match the dataset split seed")
    # Use training samples only to fit preprocessing; reuse at validation/test/inference.
    scale = max(float(arrays["modulus_detector"][i].max()) for i in splits["train"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid training amplitude scale")
    train_ids, val_ids = splits["train"], splits["validation"]
    if config.overfit_samples:
        if not 0 < config.overfit_samples <= len(train_ids):
            raise ValueError("Invalid --overfit-samples")
        train_ids = train_ids[:config.overfit_samples]
        val_ids = train_ids  # Explicit diagnostic only, never a validation/test result.
    trainloader = DataLoader(Samples(arrays, train_ids, scale), batch_size=config.batch_size,
                             shuffle=True, num_workers=config.num_workers)
    validloader = DataLoader(Samples(arrays, val_ids, scale), batch_size=config.batch_size,
                             shuffle=False, num_workers=config.num_workers)
    model = Unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)
    manifest_hash = sha256(Path(config.data_dir) / "manifest.json")
    best, start = float("inf"), 0
    if config.resume:
        # Only load checkpoints produced by this script from a trusted run.
        state = torch.load(config.resume, map_location="cpu", weights_only=False)
        for key in ("epochs", "lr", "min_lr", "batch_size", "seed", "overfit_samples", "num_workers"):
            if state["config"][key] != asdict(config)[key]:
                raise ValueError(f"Resume must preserve {key}")
        if state["dataset_manifest_sha256"] != manifest_hash or state["scale"] != scale:
            raise ValueError("Resume dataset/preprocessing differs")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and state["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng_state"], device)
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        start, best = state["epoch"], state["best_val_mae"]
        check_optimizer(model, optimizer)
    np.savez(out / "split_indices.npz", **splits)
    environment = dict(torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    (out / "config.json").write_text(json.dumps(dict(config=asdict(config), environment=environment,
        scale=scale, dataset_manifest_sha256=manifest_hash, dataset_schema=manifest["schema"]), indent=2), encoding="utf-8")
    completed = start
    try:
        stop = min(config.epochs, config.stop_after or config.epochs)
        for epoch in range(start, stop):
            started = time.perf_counter()
            lr_used = optimizer.param_groups[0]["lr"]
            training = run_epoch(model, trainloader, device, optimizer)
            validation = run_epoch(model, validloader, device)
            scheduler.step()  # Exactly once per completed epoch.
            improved = validation["mae"] < best
            best = min(best, validation["mae"])
            completed = epoch + 1
            record = dict(epoch=completed, lr=lr_used, next_lr=optimizer.param_groups[0]["lr"],
                          train=training, validation=validation, seconds=time.perf_counter() - started,
                          diagnostic_overfit=bool(config.overfit_samples))
            state = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                         scheduler_state_dict=scheduler.state_dict(), epoch=completed, best_val_mae=best,
                         config=asdict(config), metrics=record, scale=scale, dataset_manifest_sha256=manifest_hash,
                         torch_rng_state=torch.get_rng_state(),
                         cuda_rng_state=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                         python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state())
            save_checkpoint(out / "last.pt", state)
            if improved:
                save_checkpoint(out / "best.pt", state)
            record["seconds"] = time.perf_counter() - started
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            write_status(out, state="running", **record)
            LOG.info("Epoch %d/%d train=%.6g val=%.6g zero=%.6g amp_ratio=%.4g lr=%.6g %.1fs",
                     completed, config.epochs, training["mae"], validation["mae"], validation["zero_mae"],
                     validation["predicted_amplitude_mean"] / validation["target_amplitude_mean"], lr_used, record["seconds"])
        write_status(out, state="complete" if completed == config.epochs else "paused", completed_epochs=completed,
                     best_val_mae=best, diagnostic_overfit=bool(config.overfit_samples))
    except Exception as error:
        write_status(out, state="failed", completed_epochs=completed, error=repr(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    for name in ("epochs", "batch_size", "num_workers", "seed", "threads", "stop_after", "overfit_samples"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=Config.__dataclass_fields__[name].default)
    for name in ("lr", "min_lr"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=Config.__dataclass_fields__[name].default)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", default="")
    train(Config(**vars(parser.parse_args())))
