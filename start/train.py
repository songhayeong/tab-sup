import argparse
import math
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from . import utils
from .data import Dataset, Transformations, prepare_torch_dataloader
from .ema import ExponentialMovingAverage
from .losses import get_optimizer, get_step_fn, optimization_manager
from .model import modules as model_modules
from .noise_lib import get_noise
from .pipeline import prepare_dataset_and_graph


@dataclass
class DatasetConfig:
    path: str
    cache: bool = True
    batch_size: int = 128
    eval_batch_size: Optional[int] = None
    shuffle: bool = True
    num_workers: int = 0
    drop_last: bool = False


@dataclass
class GraphConfig:
    type: str = "block_uniform"
    group_sizes: Optional[List[int]] = None


@dataclass
class NoiseConfig:
    type: str = "geometric"
    sigma_min: float = 1e-3
    sigma_max: float = 1.0


@dataclass
class ModelConfig:
    type: str = "mlp"
    is_y_cond: bool = False
    dim_t: int = 128
    num_classes: Optional[int] = None
    rtdl_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimConfig:
    optimizer: str = "AdamW"
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    warmup: int = 0
    grad_clip: float = -1.0


@dataclass
class TrainLoopConfig:
    epochs: int = 100
    grad_accum: int = 1
    log_every: int = 50
    eval_every: int = 500
    checkpoint_every: int = 1000
    checkpoint_dir: Optional[str] = None
    resume: Optional[str] = None
    ema_decay: float = 0.999
    max_steps: Optional[int] = None
    seed: int = 0
    device: Optional[str] = None


@dataclass
class Config:
    dataset: DatasetConfig
    transformations: Transformations
    graph: GraphConfig
    noise: NoiseConfig
    model: ModelConfig
    optim: OptimConfig
    train: TrainLoopConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_dataloaders(
    dataset: Dataset,
    cfg: DatasetConfig,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_loader = prepare_torch_dataloader(
        dataset=dataset,
        split="train",
        shuffle=cfg.shuffle,
        batch_size=cfg.batch_size,
    )

    val_loader = None
    if "val" in dataset.y:
        val_loader = prepare_torch_dataloader(
            dataset=dataset,
            split="val",
            shuffle=False,
            batch_size=cfg.eval_batch_size or cfg.batch_size,
        )

    return train_loader, val_loader


def _count_tokens(dataset: Dataset) -> int:
    if dataset.X_num is not None:
        return dataset.X_num["train"].shape[1]
    if dataset.X_cat is not None:
        return dataset.X_cat["train"].shape[1]
    raise ValueError("Dataset does not contain numerical or categorical features.")


def build_model(cfg: ModelConfig, dataset: Dataset, input_dim: int) -> torch.nn.Module:
    model_type = cfg.type.lower()
    num_classes = cfg.num_classes
    if num_classes is None:
        num_classes = dataset.n_classes or 0

    rtdl_params = dict(cfg.rtdl_params)
    if model_type == "mlp":
        return model_modules.MLPDiffusion(
            d_in=input_dim,
            num_classes=num_classes,
            is_y_cond=cfg.is_y_cond,
            rtdl_params=rtdl_params,
            dim_t=cfg.dim_t,
        )
    elif model_type == "resnet":
        return model_modules.ResNetDiffusion(
            d_in=input_dim,
            num_classes=num_classes,
            rtdl_params=rtdl_params,
            dim_t=cfg.dim_t,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.type}")


def save_checkpoint(
    path: Path,
    state: Dict[str, Any],
    config: Config,
    epoch: int,
) -> None:
    checkpoint = {
        "model": state["model"].state_dict(),
        "optimizer": state["optimizer"].state_dict(),
        "scaler": state["scaler"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
        "epoch": epoch,
        "config": asdict(config),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    state: Dict[str, Any],
    device: torch.device,
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)
    state["model"].load_state_dict(checkpoint["model"])
    state["optimizer"].load_state_dict(checkpoint["optimizer"])
    state["scaler"].load_state_dict(checkpoint["scaler"])
    state["ema"].load_state_dict(checkpoint["ema"])

    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def evaluate(
    eval_loader: DataLoader,
    step_fn,
    state: Dict[str, Any],
    device: torch.device,
    model_cfg: ModelConfig,
) -> float:
    losses = []
    with torch.no_grad():
        for batch, cond in _iter_batches(eval_loader, device, model_cfg.is_y_cond):
            loss = step_fn(state, batch, cond=cond)
            losses.append(loss.item())
    return float(sum(losses) / max(len(losses), 1))


def _iter_batches(loader: DataLoader, device: torch.device, use_cond: bool):
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            data, meta = batch
            cond = meta.get("y") if isinstance(meta, dict) else None
        else:
            data, cond = batch, None

        data = data.to(device)
        if use_cond and cond is not None:
            cond = cond.to(device)
        else:
            cond = None

        yield data.long(), cond


def train(config: Config) -> None:
    set_seed(config.train.seed)

    device = _resolve_device(config.train.device)
    dataset, graph = prepare_dataset_and_graph(
        config,
        config.dataset.path,
        config.transformations,
        device,
        cache=config.dataset.cache,
    )

    train_loader, val_loader = _get_dataloaders(dataset, config.dataset)
    input_dim = _count_tokens(dataset)
    model = build_model(config.model, dataset, input_dim).to(device)

    noise = get_noise(config)

    optimizer = get_optimizer(config, model.parameters())
    ema = ExponentialMovingAverage(model.parameters(), config.train.ema_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    state = {
        "model": model,
        "optimizer": optimizer,
        "ema": ema,
        "scaler": scaler,
        "step": 0,
    }

    if config.train.resume:
        resume_path = Path(config.train.resume)
        if resume_path.exists():
            epoch, step = load_checkpoint(resume_path, state, device)
            state["step"] = step
            start_epoch = epoch
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    else:
        start_epoch = 0

    optimize_fn = optimization_manager(config)
    train_step_fn = get_step_fn(noise, graph, train=True, optimize_fn=optimize_fn, accum=config.train.grad_accum)
    eval_step_fn = None
    if val_loader is not None:
        eval_step_fn = get_step_fn(noise, graph, train=False, optimize_fn=None, accum=1)

    best_val = math.inf
    steps_per_epoch = math.ceil(len(dataset.y["train"]) / config.dataset.batch_size)

    for epoch in range(start_epoch, config.train.epochs):
        if config.train.max_steps and state["step"] >= config.train.max_steps:
            break

        for batch_idx, (batch, cond) in enumerate(_iter_batches(train_loader, device, config.model.is_y_cond)):
            if config.train.max_steps and state["step"] >= config.train.max_steps:
                break

            prev_step = state["step"]
            loss = train_step_fn(state, batch, cond=cond)

            if state["step"] != prev_step:
                if state["step"] % config.train.log_every == 0:
                    print(f"[epoch {epoch} | step {state['step']}] loss: {loss.item():.4f}")

                if (
                    eval_step_fn is not None
                    and config.train.eval_every > 0
                    and state["step"] % config.train.eval_every == 0
                ):
                    val_loss = evaluate(val_loader, eval_step_fn, state, device, config.model)
                    print(f"Validation at step {state['step']}: loss {val_loss:.4f}")
                    best_val = min(best_val, val_loss)

                if (
                    config.train.checkpoint_every > 0
                    and config.train.checkpoint_dir
                    and state["step"] % config.train.checkpoint_every == 0
                ):
                    ckpt_path = Path(config.train.checkpoint_dir) / f"step_{state['step']:07d}.pt"
                    save_checkpoint(ckpt_path, state, config, epoch)

            if (batch_idx + 1) >= steps_per_epoch:
                break

        if val_loader is not None and eval_step_fn is not None:
            val_loss = evaluate(val_loader, eval_step_fn, state, device, config.model)
            print(f"[epoch {epoch}] Validation loss: {val_loss:.4f}")
            best_val = min(best_val, val_loss)

        if config.train.checkpoint_dir:
            ckpt_path = Path(config.train.checkpoint_dir) / f"epoch_{epoch:04d}.pt"
            save_checkpoint(ckpt_path, state, config, epoch)

    print("Training finished.")
    if val_loader is not None:
        print(f"Best validation loss: {best_val:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tab-SEDD model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a TOML config file describing training setup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_config = utils.load_config(args.config)
    config = utils.from_dict(Config, raw_config)
    train(config)


if __name__ == "__main__":
    main()
