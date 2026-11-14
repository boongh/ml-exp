"""Parameter-matched Transformer/RNN sweep: python -m ml_exp."""

import json
import math
import os
import random
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import wandb

from .data import Dataset
from .rnn.rnn import RNNLangModel, RNNLangModelHyperparam
from .transformer.transformer import LangModel


DEFAULT_CONFIG = {
    "arch": "transformer",
    "n_embed": 128,
    "n_head": 4,
    "layer_count": 6,
    "batch_size": 16,
    "block_size": 48,
    "learning_rate": 1e-4,
    "dropout": 0.2,
    "weight_decay": 0.01,
    "seed": 1337,
    # Tokens = optimizer_steps * batch_size * block_size.
    # Reference budget: 10,000 * 16 * 48 = 7,680,000 predicted tokens.
    "training_tokens": 10_000 * 16 * 48,
    "eval_every_tokens": 200 * 16 * 48,
    "eval_tokens_per_split": 16 * 16 * 48,
    "memory_log_interval": 10,
    "train_fraction": 0.9,
    "data_source": "./data/data.txt",
    "device": "auto",
    "output_dir": "./data/training_log",
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def format_duration(seconds):
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def build_model(config, vocab_size):
    """Use the Transformer count as the budget; only resize RNN state_size."""
    arch, width = config["arch"], config["n_embed"]
    if arch not in ("rnn", "transformer"):
        raise ValueError("arch must be 'rnn' or 'transformer'")
    if width % 2 or width % config["n_head"]:
        raise ValueError("n_embed must be even and divisible by n_head")

    # A non-Transformer run only needs the reference shapes/count. The meta
    # device avoids allocating reference tensors or polluting memory peaks.
    with torch.device("meta" if arch == "rnn" else "cpu"):
        reference = LangModel(
            vocab_size,
            config["block_size"],
            n_embed=width,
            head_size=width // config["n_head"],
            n_head=config["n_head"],
            transformer_blocks=config["layer_count"],
            baseline_hidden=width,
            mlp_rank=None,  # Full-rank MLP: width -> width -> width.
            causal=True,
            dropout_rate=config["dropout"],
            dtype=torch.float32,
        )
    target = reference.count_parameters()["total"]
    if arch == "transformer":
        model = reference
    else:
        del reference
        # Shared embedding/MLP widths, depth and dropout equal the Transformer.
        # State size is the sole extra degree of freedom used for matching P.
        source = RNNLangModel(
            1,
            vocab_size,
            vocab_embedding_size=width,
            hidden_size=width,
            layer_count=config["layer_count"],
            dropout=config["dropout"],
        )
        model = source.copy_scaled_to_fixed_p(
            RNNLangModelHyperparam.state_size, target
        )
    actual = model.count_parameters()["total"]
    return model, {
        "reference_parameter_count": target,
        "parameter_count": actual,
        # Integer state sizes can land on either side of the closest target.
        "parameter_count_difference": actual - target,
        "parameter_count_error_percent": 100 * abs(actual - target) / target,
        "vocab_embedding_size": width,
        "hidden_size": width,
        "state_size": model.state_size if arch == "rnn" else None,
        "head_size": width // config["n_head"],
    }


def sample_batch(dataset, split, config, generator, device, valid_tokens):
    # Use a dedicated CPU generator: model initialization, dropout and evaluation
    # must not change the sampled training windows in an architecture pair.
    data = dataset.train_data if split == "train" else dataset.val_data
    block_size = config["block_size"]
    offsets = torch.randint(
        len(data) - block_size, (config["batch_size"],), generator=generator
    )
    x = torch.stack([data[i : i + block_size] for i in offsets])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in offsets])
    # Both models' cross-entropy losses ignore -100. A partial final batch
    # therefore consumes exactly the remaining target tokens, without padding
    # the budget upward or changing the configured batch/block shape.
    y.view(-1)[valid_tokens:] = -100
    return x.to(device), y.to(device)


def model_loss(model, x, y):
    if isinstance(model, RNNLangModel):
        # Independent random windows: reset state every batch in train AND eval.
        state = model.newstate(x.size(0), device=x.device)
        _, _, loss = model(x, state, y)
    else:
        _, loss = model(x, y)
    return loss


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def estimate_loss(model, dataset, config, device):
    """Fixed token budgets per split; repeat identical evaluation samples."""
    was_training = model.training
    model.eval()
    synchronize(device)
    evaluation_start = time.perf_counter()
    metrics = {}
    try:
        for index, split in enumerate(("train", "val")):
            generator = torch.Generator().manual_seed(config["seed"] + 1 + index)
            remaining = config["eval_tokens_per_split"]
            weighted_loss = forward_seconds = 0.0
            batches = 0
            while remaining:
                tokens = min(remaining, config["batch_size"] * config["block_size"])
                x, y = sample_batch(dataset, split, config, generator, device, tokens)
                # Exclude batch construction/transfers; include RNN zero-state
                # allocation and the forward/loss call. Synchronize GPU kernels
                # so latency measures completion, not just kernel submission.
                synchronize(device)
                forward_start = time.perf_counter()
                loss = model_loss(model, x, y)
                synchronize(device)
                forward_seconds += time.perf_counter() - forward_start
                weighted_loss += loss.item() * tokens
                remaining -= tokens
                batches += 1
            metrics[f"{split}_loss"] = weighted_loss / config["eval_tokens_per_split"]
            metrics[f"eval_latency/{split}_forward_ms_per_batch"] = (
                1000 * forward_seconds / batches
            )
            metrics[f"eval_latency/{split}_forward_us_per_token"] = (
                1e6 * forward_seconds / config["eval_tokens_per_split"]
            )
            metrics[f"eval_latency/{split}_batches"] = batches
    finally:
        model.train(was_training)
    metrics["eval_latency/total_seconds"] = time.perf_counter() - evaluation_start
    return metrics


def memory_usage(device):
    """Process RSS and PyTorch allocator memory, in MiB (also works on ROCm)."""
    mib = 1024**2
    # ru_maxrss is a process-lifetime high-water mark, not a sampled maximum.
    # It is bytes on macOS and KiB on Linux. Each sweep run is a fresh process.
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metrics = {
        "memory/cpu_peak_rss_mib": peak_rss / (mib if sys.platform == "darwin" else 1024)
    }
    statm = Path("/proc/self/statm")
    if statm.exists():
        resident_pages = int(statm.read_text().split()[1])
        metrics["memory/cpu_rss_mib"] = (
            resident_pages * os.sysconf("SC_PAGE_SIZE") / mib
        )
    if device.type == "cuda":
        metrics.update({
            "memory/gpu_allocated_mib": torch.cuda.memory_allocated(device) / mib,
            "memory/gpu_reserved_mib": torch.cuda.memory_reserved(device) / mib,
            "memory/gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / mib,
            "memory/gpu_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / mib,
        })
    return metrics


def train(run):
    config = dict(run.config)
    for key in (
        "n_embed", "n_head", "layer_count", "batch_size", "block_size",
        "training_tokens", "eval_every_tokens", "eval_tokens_per_split",
        "memory_log_interval",
    ):
        if (
            isinstance(config[key], bool)
            or not isinstance(config[key], int)
            or config[key] < 1
        ):
            raise ValueError(f"{key} must be a positive integer")
    if not 0 <= config["dropout"] < 1 or not 0 < config["train_fraction"] < 1:
        raise ValueError("dropout must be in [0, 1) and train_fraction in (0, 1)")
    if config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This experiment supports CPU and CUDA/ROCm devices")
    set_seed(config["seed"])
    # Keep the corpus on CPU for both architectures; only batches go to GPU.
    dataset = Dataset(config["data_source"], train=config["train_fraction"], device="cpu")
    if min(len(dataset.train_data), len(dataset.val_data)) <= config["block_size"]:
        raise ValueError("Both dataset splits must contain more than block_size tokens")

    tokens_per_step = config["batch_size"] * config["block_size"]
    # PREDICTED TOKENS = steps * batch_size * block_size.
    # Inverse: steps = ceil(training_tokens / (batch_size * block_size)).
    # These are optimizer steps, not corpus epochs. Effective corpus epochs
    # = training_tokens / len(train_data); inverse tokens = epochs * len(train_data).
    # Mask the final batch's extra targets if the inverse is not an integer.
    step_count = (config["training_tokens"] + tokens_per_step - 1) // tokens_per_step
    model, metadata = build_model(config, dataset.vocab_size)
    metadata.update({
        "resolved_device": str(device),
        "dtype": "torch.float32",
        "optimizer": "AdamW",
        "vocab_size": dataset.vocab_size,
        "step_count": step_count,
        "tokens_per_step": tokens_per_step,
        "effective_corpus_epochs": config["training_tokens"] / len(dataset.train_data),
    })
    run.config.update(metadata)
    summary = dict(metadata)

    def update_summary(values):
        summary.update(values)
        run.summary.update(values)

    update_summary(metadata)
    run.define_metric("*", step_metric="tokens_seen")
    run.define_metric("memory/*", step_metric="elapsed_seconds", summary="max")
    run.define_metric("eval_latency/*", step_metric="tokens_seen", summary="mean,max,last")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_directory = Path(config["output_dir"]) / (
        f"{timestamp}_{run.id}_{config['arch']}"
        f"_width-{config['n_embed']}_seed-{config['seed']}"
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    (run_directory / "config.json").write_text(json.dumps(dict(run.config), indent=2) + "\n")
    print(
        f"{config['arch']} width={config['n_embed']} layers={config['layer_count']} "
        f"parameters={metadata['parameter_count']:,} target={metadata['reference_parameter_count']:,} "
        f"steps={step_count:,} tokens={config['training_tokens']:,} on {device}"
    )
    print(f"Saving metrics to {run_directory}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        # Include model placement, optimizer, training AND evaluation in peaks.
        torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    generator = torch.Generator().manual_seed(config["seed"])
    best_train_loss = best_val_loss = float("inf")
    best_step = best_tokens = tokens_seen = 0
    completed_steps = 0
    training_seconds = evaluation_seconds = 0.0
    next_eval_tokens = config["eval_every_tokens"]
    synchronize(device)
    loop_start = time.perf_counter()
    metrics_path = run_directory / "metrics.jsonl"

    def log(metrics):
        record = {
            "tokens_seen": tokens_seen,
            "elapsed_seconds": time.perf_counter() - loop_start,
            **metrics,
            **memory_usage(device),
        }
        run.log(record)
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        # Explicit, discoverable maxima as well as W&B's memory/* max summaries.
        peaks = {}
        for name in ("cpu_peak_rss", "gpu_peak_allocated", "gpu_peak_reserved"):
            key = f"memory/{name}_mib"
            if key in record:
                peaks[f"max_{name}_mib"] = record[key]
        update_summary(peaks)

    log({"training_step": 0})
    model.train()
    interval_loss_sum = torch.zeros((), device=device)
    interval_steps = interval_tokens = 0
    interval_start = time.perf_counter()
    try:
        for step in range(1, step_count + 1):
            batch_tokens = min(tokens_per_step, config["training_tokens"] - tokens_seen)
            x, y = sample_batch(dataset, "train", config, generator, device, batch_tokens)
            optimizer.zero_grad(set_to_none=True)
            loss = model_loss(model, x, y)
            loss.backward()
            optimizer.step()
            interval_loss_sum.add_(loss.detach() * batch_tokens)
            interval_steps += 1
            interval_tokens += batch_tokens
            tokens_seen += batch_tokens
            completed_steps = step
            should_evaluate = tokens_seen >= next_eval_tokens or step == step_count
            if step % config["memory_log_interval"] == 0 or should_evaluate:
                synchronize(device)
                interval_seconds = time.perf_counter() - interval_start
                training_seconds += interval_seconds
                metrics = {
                    "training_step": step,
                    "batch_train_loss": loss.item(),
                    "mean_training_loss": interval_loss_sum.item() / interval_tokens,
                    "milliseconds_per_step": 1000 * interval_seconds / interval_steps,
                    "training_tokens_per_second": interval_tokens / interval_seconds,
                    "training_seconds": training_seconds,
                }
                if should_evaluate:
                    optimizer.zero_grad(set_to_none=True)
                    losses = estimate_loss(model, dataset, config, device)
                    if not all(
                        math.isfinite(losses[key]) for key in ("train_loss", "val_loss")
                    ):
                        raise RuntimeError("Non-finite evaluation loss")
                    metrics.update(losses)
                    evaluation_seconds += losses["eval_latency/total_seconds"]
                    best_train_loss = min(best_train_loss, losses["train_loss"])
                    if losses["val_loss"] < best_val_loss:
                        best_val_loss = losses["val_loss"]
                        best_step, best_tokens = step, tokens_seen
                    update_summary({
                        "best_train_loss": best_train_loss,
                        "best_val_loss": best_val_loss,
                        "best_step": best_step,
                        "best_tokens_seen": best_tokens,
                        "last_train_loss": losses["train_loss"],
                        "last_val_loss": losses["val_loss"],
                        "final_train_loss": losses["train_loss"],
                        "final_val_loss": losses["val_loss"],
                    })
                    metrics.update(best_train_loss=best_train_loss, best_val_loss=best_val_loss)
                    next_eval_tokens = (
                        tokens_seen // config["eval_every_tokens"] + 1
                    ) * config["eval_every_tokens"]
                    elapsed = time.perf_counter() - loop_start
                    eta = elapsed / tokens_seen * (config["training_tokens"] - tokens_seen)
                    metrics["eta_seconds"] = eta
                    print(
                        f"step {step}/{step_count}, tokens {tokens_seen:,}: "
                        f"train {losses['train_loss']:.4f}, val {losses['val_loss']:.4f}, "
                        f"eval {losses['eval_latency/total_seconds']:.3f}s, "
                        f"val forward {losses['eval_latency/val_forward_ms_per_batch']:.2f}ms/batch, "
                        f"ETA {format_duration(eta)}"
                    )
                metrics["wall_tokens_per_second"] = tokens_seen / (
                    time.perf_counter() - loop_start
                )
                log(metrics)
                interval_loss_sum.zero_()
                interval_steps = interval_tokens = 0
                # Exclude evaluation and log overhead from training throughput.
                interval_start = time.perf_counter()
    finally:
        log({"training_step": completed_steps, "training_seconds": training_seconds,
             "evaluation_seconds": evaluation_seconds})
        update_summary({
            "tokens_seen": tokens_seen,
            "completed_steps": completed_steps,
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "elapsed_seconds": time.perf_counter() - loop_start,
        })
        (run_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    # W&B's sweep agent supplies YAML parameter values through run.config.
    # Defaults also allow a standalone/offline run with python -m ml_exp.
    with wandb.init(
        project="ml-exp", group="transformer-vs-rnn", job_type="train",
        config=DEFAULT_CONFIG,
    ) as run:
        train(run)


if __name__ == "__main__":
    main()
