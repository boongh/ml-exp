"""Benchmark one W&B sweep configuration on synthetic language-model batches."""

import os
import platform
import resource
import statistics
import time

import torch
import wandb

from ml_exp.rnn.rnn import RNNLangModel, RNNTokenMajorLangModel


DEFAULTS = {
    "model": "rnn", "device": "auto", "seed": 1337,
    "sequence_length": 32, "batch_size": 8, "vocab_size": 256,
    "state_size": 128, "vocab_embedding_size": 128, "hidden_size": 128,
    "layer_count": 2, "dropout": 0.0, "learning_rate": 0.0003,
    "weight_decay": 0.01, "warmup": 5, "iterations": 50,
}
MODELS = {"rnn": RNNLangModel, "token_major": RNNTokenMajorLangModel}


def benchmark(config, log):
    c = dict(config)
    if c["model"] not in MODELS:
        raise ValueError(f"Unknown model {c['model']!r}; choose from {list(MODELS)}")
    if c["iterations"] < 1 or c["warmup"] < 0:
        raise ValueError("iterations must be positive and warmup nonnegative")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if c["device"] == "auto" else c["device"])
    torch.manual_seed(c["seed"])
    model = MODELS[c["model"]](
        c["state_size"], c["vocab_size"], dropout=c["dropout"],
        vocab_embedding_size=c["vocab_embedding_size"], hidden_size=c["hidden_size"],
        layer_count=c["layer_count"],
    ).to(device)
    x = torch.randint(c["vocab_size"], (c["batch_size"], c["sequence_length"]), device=device)
    targets = torch.randint(c["vocab_size"], x.shape, device=device)
    state = model.newstate(c["batch_size"], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
    tokens = x.numel()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def step(training):
        if training:
            optimizer.zero_grad(set_to_none=True)
            _, _, loss = model(x, state, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.inference_mode():
                model(x, state)

    log({"parameter_count": sum(p.numel() for p in model.parameters()),
         "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or platform.machine(),
         "torch_version": torch.__version__, "cpu_threads": torch.get_num_threads(),
         "pid": os.getpid()})
    for training in (False, True):
        mode = "training" if training else "inference"
        model.train(training)
        for _ in range(c["warmup"]):
            step(training)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for _ in range(c["iterations"]):
            sync()
            start = time.perf_counter()
            step(training)
            sync()
            durations.append(time.perf_counter() - start)
        mean = statistics.mean(durations)
        metrics = {
            f"{mode}/seconds_per_batch_mean": mean,
            f"{mode}/seconds_per_batch_median": statistics.median(durations),
            f"{mode}/seconds_per_batch_stdev": statistics.stdev(durations) if len(durations) > 1 else 0.0,
            f"{mode}/seconds_per_token": mean / tokens,
            f"{mode}/tokens_per_second": tokens / mean,
            # Linux reports KiB; this is the process lifetime high-water mark.
            f"{mode}/process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
        }
        if device.type == "cuda":
            metrics[f"{mode}/gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            metrics[f"{mode}/gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        log(metrics)


def main():
    with wandb.init(config=DEFAULTS, project="ml-exp") as run:
        benchmark(run.config, run.log)


if __name__ == "__main__":
    main()
