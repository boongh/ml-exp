from quopri import encode
from datetime import datetime
from pathlib import Path
import random
import time

import torch
import wandb
from .langmodel import LangModel
from .data import Dataset

#Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
#device = "cpu"


block_size = 128
batch_size = 16
head_size = 32 // 4
n_embed = 32 * 4
n_head = 4
causal = True
dtype = torch.float32
train_fraction = 0.9
transformer_blocks = 6
step_count = 10_000
loss_est_count = 100
checkin_iteration = 100
pretrain_generation_tokens = 50
posttrain_generation_tokens = 500
dropout_rate = 0.2

eval_prompts = [
    "Once upon a time ",
    "Klein Moretti was ",
    "Melissa was a ",
]

#Data loading
data_source = "./data/data.txt"
dataset = Dataset(data_source, train=train_fraction, device=device)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.empty(loss_est_count, device=device)
        for k in range(loss_est_count):
            if split == "train":
                X, Y = dataset.get_batch("train", block_size=block_size, batch_size=batch_size)
            else:
                X, Y = dataset.get_batch("val", block_size=block_size, batch_size=batch_size)

            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out

def generate_eval_outputs(max_new_tokens):
    outputs = []
    for prompt in eval_prompts:
        prompt_tokens = torch.tensor(
            [dataset.encode(prompt)],
            dtype=torch.long,
            device=device,
        )
        generated = dataset.decode(
            model.generation(
                prompt_tokens,
                max_new_tokens=max_new_tokens,
            )[0].tolist()
        )
        outputs.append({"prompt": prompt, "generation": generated})
    return outputs

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    
hyperparameters = {
    "data_source": data_source,
    "device": device,
    "dtype": str(dtype),
    "vocab_size": dataset.vocab_size,
    "block_size": block_size,
    "batch_size": batch_size,
    "head_size": head_size,
    "n_embed": n_embed,
    "n_head": n_head,
    "causal": causal,
    "train_fraction": train_fraction,
    "step_count": step_count,
    "loss_est_count": loss_est_count,
    "checkpoint_interval": checkin_iteration,
    "pretrain_generation_tokens": pretrain_generation_tokens,
    "posttrain_generation_tokens": posttrain_generation_tokens,
    "eval_prompts": eval_prompts,
    "transformer_blocks": transformer_blocks,
    "dropout_rate": dropout_rate,
}

def format_duration(seconds):
        seconds = max(0, round(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours:
            return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
        if minutes:
            return f"{minutes:d}m {seconds:02d}s"
        return f"{seconds:d}s"


with wandb.init(
    project="ml-exp",
    group="mlp-fixed-param-count-sweep",
    config=hyperparameters,
) as run:
    config = run.config

        
    # Create a separate artifact directory for every training run.
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_tag = "mlp_fixed_param_count_sweep"
    run_directory = (
        Path("./data/training_log")
        / f"{run_timestamp}_{run.id}_mlp_rank-{config.mlp_rank}_seed-{config.seed}"
    )
    run_directory.mkdir(parents=True, exist_ok=False)

    
    set_seed(config.seed)
    print(f"MLP rank {config.mlp_rank} seed {config.seed} on device {device}")

    #Model definition
    model = LangModel(
        dataset.vocab_size,
        block_size,
        head_size,
        n_embed=n_embed,
        transformer_blocks=transformer_blocks,
        n_head=n_head,
        mlp_rank=config.mlp_rank,
        dtype=dtype,
        causal=causal,
        dropout_rate=dropout_rate,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate
    )

    config["hidden_layer_size"] = hidden_layer = int((2 * n_embed ** 2 + n_embed - 2 * config.mlp_rank * n_embed - 2 * config.mlp_rank) / (2 * config.mlp_rank + 1))
    config["optimizer"] = optimizer.__class__.__name__
    config["run_timestamp"] = run_timestamp
    config["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())

    with (run_directory / f"hyper_param_{run_timestamp}.txt").open("w", encoding="utf-8") as file:
        for name, value in dict(config).items():
            file.write(f"{name}={value}\n")

        
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_step = 0
    final_val_loss = None
    final_train_loss = None

    #Pretrain
    model.eval()
    print("Pretrain generation")
    pretrain_generations = generate_eval_outputs(pretrain_generation_tokens)
    for output in pretrain_generations:
        print(f"Prompt: {output['prompt']}\n{output['generation']}")


    run.log({
        "pretrain_generations": wandb.Table(
            columns=["prompt", "generation"],
            data=[
                [output["prompt"], output["generation"]]
                for output in pretrain_generations
            ],
        ),
    })

    #training loop
    model.train()

    # checkpoint = torch.load("model_checkpoint.pt", map_location=device)
    # torch.save(model, "model_checkpoint.pt")

    print("Training loop")
    print(f"Saving run artifacts to {run_directory}")
    loop_start = time.perf_counter()
    interval_start = loop_start
    previous_completed_steps = 0

    for step in range(config.step_count):

        x, y = dataset.get_batch("train", block_size=block_size, batch_size=batch_size)
        _ , loss = model(x, y)

        # lrate *= 0.999995
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lrate

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % checkin_iteration == 0 or step == config.step_count - 1:
            if device == "cuda":
                torch.cuda.synchronize()

            training_end = time.perf_counter()
            completed_steps = step + 1
            interval_steps = completed_steps - previous_completed_steps
            milliseconds_per_step = (
                (training_end - interval_start) * 1000 / interval_steps
            )

            losses = estimate_loss()

            if losses['train'] < best_train_loss:
                best_train_loss = losses['train']

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                best_step = step

            final_train_loss = losses['train']
            final_val_loss = losses['val']

            # Include validation/checkpoint overhead in the ETA so it estimates
            # actual wall-clock time until the run finishes.
            checkpoint_end = time.perf_counter()
            elapsed = checkpoint_end - loop_start
            remaining_steps = config.step_count - completed_steps
            eta = elapsed / completed_steps * remaining_steps
            eta_text = (
                format_duration(eta)
                if completed_steps > 1
                else "estimating..."
            )

            print(
                f"step {step}: train_loss {losses['train']:.4f}, "
                f"val_loss {losses['val']:.4f}, "
                f"{milliseconds_per_step:.2f} ms/step, "
                f"elapsed {format_duration(elapsed)}, ETA {eta_text}"
            )

            run.log(
                {
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "milliseconds_per_step": milliseconds_per_step,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                },
                step=step,
            )

            interval_start = checkpoint_end
            previous_completed_steps = completed_steps

    #Post train
    model.eval()
    print("Posttrain generation")
    posttrain_generations = generate_eval_outputs(posttrain_generation_tokens)
    for output in posttrain_generations:
        print(f"Prompt: {output['prompt']}\n{output['generation']}")

    run.log({
        "posttrain_generations": wandb.Table(
            columns=["prompt", "generation"],
            data=[
                [output["prompt"], output["generation"]]
                for output in posttrain_generations
            ],
        ),
    })

    run.summary["best_val_loss"] = best_val_loss
    run.summary["best_step"] = best_step
    run.summary["final_val_loss"] = final_val_loss
    run.summary["final_train_loss"] = final_train_loss