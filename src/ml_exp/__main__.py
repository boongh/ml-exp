from quopri import encode
from datetime import datetime
from pathlib import Path
import re
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

#Construct a geometric progression of rank divisor from 1 to n_embed with ratio 2
rank_divisors = [n_embed // (2 ** i) for i in range(int(n_embed.bit_length()) - 1, -1, -1)]

hidden_layer_size = 256
n_head = 4
causal = True
dtype = torch.float32
train_fraction = 0.9
transformer_blocks = 6
step_count = 10_000
lrate = 6e-4
loss_est_count = 100
checkin_iteration = 100
random_seed = 1337
pretrain_generation_tokens = 1000
posttrain_generation_tokens = 10000
dropout_rate = 0.2

eval_prompts = [
    "Once upon a time ",
    "Klein Moretti was ",
    "Melissa was a ",
]

while True:
    run_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", input("Tag this training run: ").strip()).strip("._-")
    if run_tag:
        break
    print("Please enter a non-empty run tag.")

#Reproducability
torch.manual_seed(random_seed)

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

for divisor in rank_divisors:
    print(f"Rank divisor: {divisor}, resulting rank: {n_embed // divisor}")
    rank_division = divisor

    #Model definition
    model = LangModel(
        dataset.vocab_size,
        block_size,
        head_size,
        n_embed=n_embed,
        transformer_blocks=transformer_blocks,
        n_head=n_head,
        hidden_layer_size=hidden_layer_size,
        rank_division=rank_division,
        dtype=dtype,
        causal=causal,
        dropout_rate=dropout_rate,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lrate
    )

    # Create a separate artifact directory for every training run.
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_directory = Path("./data/training_log") / f"{run_timestamp}_{run_tag}"
    run_directory.mkdir(parents=True, exist_ok=False)

    hyperparameters = {
        "run_timestamp": run_timestamp,
        "run_tag": run_tag,
        "data_source": data_source,
        "device": device,
        "dtype": str(dtype),
        "random_seed": random_seed,
        "vocab_size": dataset.vocab_size,
        "block_size": block_size,
        "batch_size": batch_size,
        "head_size": head_size,
        "n_embed": n_embed,
        "hidden_layer_size": hidden_layer_size,
        "n_head": n_head,
        "rank_division": rank_division,
        "causal": causal,
        "train_fraction": train_fraction,
        "step_count": step_count,
        "learning_rate": lrate,
        "optimizer": type(optimizer).__name__,
        "loss_est_count": loss_est_count,
        "checkpoint_interval": checkin_iteration,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "pretrain_generation_tokens": pretrain_generation_tokens,
        "posttrain_generation_tokens": posttrain_generation_tokens,
        "eval_prompts": eval_prompts,
        "transformer_blocks": transformer_blocks,
        "dropout_rate": dropout_rate,
    }

    with (run_directory / f"hyper_param_{run_timestamp}.txt").open("w", encoding="utf-8") as file:
        for name, value in hyperparameters.items():
            file.write(f"{name}={value}\n")


    wandb_run = wandb.init(
        project="ml-exp",
        name=run_tag + f"laptop_{run_timestamp} rank_div_{rank_division}",
        config=hyperparameters,
    )

    #Pretrain
    model.eval()
    print("Pretrain generation")
    pretrain_generations = generate_eval_outputs(pretrain_generation_tokens)
    for output in pretrain_generations:
        print(f"Prompt: {output['prompt']}\n{output['generation']}")

    wandb.log({
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

    def format_duration(seconds):
        seconds = max(0, round(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours:
            return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
        if minutes:
            return f"{minutes:d}m {seconds:02d}s"
        return f"{seconds:d}s"

    # checkpoint = torch.load("model_checkpoint.pt", map_location=device)
    # torch.save(model, "model_checkpoint.pt")

    print("Training loop")
    print(f"Saving run artifacts to {run_directory}")
    loop_start = time.perf_counter()
    interval_start = loop_start
    previous_completed_steps = 0

    for step in range(step_count):

        x, y = dataset.get_batch("train", block_size=block_size, batch_size=batch_size)
        _ , loss = model(x, y)

        # lrate *= 0.999995
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lrate

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % checkin_iteration == 0 or step == step_count - 1:
            if device == "cuda":
                torch.cuda.synchronize()

            training_end = time.perf_counter()
            completed_steps = step + 1
            interval_steps = completed_steps - previous_completed_steps
            milliseconds_per_step = (
                (training_end - interval_start) * 1000 / interval_steps
            )

            losses = estimate_loss()

            # Include validation/checkpoint overhead in the ETA so it estimates
            # actual wall-clock time until the run finishes.
            checkpoint_end = time.perf_counter()
            elapsed = checkpoint_end - loop_start
            remaining_steps = step_count - completed_steps
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

            wandb.log(
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

    wandb.log({
        "posttrain_generations": wandb.Table(
            columns=["prompt", "generation"],
            data=[
                [output["prompt"], output["generation"]]
                for output in posttrain_generations
            ],
        ),
    })

    wandb_run.finish()
