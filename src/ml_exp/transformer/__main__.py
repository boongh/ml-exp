import random
import time
from datetime import datetime
from pathlib import Path

import torch

import wandb
from ml_exp.data import Dataset

from .transformer import LangModel

# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# device = "cpu"

eval_prompts = ""

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

# Data loading
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
                X, Y = dataset.get_batch(
                    "train", block_size=block_size, batch_size=batch_size
                )
            else:
                X, Y = dataset.get_batch(
                    "val", block_size=block_size, batch_size=batch_size
                )

            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out
