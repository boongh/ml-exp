"""Run with python -m ml_exp.sanity_check [--device cuda]."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from .data import Dataset
from .langmodel import LangModel


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")
    args = parser.parse_args()
    torch.manual_seed(42)
    block_size, batch_size = 16, 2

    with TemporaryDirectory() as directory:
        source = Path(directory) / "sample.txt"
        source.write_text("A small language model sanity check.\n" * 100, encoding="utf-8")
        dataset = Dataset(source, device=args.device)

    check(dataset.decode(dataset.encode(dataset.text)) == dataset.text, "Token round trip failed")
    model = LangModel(
        dataset.vocab_size,
        block_size,
        head_size=8,
        n_embed=32,
        transformer_blocks=2,
        n_head=4,
        mlp_rank=4,
    ).to(args.device)
    for block in model.transformer_blocks[::2]:
        check(block.attention.n_head == 4, "Incorrect attention head count")
        # The bias-inclusive budget gives floor(1816 / 9) = 201 hidden units.
        linear_shapes = [
            (layer.in_features, layer.out_features)
            for layer in block.ffwdn.net
            if isinstance(layer, torch.nn.Linear)
        ]
        check(
            linear_shapes == [(32, 4), (4, 201), (201, 4), (4, 32)],
            "Incorrect fixed-budget MLP dimensions",
        )
        check(
            sum(parameter.numel() for parameter in block.ffwdn.parameters()) == 2105,
            "Incorrect MLP parameter count (baseline 2112, rounding gap 7)",
        )

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = dataset.get_batch("train", block_size=block_size, batch_size=batch_size)
    check(x.shape == y.shape == (batch_size, block_size), "Incorrect batch shape")
    check(torch.equal(x[:, 1:], y[:, :-1]), "Targets are not shifted by one token")
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(x, y)
    check(logits.shape == (batch_size, block_size, dataset.vocab_size), "Incorrect logits shape")
    check(torch.isfinite(logits).all().item(), "Non-finite logits")
    check(loss is not None and loss.ndim == 0 and torch.isfinite(loss).item(), "Invalid loss")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            check(parameter.grad is not None, f"Missing gradient: {name}")
            check(torch.isfinite(parameter.grad).all().item(), f"Non-finite gradient: {name}")
    before = model.lm_head.weight.detach().clone()
    optimizer.step()
    check(not torch.equal(before, model.lm_head.weight), "Optimizer did not update weights")
    for name, parameter in model.named_parameters():
        check(torch.isfinite(parameter).all().item(), f"Non-finite parameter after update: {name}")

    model.eval()
    with torch.no_grad():
        x_val, y_val = dataset.get_batch("val", block_size=block_size, batch_size=batch_size)
        _, val_loss = model(x_val, y_val)
        check(torch.isfinite(val_loss).item(), "Non-finite validation loss")
        generated = model.generation(x_val[:1], max_new_tokens=4)
        check(generated.shape == (1, block_size + 4), "Incorrect generation shape")
        dataset.decode(generated[0].tolist())

    print(
        f"PASS on {args.device}: forward, backward, optimizer, validation, generation "
        f"(train loss={loss.item():.4f}, val loss={val_loss.item():.4f})"
    )


if __name__ == "__main__":
    main()
