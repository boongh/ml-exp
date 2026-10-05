<<<<<<< HEAD
# Machine Learning Experiments
This repo is merely a public log of my beginner journey into experimenting with neural network machine learning.

##### Lab notebook can be found [here](data/labnotebook.md), along training logs, data, and diagrams
=======
Run a small end-to-end sanity check (no training corpus or W&B required):

```sh
python -m ml_exp.sanity_check
```

Use `--device cuda` to check the GPU pipeline. The script checks tokenization,
batching, forward loss, gradients, an optimizer update, validation, and generation.
>>>>>>> c10e807 (sanity test script and small fixes on langmodel before experiment)
