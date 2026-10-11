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
<<<<<<< HEAD
>>>>>>> c10e807 (sanity test script and small fixes on langmodel before experiment)
=======

For Google Colab, open [`notebooks/ml_exp_colab.ipynb`](notebooks/ml_exp_colab.ipynb).
It contains the current fixed-parameter MLP experiment and sanity check in one
self-contained notebook. Select a GPU runtime, run the cells in order, and upload
`data/data.txt` when prompted. By default it joins the existing W&B sweep
`boonsvj-university-of-manchester-students-union/ml-exp/4oxvbvjf`.
Clear `SWEEP_PATH` to use the editable manual rank/seed grid, where W&B is optional.
The final cell downloads metrics, generated text, and checkpoints.
>>>>>>> 0a65059 (fixed affine layer size, vary hidden width perf test)
