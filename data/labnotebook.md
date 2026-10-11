# Laboratory experiments results

## Preface

### Training data

This project will always use the full complete volumes of "In Search of Lost Time" digital text copy from the Internet Archive. This choice was made due to it being the single longest public domain literature I knew of (7672232 bytes).

## EXP0. Baseline
###### Base commit: `3f8b4fd4590c917bc6d6454532a8943a56b564cd`

###### **Note:** Part of the hyperparameters of this experiment was produced in commit `5b9440d96d9c17b5f2b44b03e514878aa53d03bd` due to a mistake.

The first experimental run to establish a baseline performance for the model. The results and hyperparameters are as below. Changes to the model will be made and recorded throughout the experiment's lifetime.

### Hyperparameters

| Parameter                         |                                                       Value |
| --------------------------------- | ----------------------------------------------------------: |
| Run tag                           |                                                  `Baseline` |
| Run timestamp                     |                                       `2026-09-26_15-56-36` |
| Training split                    |                                                         90% |
| Validation split                  |                                                         10% |
| Random seed                       |                                                        1337 |
| Vocabulary size                   |                                               90 characters |
| Context length (`block_size`)     |                                                  128 tokens |
| Batch size                        |                                                          16 |
| Training steps                    |                                                      10,000 |
| Optimizer                         |                                                       AdamW |
| Learning rate                     |                                                      0.0006 |
| Loss-estimation batches per split |                                                         100 |
| Checkpoint interval               |                                                   100 steps |
| Device                            |                                                        CUDA |
| Data type                         |                                             `torch.float32` |
| Dropout rate                      |                                                         0.2 |
| Pre-training generation length    |                                     1,000 tokens per prompt |
| Post-training generation length   |                                    10,000 tokens per prompt |
| Evaluation prompts                | `Once upon a time `; `Klein Moretti was `; `Melissa was a ` |

### Model architecture

| Component            | Configuration                                                                                   |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| Model type           | Character-level causal language model                                                           |
| Token embedding      | Vocabulary size 90; embedding dimension 128                                                     |
| Positional encoding  | Learned-scale sinusoidal encoding; applied to token embeddings and after each Transformer block |
| Input projection     | Linear, 128 → 256                                                                               |
| Transformer stack    | 6 pre-norm Transformer blocks; model width 256                                                  |
| Self-attention       | 4 heads; head size 8; fused QKV projection; scaled dot-product causal attention                 |
| Feed-forward network | `MLP_Full`; 256 → 256 → 256; GELU activation                                                    |
| Block structure      | Layer normalization, residual connections, and dropout 0.2                                      |
| Language-model head  | Linear, 256 → 90 logits                                                                         |
| Parameter count      | 1,063,130                                                                                       |

### Training results

![Baseline validation loss](./media/wandb_baseline_val_loss.svg)

### Result summary

| Result                     | Train loss | Validation loss | Step / value               |
| -------------------------- | ---------: | --------------: | -------------------------- |
| Initial loss               |     5.4718 |          5.4825 | Step 0                     |
| Best validation loss       |          — |          1.6368 | Step 9,600                 |
| Final loss                 |     1.6001 |          1.6481 | Step 9,999                 |
| Final measured speed       |          — |               — | 57.45 ms/step              |
| Training-loop elapsed time |          — |               — | 1,008.54 seconds (16m 49s) |


## EXP1. Exploring the limit of Low-rank MLP
###### Commit `5b9440d96d9c17b5f2b44b03e514878aa53d03bd`

Our first series of experiments will be one concerning the idea of using 2 linear projection layer of dimension (N, k) and (k, M) to achieve the work of an single linear projection layer of dimension (N, M).

We aim to explore the limit different thresholds of using Low-rank MLP instead of a full rank MLP.

### Test 0 - Preliminary

#### Goal

This first test covers exploring potential problems in using Low-rank MLP by non-rigorously conducting simple experiment comparing the differences in the model results of different rank of Low-rank MLP.

#### Methodology

We substitude `MLP_Full` with a `MLP_Lowrank` module with with `rank = n_embed // rank_divisor`, which internally does a double linear projection to achieve equivalent input-output dimension as `MLP_Full`. Each test will progressively increase the `rank_divisor` until `rank_divisor` is equal to the `hidden_layer_size` of `TransformerBlocks` (In this case, 1 to 256).

Experiment for `rank_divisor` 1 to 128 was done in commit `63be74b9624e9c340174354d10c9f53439d34353` and for `rank_divisor` 256 was done in `55667c29709eb5a612d32dbbc575550ed8a5a443`

`rank divisor` is incremented in a geometric sequence of ratio 2

#### Results


![graph of rank divisor against validation loss](./media/wandb_rank_divisor_loss.svg)*[Figure 5]*

The figure above shows that, as expected increasing the rank divisor substanially degrades the validation loss of the model. Rank divisor of 1, which doubles the parameter count of the `MLP_Lowrank` compares to and `MLP_Full` of the same `n_embed`, does not show any improvement in validation loss compared to the baseline. Rank divisor of 2, which makes the parameter count in the transformer block equal to the baseline, shows slightly higher validation loss compared to the baseline.

Surprisingly, though, increasing the rank divisor from 16 to 32 caused a significantly larger validation loss grap than other increments. Rank divisor 128 and 256 validation loss converges to 1.992.

###### Figure 6
![graph of parameter count against final validation loss](./media/wandb_val_loss_vs_param_count_rank_divisor_exp.svg)

There is a visible sudden change in gradient below 400k parameter (rank divisor 16) which is worth investigating into.

#### Conclusion & next step

From the preliminary tests, the gap in validation loss at divisor 16 and 32, along side with the gap in the parameter, exposes that there might be something interesting going on around there.

### Test 1 - Exploring the Validation Loss Jump

*Relevant commits*

***[1]*** `72265c15ca1996351e17e39afc4843f643594f3c`

***

Further exploration of the validation loss jump was done with W&B sweep in commit **[1]**.

![Graph of best valudation loss against mlp ranks from 6 to 20](./media/bvl.v.rank_6-20.svg)

Using a sample of 5 different seeds, I tested frequently for best validation loss between mlp rank 6 and 20 (and just to make sure up there is a pattern, to 256) and found that the graph does not exhibit any visually distinctive jump up and down in gradient as we might expect to observe from *[Figure 5]*