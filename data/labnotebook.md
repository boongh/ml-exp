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

|  Step | Train loss | Validation loss | Time per step (ms) | Elapsed time |
| ----: | ---------: | --------------: | -----------------: | -----------: |
|     0 |     5.4718 |          5.4825 |             382.99 |           5s |
| 1,000 |     2.1318 |          2.1494 |              60.88 |       1m 49s |
| 2,000 |     1.9474 |          1.9770 |              54.93 |       3m 29s |
| 3,000 |     1.8593 |          1.8905 |              57.07 |       5m 06s |
| 4,000 |     1.7784 |          1.8119 |              61.95 |       6m 44s |
| 5,000 |     1.7404 |          1.7636 |              55.84 |       8m 22s |
| 6,000 |     1.6942 |          1.7353 |              60.05 |      10m 02s |
| 7,000 |     1.6462 |          1.6887 |              57.22 |      11m 40s |
| 8,000 |     1.6218 |          1.6734 |              59.96 |      13m 20s |
| 9,000 |     1.6098 |          1.6531 |              60.92 |      15m 04s |
| 9,999 |     1.6001 |          1.6481 |              57.45 |      16m 49s |

### Result summary

| Result                     | Train loss | Validation loss | Step / value               |
| -------------------------- | ---------: | --------------: | -------------------------- |
| Initial loss               |     5.4718 |          5.4825 | Step 0                     |
| Best validation loss       |          — |          1.6368 | Step 9,600                 |
| Final loss                 |     1.6001 |          1.6481 | Step 9,999                 |
| Final measured speed       |          — |               — | 57.45 ms/step              |
| Training-loop elapsed time |          — |               — | 1,008.54 seconds (16m 49s) |


## EXP1. Low-rank MLP
###### Commit `5b9440d96d9c17b5f2b44b03e514878aa53d03bd`

Our first series of experiments will be one concerning the idea of using 2 linear projection layer of dimension (N, k) and (k, M) to achieve the work of an single linear projection layer of dimension (N, M).

### Methodology

We first substitude `MLP_Full` with a `MLP_Lowrank` module with with `rank = n_embed // rank_divisor`, which internally does a double linear projection to achieve equivalent input-output dimension as `MLP_Full`. Each test will progressively increase the `rank_divisor` until `rank_divisor` is equal to the `hidden_layer_size` of `TransformerBlocks` (In this case, 1 to 256).

### Tests
Experiment for `rank_divisor` 1 to 128 was done in commit `63be74b9624e9c340174354d10c9f53439d34353` and for `rank_divisor` 256 

![graph](./media/wandb_rank_divisor_loss.svg)