# [Anatomy of Massive Activations and Attention Sinks](https://arxiv.org/abs/2603.05498)

Official PyTorch implementation of the ablation experiments in Section 4 of the following paper:

[The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks](https://arxiv.org/abs/2603.05498).
[Shangwen Sun](https://savinasun.github.io/), [Alfredo Canziani](https://atcold.github.io/), [Yann LeCun](http://yann.lecun.com) and [Jiachen Zhu](https://jiachenzhu.github.io)
New York University

---

## Installation

```bash
pip install torch torchvision torchdata pyarrow wandb
```

You will also need a [Weights &amp; Biases](https://wandb.ai/) account for experiment tracking:

```bash
wandb login
```

## Configuration

Training is configured through YAML files passed as command-line arguments. Multiple files are merged in order, with later files overriding earlier ones. [configs/base.yaml](configs/base.yaml) provides shared defaults, while per-experiment configs (e.g., [configs/train_0.yaml](configs/train_0.yaml)) specify experiment-specific settings.

The following fields must be set to match your environment:

| Field               | Description                                                              |
| ------------------- | ------------------------------------------------------------------------ |
| `experiment_name` | Unique name for your experiment (used for checkpointing and W&B logging) |
| `checkpoint_dir`  | Directory for saving checkpoints                                         |
| `dataset_dir`     | Path to the tokenized dataset directory (Arrow format)                   |
| `batch_size`      | Per-GPU batch size                                                       |
| `tracker_project` | Weights & Biases project name                                            |
| `tracker_dir`     | Local directory for W&B run files                                        |

## Training

This codebase uses PyTorch distributed training via `torchrun`. Pass the base config followed by your experiment config:

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train.py configs/base.yaml configs/train_0.yaml

# Multi-node (run on each node)
torchrun --nnodes=<NUM_NODES> --nproc_per_node=<GPUS_PER_NODE> \
    --rdzv_backend=c10d --rdzv_endpoint=<MASTER_ADDR>:<PORT> \
    train.py configs/base.yaml configs/train_0.yaml
```

Training automatically resumes from the latest checkpoint if one exists in the checkpoint directory.
