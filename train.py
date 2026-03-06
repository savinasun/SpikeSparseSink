import os
import math
import random
import datetime

# os.environ["NCCL_IB_GID_INDEX"] = "0"

import torch
import torch.distributed.fsdp
import torchdata.stateful_dataloader

from helper.cfg import get_cfg
from helper.dataset import DistributedDataset
from models import get_model
from helper.checkpointer import Checkpointer
from helper.engine import train


def main(cfg):
    ################################################################################
    # set up distributed training and random seed
    ################################################################################
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group("nccl", device_id=local_rank)
    print(f"World Size: {world_size}, Rank: {rank}, Local Rank: {local_rank}")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    if rank == 0: print(cfg)

    ################################################################################
    # set up model and optimizer
    ################################################################################
    model, Block = get_model(cfg)
    model.reset_parameters()

    mp_policy = torch.distributed.fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, output_dtype=torch.bfloat16)
    for module in model.modules():
        if isinstance(module, Block):
            torch.distributed.fsdp.fully_shard(module, mp_policy=mp_policy)
    torch.distributed.fsdp.fully_shard(model, mp_policy=mp_policy)

    if rank == 0: print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)

    ################################################################################
    # set up dataloader
    ################################################################################
    dataset = DistributedDataset(cfg.dataset_dir, world_size, rank, cfg.batch_size, cfg.sequence_length)
    dataloader = torchdata.stateful_dataloader.StatefulDataLoader(dataset, batch_size=None, num_workers=0)

    ################################################################################
    # set up checkpointer
    ################################################################################
    checkpointer = Checkpointer(os.path.join(cfg.checkpoint_dir, cfg.experiment_name))
    start_step = checkpointer.load(model, optimizer, dataloader)
    if rank == 0: print(f"Starting from step {start_step}.")

    ################################################################################
    # optional: compile the model
    ################################################################################
    if cfg.compile:
        model = torch.compile(model)
        if rank == 0: print("torch.compile applied to the model.")

    ################################################################################
    # set up lr scheduler
    ################################################################################
    schedule = lambda x: min(
        1 - (1 - min(x, cfg.num_warmup_steps) / cfg.num_warmup_steps) ** 2,
        cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos((x - cfg.num_warmup_steps) / (cfg.num_steps - cfg.num_warmup_steps) * math.pi)),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: schedule(x + start_step))

    ################################################################################
    # training loop
    ################################################################################
    train(
        cfg,
        start_step,
        device,
        dataloader,
        model,
        optimizer,
        scheduler,
        checkpointer,
    )

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main(get_cfg())
