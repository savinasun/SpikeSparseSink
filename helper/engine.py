import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb


def train(cfg, start_step, device, dataloader, model, optimizer, scheduler, checkpointer):
    if torch.distributed.get_rank() == 0:
        wandb.init(
            project=cfg.tracker_project,
            dir=cfg.tracker_dir,
            name=cfg.experiment_name,
            config=cfg.to_dict(),
            save_code=True,
        )

    gc.collect()
    model.train()
    start_time = time.time()
    ddp_stats = torch.zeros(3).to(device)
    for step, (input, label) in enumerate(dataloader, start=start_step + 1):
        if step > cfg.num_steps: break
        
        input, label = input.to(device), label.to(device)
        output = model(input) # (B, T, V)
        
        if cfg.context_length is not None:
            output = output[:, cfg.context_length[0]:cfg.context_length[1]]
            label = label[:, cfg.context_length[0]:cfg.context_length[1]]
        loss = F.cross_entropy(output.reshape(-1, output.shape[-1]), label.reshape(-1)) # (B*T, V) vs (B*T,)

        optimizer.zero_grad()
        loss.backward()
        
        ddp_stats[0] += 1
        ddp_stats[1] += loss.item()
        ddp_stats[2] += nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm).item()
        
        optimizer.step()
        scheduler.step()

        if step % cfg.gc_interval == 0:
            gc.collect()

        if step % cfg.report_interval == 0:
            torch.distributed.all_reduce(ddp_stats, op=torch.distributed.ReduceOp.SUM)
            average_loss = ddp_stats[1] / ddp_stats[0]
            average_grad_norm = ddp_stats[2] / ddp_stats[0]
            time_per_step = (time.time() - start_time) / cfg.report_interval

            if torch.distributed.get_rank() == 0:
                print(f"Step {step:05d}/{cfg.num_steps}: LR={scheduler.get_last_lr()[0]:.6f}, Loss={average_loss.item():.4f}, Grad Norm={average_grad_norm.item():.4f}, Time/Step={time_per_step:.4f}s")
                
                wandb.log({
                    "train/loss": average_loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": average_grad_norm.item(),
                    "train/time_per_step": time_per_step,
                }, step=step)

            ddp_stats.zero_()
            start_time = time.time()

        if step % cfg.checkpoint_interval == 0 or step == cfg.num_steps:
            checkpointer.save(
                step,
                model,
                optimizer,
                dataloader,
            )
