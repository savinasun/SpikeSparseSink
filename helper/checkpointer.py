import os

import torch
from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict, set_model_state_dict, set_optimizer_state_dict, StateDictOptions


def get_latest_checkpoint(checkpoint_dir):
    return max([os.path.join(checkpoint_dir, checkpoint) for checkpoint in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, checkpoint))], key=os.path.getctime) if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir) else None


class Checkpointer:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def load(self, model, optimizer, dataloader):
        latest_checkpoint = get_latest_checkpoint(self.checkpoint_dir)

        if latest_checkpoint is None:
            start_step = 0
        else:
            start_step = int(latest_checkpoint.split("_")[-1])

            model_state_dict = torch.load(os.path.join(latest_checkpoint, "model.pth"), map_location="cpu", weights_only=True, mmap=True)
            set_model_state_dict(model=model, model_state_dict=model_state_dict, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))

            optim_state_dict = torch.load(os.path.join(latest_checkpoint, "optimizer.pth"), map_location="cpu", weights_only=True, mmap=True)
            set_optimizer_state_dict(model=model, optimizers=optimizer, optim_state_dict=optim_state_dict, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))

            dataloader_state_dict = torch.load(os.path.join(latest_checkpoint, f"dataloader_rank_{torch.distributed.get_rank()}.pth"), weights_only=True)
            dataloader.load_state_dict(dataloader_state_dict)

        return start_step
    
    def save(self, step, model, optimizer, dataloader):
        model_state_dict = get_model_state_dict(model=model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        optim_state_dict = get_optimizer_state_dict(model=model, optimizers=optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True))

        os.makedirs(os.path.join(self.checkpoint_dir, f"checkpoint_{step}"), exist_ok=True)
        if torch.distributed.get_rank() == 0:
            torch.save(model_state_dict, os.path.join(self.checkpoint_dir, f"checkpoint_{step}", "model.pth"))
            torch.save(optim_state_dict, os.path.join(self.checkpoint_dir, f"checkpoint_{step}", "optimizer.pth"))
        torch.save(dataloader.state_dict(), os.path.join(self.checkpoint_dir, f"checkpoint_{step}", f"dataloader_rank_{torch.distributed.get_rank()}.pth"))
