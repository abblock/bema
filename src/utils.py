
import os
from omegaconf import OmegaConf




def cache_sync(accelerator, fn):  # ensure first call is blocking
    if accelerator.is_local_main_process:
        result = fn()
    accelerator.wait_for_everyone()
    if not accelerator.is_local_main_process:
        result = fn()
    return result



def load_wandb_key(cfg):
    if cfg.wandb.use:
        with open('.wandb_token', 'r') as f:
            cfg.wandb.key = f.read().strip()




