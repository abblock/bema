import os
import time
import hydra
from omegaconf import OmegaConf
import pickle
import torch
import random
import transformers

from accelerate import Accelerator
from utils import load_wandb_key, cache_sync

from data import get_dataset, truncate_dataset, compute_training_metadata
from trl import SFTConfig, SFTTrainer
from torch.optim.lr_scheduler import LambdaLR
from functools import partial

from core_online import BEMACallback





CHAT_TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

def get_tokenizer(cfg, CHAT_TEMPLATE=CHAT_TEMPLATE):
    """
    Returns a tokenizer based on the Hydra config.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.tokenizer.name)

    if type(cfg.tokenizer.eos_token) is str:
        if cfg.tokenizer.eos_token in tokenizer.special_tokens_map.values():
            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(cfg.tokenizer.eos_token)
        else:
            tokenizer.add_special_tokens({'eos_token': cfg.tokenizer.eos_token})
    elif cfg.tokenizer.eos_token is int:
        tokenizer.eos_token_id = cfg.tokenizer.eos_token

        
    if type(cfg.tokenizer.pad_token) is str:
        if cfg.tokenizer.pad_token in tokenizer.special_tokens_map.values():
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(cfg.tokenizer.pad_token)
        else:
            tokenizer.add_special_tokens({'pad_token': cfg.tokenizer.pad_token})
    elif cfg.tokenizer.pad_token is int:
        tokenizer.pad_token_id = cfg.tokenizer.pad_token
    elif cfg.tokenizer.pad_token is None:
        tokenizer.pad_token_id = len(tokenizer)
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    tokenizer.padding_side = 'left'

    
    tokenizer.chat_template = CHAT_TEMPLATE
    
    cfg.meta.eos_token_id = tokenizer.eos_token_id
    cfg.meta.pad_token_id = tokenizer.pad_token_id
    cfg.meta.num_tokens = len(tokenizer)

    return tokenizer





def compute_model_metadata(cfg, model):
    cfg.meta.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)




def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int, min_lr_multiplier: float = 0.0):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(min_lr_multiplier, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))



def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_multiplier=0.0):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_multiplier=min_lr_multiplier,
    )
    return LambdaLR(optimizer, lr_lambda)



class BEMASFTTrainer(SFTTrainer):
    def __init__(self, cfg, *args, bema_cb=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

        # If user didn’t pass the callback explicitly, try to find it.
        if bema_cb is None:
            for cb in getattr(self.callback_handler, "callbacks", []):
                # Import class name only; user might subclass BEMACallback.
                if cb.__class__.__name__.lower().startswith("bema"):
                    bema_cb = cb
                    break
        if bema_cb is None:
            print("Warning: BEMASFTTrainer initialized without a BEMACallback instance. BEMA eval will be skipped.")
        self.bema_cb = bema_cb

    def create_optimizer_and_scheduler(self, num_training_steps):
        # Respect config-customized linear schedule
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            self.cfg.training.warmup_steps,
            self.cfg.meta.num_steps,
            self.cfg.training.min_lr_multiplier,
        )

    def evaluate(self, *args, **kwargs):
        bema_eval_cfg = self.cfg.stabilizer.eval

        # 1) Vanilla eval (live weights)
        metrics_base = {}
        if getattr(bema_eval_cfg, "eval_vanilla", True):
            mb = super().evaluate(*args, **{**kwargs, "metric_key_prefix": "vanilla"})
            # Prefix eval/vanilla/*
            metrics_base = {f"eval/vanilla/{k}": v for k, v in mb.items()}

        # 2) BEMA eval (if callback present)
        if self.bema_cb is None:
            return metrics_base

        if getattr(bema_eval_cfg, "eval_bema", True):
            # Swap -> evaluate -> restore using the callback API
            try:
                self.bema_cb.swap_to_bema(self)
                mbema = super().evaluate(*args, **{**kwargs, "metric_key_prefix": "bema"})
            finally:
                # Always restore live weights
                self.bema_cb.swap_to_live(self)

            metrics_bema = {f"eval/{k}": v for k, v in mbema.items()}
        else:
            metrics_bema = {}

        return {**metrics_base, **metrics_bema}


def get_model(cfg):
    kwargs = {}
    if cfg.model.use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
        assert cfg.training.bf16, 'Flash attention requires bf16'
    if cfg.training.bf16:
        if cfg.training.fp16:
            raise ValueError('Cannot use both fp16 and bf16')
        kwargs['torch_dtype'] = torch.bfloat16


    kwargs['device_map'] = None

    model = transformers.AutoModelForCausalLM.from_pretrained(cfg.model.name, **kwargs)

    
    return model




def get_sft_args(cfg):  # Hydra args -> HuggingFace Trainer args
    args = SFTConfig(
        output_dir=os.path.join(cfg.master_parent, cfg.logging.output_dir),
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        eval_strategy="steps",
        eval_steps=cfg.logging.eval_interval,
        save_steps=cfg.logging.checkpoint_interval,
        logging_steps=cfg.logging.log_interval,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_steps=cfg.meta.num_steps,
        dataloader_num_workers=cfg.training.num_workers,
        dataloader_pin_memory=True,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        report_to="none",
        learning_rate=cfg.training.lr,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        seed=cfg.seed + 121,
        warmup_steps=cfg.training.warmup_steps,
        max_length=cfg.data.max_len_truncation,
    )
    return args



@hydra.main(config_path='../hydra_configs/', config_name='master', version_base=None)
def main(cfg):

    torch.manual_seed(cfg.seed)
    torch.multiprocessing.set_start_method('spawn')
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    os.environ.update({'TOKENIZERS_PARALLELISM': 'true'})
    ### Meta and Config stuff
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)


    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    cfg.meta = OmegaConf.create({})
    cfg.meta.original_dir = hydra.utils.get_original_cwd()
    cfg.meta.run_dir = os.getcwd()
    cfg.meta.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    load_wandb_key(cfg)
    if (cfg.stabilizer.ema_update_after_lag is not None) and (cfg.stabilizer.ema_update_after_lag > 0):
        cfg.stabilizer.ema_update_after = cfg.stabilizer.update_after + cfg.stabilizer.ema_update_after_lag
    
    if cfg.meta.local_rank == 0:
        print('Config:', cfg)   




    ### Get the model and tokenizer
    model_load_start = time.time()
    tokenizer = get_tokenizer(cfg)
    model = get_model(cfg)
    if cfg.model.dtype == 'bfloat16':
        model.to(torch.bfloat16)

    model.resize_token_embeddings(len(tokenizer))
    if cfg.meta.num_tokens != model.config.vocab_size: # resize if tokenizer added tokens
        print(f'Model: resizing vocab size from {model.config.vocab_size} to {cfg.meta.num_tokens}')
        model.resize_token_embeddings(cfg.meta.num_tokens)
    
    model_load_end = time.time()
    print(f"\nModel loaded in {model_load_end - model_load_start:.0f} seconds\n")




    
    accelerator = Accelerator()

    print(f"Getting datset {cfg.data.name}...")
    data_load_start = time.time()
    dataset = cache_sync(accelerator, lambda: get_dataset(cfg))
    raw_train_dataset = dataset['train']
    try:
        raw_eval_dataset =  dataset['test'] 
    except KeyError:
        raw_eval_dataset = dataset['validation']


    if cfg.data.truncate_train is not None:
        raw_train_dataset = truncate_dataset(raw_train_dataset, cfg.data.truncate_train)
    if cfg.data.truncate_eval is not None:
        raw_eval_dataset = truncate_dataset(raw_eval_dataset, cfg.data.truncate_eval)

    train_dataset = raw_train_dataset
    eval_dataset = raw_eval_dataset
    compute_model_metadata(cfg, model)
    compute_training_metadata(cfg, train_dataset)
    data_load_end = time.time()
    print(f"\nDataset loaded in {data_load_end - data_load_start:.0f} seconds\n")




    trainer_args = get_sft_args(cfg)

    if cfg.wandb.use and accelerator.is_main_process:
        import wandb
        wandb.login(
                    key=cfg.wandb.key,
                    relogin=True,
                    host=cfg.wandb.host
                    )
        cfg.wandb.key = None  # redact key
        wandb.init(project=cfg.wandb.project,
                   entity=cfg.wandb.entity,
                   group=cfg.wandb.group,
                   config=OmegaConf.to_container(cfg),
                   name=cfg.wandb.name)
        trainer_args.report_to.append('wandb')
        accelerator.print(f'Logging to W&B: {wandb.run.url}')


    bema_cb = BEMACallback(
        update_freq=cfg.stabilizer.update_freq,
        ema_power=cfg.stabilizer.ema_power,
        eta_power=cfg.stabilizer.eta_power,
        update_after=cfg.stabilizer.ema_update_after,
        scaling_lag=cfg.stabilizer.scaling_lag,
        ema_gamma=cfg.stabilizer.ema_gamma,
        min_ema_multiplier=cfg.stabilizer.min_ema_multiplier,
        device='cpu'
    )

    if not cfg.training.use_bema:
        print("Warning: BEMA is disabled. Set `use_bema: True` in the config to enable BEMA.")
        bema_cb = None
        trainer = BEMASFTTrainer(
        cfg,
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        bema_cb=bema_cb
        )
    else:
        print("BEMA is enabled. Initializing BEMASFTTrainer with BEMACallback.")
        trainer = BEMASFTTrainer(
            cfg,
            model=model,
            args=trainer_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=[bema_cb],
            bema_cb=bema_cb
        )


    trainer.train()


    if cfg.logging.eval_final:
        trainer.evaluate()
    

    if cfg.meta.local_rank == 0:
        outputs = {
        'log_history': trainer.state.log_history,
        'cfg': OmegaConf.to_container(cfg),
        }
        ## Preparing Saving
        outputs_dir = os.path.join(cfg.master_parent, 'train')
        os.makedirs(outputs_dir, exist_ok=True)
        with open(os.path.join(outputs_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(outputs, f)

        print('Saved results to', outputs_dir)


if __name__ == '__main__':
    master_start = time.time()
    main()
    master_end = time.time()
    print(f"Master took {master_end - master_start:.0f} seconds")