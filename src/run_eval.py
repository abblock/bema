import os
import shutil
import json
import time
import hydra
from omegaconf import OmegaConf
import pickle
import wandb
import torch
import random

import transformers

from utils import load_wandb_key

from run_bema_training import get_tokenizer, CHAT_TEMPLATE


from core_offline import BEMA, OUEMA, DEMA
from data import get_dataset, truncate_dataset



from vllm import LLM, SamplingParams
import numpy as np
from eval_utils import make_updateable_vllm, update_vllm_weights, eval_model_loss




def get_model(cfg):
    kwargs = {}
    if cfg.model.use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
        assert cfg.training.bf16, 'Flash attention requires bf16'
    if cfg.training.bf16:
        if cfg.training.fp16:
            raise ValueError('Cannot use both fp16 and bf16')
        kwargs['torch_dtype'] = torch.bfloat16
    # if cfg.training.fp16:
        # kwargs['torch_dtype'] = torch.float16

    kwargs['device_map'] = None

    model = transformers.AutoModelForCausalLM.from_pretrained(cfg.model.name, **kwargs)

    
    return model





def get_train_config(parent):
    """"
    Loads results.pkl and returns the config used for training
    Args:
        parent: str, path to folder containing results.pkl
    """
    path = os.path.join(parent, 'results.pkl')
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return results['cfg']



def get_ouema(model, cfg):
    return OUEMA(
        model=model,
        ema_power=cfg.stabilizer.ema_power,
        update_after=cfg.stabilizer.update_after,
        eta_power=cfg.stabilizer.eta_power,
        ema_gamma=cfg.stabilizer.ema_gamma,
        scale_ou_term=cfg.stabilizer.scale_ou_term,
        max_weight=cfg.stabilizer.max_weight,
        scaling_lag=cfg.stabilizer.scaling_lag,
        min_ema_multiplier=cfg.stabilizer.min_ema_multiplier,
        ema_update_after=cfg.stabilizer.ema_update_after,
        device='cpu'
    )



def get_bema(model, cfg):
    return BEMA(
        model=model,
        ema_power=cfg.stabilizer.ema_power,
        update_after=cfg.stabilizer.update_after,
        eta_power=cfg.stabilizer.eta_power,
        ema_gamma=cfg.stabilizer.ema_gamma,
        scaling_lag=cfg.stabilizer.scaling_lag,
        min_ema_multiplier=cfg.stabilizer.min_ema_multiplier,
        ema_update_after=cfg.stabilizer.ema_update_after,
        device='cpu'
    )

def get_dema(model, cfg):
    return DEMA(
        model=model,
        ema_power=cfg.stabilizer.ema_power,
        update_after=cfg.stabilizer.update_after,
        ema_gamma=cfg.stabilizer.ema_gamma,
        scaling_lag=cfg.stabilizer.scaling_lag,
        min_ema_multiplier=cfg.stabilizer.min_ema_multiplier,
        ema_update_after=cfg.stabilizer.ema_update_after,
        device='cpu'
    )





def get_eval_dataloader(cfg, tokenizer, split='test'):
    """
    Gets the eval dataloader and tokenizes it.
    Args:
        cfg: OmegaConf, the config
        tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use
    """
    dataset = get_dataset(cfg)
    if split == 'test':
        try:
            eval_dataset =  dataset['test'] 
        except KeyError:
            eval_dataset = dataset['validation']
    elif split == 'train':
        eval_dataset = dataset['train']
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'.")
    
    if cfg.stabilizer_eval.loss_eval.max_eval_samples is not None:
        eval_dataset = truncate_dataset(eval_dataset, cfg.stabilizer_eval.loss_eval.max_eval_samples)
    eval_dataset = eval_dataset.select_columns(cfg.data.messages_column_name)
    if cfg.data.apply_chat_template:
        eval_dataset = eval_dataset.map(lambda row:{'text':tokenizer.apply_chat_template(row[cfg.data.messages_column_name], tokenize=False, chat_template=CHAT_TEMPLATE)}, batched=True, remove_columns=[cfg.data.messages_column_name])
    else:
        eval_dataset = eval_dataset.rename_column(cfg.data.messages_column_name, 'text')

    if not cfg.stabilizer_eval.use_vllm:
        eval_dataset = eval_dataset.map(lambda row:tokenizer(row['text'], padding=True, truncation=True, max_length=cfg.stabilizer_eval.loss_eval.eval_truncation_level), batched=True, remove_columns=['text'])
        eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.stabilizer_eval.loss_eval.eval_batch_size, num_workers=cfg.stabilizer_eval.loss_eval.eval_num_workers)
    else:
        eval_dataloader = eval_dataset['text']
    return eval_dataloader




def get_sampling_params(generations_cfg, use_vllm=True):
    """
    Given a generations cfg such as `cfg.stabilizer_eval.generations_eval.tasks[i]` returns the sampling params for the task
    """
    if use_vllm:
        sampling_params = {
            'n': generations_cfg.num_return_sequences,
            'temperature': generations_cfg.temperature,
            'max_tokens': generations_cfg.max_new_tokens,
            'seed': generations_cfg.seed,
        }
    else:
        raise ValueError('Sampling params can only be used with vLLM')
    return sampling_params




def get_generation_evaluators(cfg, tokenizer):
    """
    Returns generations evaluators for each task in the config
    Args:
        cfg: OmegaConf, the config
    """
    evaluators, aliases = [], []
    for task in cfg.stabilizer_eval.generations_eval.tasks:
        if task.alias == 'GSM8K' and task.do_task:
            from eval_utils import GSM8KEvaluator
            sampling_arg = get_sampling_params(task.generation_kwargs, use_vllm=cfg.stabilizer_eval.use_vllm)
            evaluator = GSM8KEvaluator(tokenizer, task.num_examples, sampling_arg, use_vllm=cfg.stabilizer_eval.use_vllm, selection_seed=task.selection_seed, save_gens=task.save_gens, num_shots=task.num_shots)
            evaluators.append(evaluator)
            aliases.append(evaluator.alias)
        elif 'MMLU' in task.alias and task.do_task:
            from eval_utils import MMLUEvaluator
            subset = task.alias.split('-')[1]
            sampling_arg = get_sampling_params(task.generation_kwargs, use_vllm=cfg.stabilizer_eval.use_vllm)
            evaluator = MMLUEvaluator(subset, tokenizer, task.num_examples, sampling_arg, use_vllm=cfg.stabilizer_eval.use_vllm, selection_seed=task.selection_seed, use_cot=task.use_cot, save_gens=task.save_gens, num_shots=task.num_shots)
            evaluators.append(evaluator)
            aliases.append(evaluator.alias)
        elif task.alias == 'BoolQ' and task.do_task:
            from eval_utils import BoolQEvaluator
            sampling_arg = get_sampling_params(task.generation_kwargs, use_vllm=cfg.stabilizer_eval.use_vllm)
            evaluator = BoolQEvaluator(tokenizer, task.num_examples, sampling_arg, use_vllm=cfg.stabilizer_eval.use_vllm, selection_seed=task.selection_seed, save_gens=task.save_gens, num_shots=task.num_shots)
            evaluators.append(evaluator)
            aliases.append(evaluator.alias)
        else:
            raise ValueError(f"Unknown task alias: {task.alias}. Please add it to the eval_utils.py file.")
        
            
    return evaluators, aliases



def eval_all_generations(cfg, evaluators, aliases, model, tokenizer, current_results, outputs_dir, step, llm=None):
    """
    Evaluates model on generations tasks
    """
    if cfg.stabilizer_eval.use_vllm:
        assert llm is not None, "llm must be provided if use_vllm is True"
        generation_model = llm
    else:
        raise ValueError("Generations evaluation is only supported with vLLM. Set use_vllm to True in the config.")
    
    for evaluator, alias in zip(evaluators, aliases):
        temp_outputs = evaluator.evaluate_model(
            generation_model, outputs_dir, step, apply_chat_format=True
        )
        current_results.update(temp_outputs)

    return current_results





def do_model_eval(cfg, model, dataloaders, evaluators, aliases, tokenizer, outputs_dir, step, llm=None):
    """
    Given a model and a dataloader, evaluates the model on the dataloader.  Then adds in evaluation on listed tasks
    Args:
        cfg: OmegaConf, the config
        model: transformers.PreTrainedModel, the model to evaluate
        dataloaders: dict str:torch.utils.data.Dataloader, dictionary of the dataloaders to evaluate on
        evaluators: list, a list of evaluators to use for the tasks; each evaluator is a subclass of eval_utils.BaseGenerationsEvaluator
        aliases: list, a list of aliases for the evaluators; each alias is a string
        tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use
        outputs_dir: str, path to save the results of generations if saving
        step: int, step of the model, added to path to generations if saving
        llm: vllm.LLM, the model to use for vllm.  If None, uses the hf model
    """
    if cfg.stabilizer_eval.loss_eval.do_loss_eval and cfg.stabilizer_eval.use_vllm:
        raise ValueError("Cannot use vllm for loss eval.  Use the hf model instead.")
    
    current_results = eval_model_loss(model, dataloaders, tokenizer, device=cfg.device, do_eval=cfg.stabilizer_eval.loss_eval.do_loss_eval)


    if cfg.stabilizer_eval.generations_eval.do_generations:
        current_results = eval_all_generations(cfg, evaluators, aliases, model, tokenizer, current_results, outputs_dir, step, llm=llm)

    return current_results





@torch.no_grad()
@hydra.main(config_path='../hydra_configs/', config_name='master', version_base=None)
def main(cfg):

    
    torch.multiprocessing.set_start_method('spawn')
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    os.environ.update({'TOKENIZERS_PARALLELISM': 'true'})
    ### Meta and Config stuff
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)


    # add runtime info to cfg
    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    cfg.meta = OmegaConf.create({})
    cfg.meta.original_dir = hydra.utils.get_original_cwd()
    cfg.meta.run_dir = os.getcwd()
    cfg.meta.local_rank = int(os.environ.get('LOCAL_RANK', 0))


    if (cfg.stabilizer.ema_update_after_lag is not None) and (cfg.stabilizer.ema_update_after_lag > 0):
        cfg.stabilizer.ema_update_after = cfg.stabilizer.update_after + cfg.stabilizer.ema_update_after_lag


    
    if cfg.stabilizer.train_exp_id is not None:
        cfg.input_parent = os.path.join(cfg.master_parent, '..', cfg.stabilizer.train_exp_id)

    
    if cfg.meta.local_rank == 0:
        print('Config:', cfg)   
    

    #### Getting configs from training
    train_config = get_train_config(os.path.join(cfg.input_parent, cfg.stabilizer_eval.ckpts_directory))
    KEEP_CONFIGS_FROM_TRAIN = [
        'training',
        'logging',
        'model',
        'tokenizer',
        'wandb',

    ]
  
    
    for key in list(train_config.keys()):
        if key not in KEEP_CONFIGS_FROM_TRAIN:
            del train_config[key]

    del train_config['wandb']['use'] ## remove wandb using from train_config to not update the config
    cfg = OmegaConf.merge(cfg, train_config)


    ### Setting up wandb
    if cfg.wandb.use:
        load_wandb_key(cfg)
        cfg.wandb.project = cfg.wandb.project + '-eval'
        cfg.wandb.name = f"eta_{cfg.stabilizer.eta_power}_ema_{cfg.stabilizer.ema_power}_after_{cfg.stabilizer.update_after}_lag_{cfg.stabilizer.scaling_lag}"


        wandb.login(key=cfg.wandb.key,
                    host=cfg.wandb.host)
        cfg.wandb.key = None  # redact key
        wandb.init(project=cfg.wandb.project,
                   entity=cfg.wandb.entity,
                   group=cfg.wandb.group,
                   config=OmegaConf.to_container(cfg),
                   name=cfg.wandb.name)
    

    ### Get model checkpoints
    sorted_ckpt_paths = BEMA.get_sorted_paths(os.path.join(cfg.input_parent, cfg.stabilizer_eval.ckpts_directory), min_ckpt=cfg.stabilizer.min_ckpt, max_ckpt=cfg.stabilizer.max_ckpt)


    ### Get the model and tokenizer
    model_load_start = time.time()
    tokenizer = get_tokenizer(cfg)
    model = get_model(cfg)
    if cfg.model.dtype == 'bfloat16':
        model.to(torch.bfloat16)

    model.resize_token_embeddings(len(tokenizer))
    model.to('cpu')
    for param in model.parameters():
        param.requires_grad = False

    if cfg.stabilizer_eval.use_vllm:
        if 'gemma' in cfg.model.name.lower():
            cfg.model.dtype = 'bfloat16'
        
        
        llm = make_updateable_vllm(sorted_ckpt_paths[0][0],rank=0, world_size=1, gpu_memory_utilization=cfg.stabilizer_eval.gpu_memory_utilization, dtype=cfg.model.dtype)
        update_vllm_weights(llm, model, device='cuda')

    else:
        llm = None


    
    
    if cfg.meta.num_tokens != model.config.vocab_size: # resize if tokenizer added tokens
        print(f'Model: resizing vocab size from {model.config.vocab_size} to {cfg.meta.num_tokens}')
        model.resize_token_embeddings(cfg.meta.num_tokens)
    
    
    if cfg.stabilizer.stabilizer == 'ou_ema':
        stabilizer =  get_ouema(model, cfg)
    elif cfg.stabilizer.stabilizer == 'bema':
        stabilizer =  get_bema(model, cfg)
    elif cfg.stabilizer.stabilizer == 'dema':
        stabilizer =  get_dema(model, cfg)
    else:
        raise ValueError(f"Invalid stabilizer: {cfg.stabilizer.stabilizer}")

    




    model_load_end = time.time()
    print(f"\nModel loaded in {model_load_end - model_load_start:.0f} seconds\n")

    ### Get the eval dataloader
    data_load_start = time.time()
    if cfg.stabilizer_eval.loss_eval.do_loss_eval:
        assert cfg.stabilizer_eval.loss_eval.do_train_loss_eval or cfg.stabilizer_eval.loss_eval.do_test_loss_eval, "Must do train or test loss eval if cfg.stabilizer_eval.loss_eval.do_loss_eval is True"
        dataloaders = {}
        if cfg.stabilizer_eval.loss_eval.do_train_loss_eval:
            train_dataloader = get_eval_dataloader(cfg, tokenizer, split='train')
            dataloaders['train'] = train_dataloader
        if cfg.stabilizer_eval.loss_eval.do_test_loss_eval:
            test_dataloader = get_eval_dataloader(cfg, tokenizer, split='test')
            dataloaders['test'] = test_dataloader
    else:
        dataloaders = {}  


    if cfg.stabilizer_eval.generations_eval.do_generations:
        generations_evaluators, generation_aliases = get_generation_evaluators(cfg, tokenizer)
    else:
        generations_evaluators = []
        generation_aliases = []
    data_load_end = time.time()
    print(f"\nData loaded in {data_load_end - data_load_start:.0f} seconds\n")


    ### Preparing Saving
    outputs_dir = os.path.join(cfg.master_parent, 'eval')
    os.makedirs(outputs_dir, exist_ok=True)


    ### Evaluating
    evaluate_start = time.time()


    results = {}

    results[0] = do_model_eval(cfg, model, dataloaders, generations_evaluators, generation_aliases, tokenizer, outputs_dir, 0, llm=llm)


    if cfg.wandb.use:
        wandb.log(results[0], step=0)
    else:
        print(f"Step 0: {results[0]}")
    

    previous_ckpt = - cfg.stabilizer.update_freq

    print(f"Evaluating from ckpt 0 to ckpt {sorted_ckpt_paths[-1][1]}, every {cfg.stabilizer.update_freq} steps, i.e., {sorted_ckpt_paths[-1][1] // cfg.stabilizer.update_freq} checkpoints")
    for path, step in sorted_ckpt_paths:
        


        if step - previous_ckpt < cfg.stabilizer.update_freq: ## Skip if not enough steps have passed
            continue
        else:
            print(f"Evaluating checkpoint {step} at {path}")

        previous_ckpt = step

        current_weights = stabilizer.load_weights(path)
        if cfg.model.dtype == 'bfloat16':
            current_weights = {k: v.to(torch.bfloat16) for k, v in current_weights.items()}
        stabilizer.update(current_weights, step)

        if cfg.stabilizer_eval.use_vllm:
            update_vllm_weights(llm, stabilizer.running_model, device='cuda')

        
        current_results = do_model_eval(cfg, stabilizer.running_model, dataloaders, generations_evaluators, generation_aliases, tokenizer, outputs_dir, step, llm=llm)
        
        if cfg.stabilizer.stabilizer == 'ou_ema':
            if stabilizer.do_ou:
                stabilizer_time_from_start = stabilizer.get_stabilizer_time(step)
                mult_1, mult_2 = stabilizer.get_polynomial_multipliers(stabilizer_time_from_start, stabilizer.eta_power, stabilizer.max_weight)
                mult_2 = -mult_2
                current_results['eval/stabilizer_time'] = stabilizer_time_from_start
                current_results['eval/mult_1'] = mult_1
                current_results['eval/mult_2'] = mult_2
        elif cfg.stabilizer.stabilizer == 'bema':
            if stabilizer.do_bema:
                stabilizer_time_from_start = stabilizer.get_stabilizer_time(step)
                mult_1, mult_2 = stabilizer.get_bema_correction_weights(stabilizer_time_from_start, stabilizer.eta_power)
                mult_2 = -mult_2
                current_results['eval/stabilizer_time'] = stabilizer_time_from_start
                current_results['eval/mult_1'] = mult_1
                current_results['eval/mult_2'] = mult_2

        if stabilizer.do_ema:
            ema_time_from_start = stabilizer.get_ema_time(step)
            current_results['eval/ema_multiplier'] = stabilizer.get_ema_weights(ema_time_from_start, stabilizer.ema_power, stabilizer.ema_gamma, stabilizer.min_ema_multiplier)
        
        results[step] = current_results




        if cfg.wandb.use:
            wandb.log(current_results, step=step)
        else:
            print(f"Step {step}: {current_results}")
        

    evaluate_end = time.time()
    print(f"\nEvaluation took {evaluate_end - evaluate_start:.0f} seconds\n")

    ### Saving results
    train_config = get_train_config(os.path.join(cfg.input_parent, cfg.stabilizer_eval.ckpts_directory))
    output = {
        'cfg': OmegaConf.to_container(cfg),
        'results': results,
        'train_config': train_config,
    }
    
    with open(os.path.join(outputs_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(output, f)

    print('Saved results to', outputs_dir)





if __name__ == '__main__':
    master_start = time.time()
    main()
    master_end = time.time()
    print(f"Master took {master_end - master_start:.0f} seconds")