import datasets
import os
import torch




def get_dataset(cfg):
    name = cfg.data.name
    dataset = datasets.load_dataset(name)

    if cfg.data.train_test_split is not None and cfg.data.train_test_split < 1 and cfg.data.train_test_split > 0:
        dataset = dataset.train_test_split(train_size=cfg.data.train_test_split, seed=cfg.seed + 100)
    return dataset



def truncate_dataset(dataset, trunc):
    """
    Truncates a dataset to a specified length.
    """
    if trunc is None:
        return dataset
    if trunc < 1:
        trunc_len = int(trunc * len(dataset))
    else:
        assert trunc >= 1 and type(trunc) is int
        trunc_len = min(trunc, len(dataset))
    return dataset.select(range(trunc_len))


def compute_training_metadata(cfg, train_dataset):
    # compute steps/epochs
    if hasattr(train_dataset, '__len__'):
        cfg.meta.num_train_examples = len(train_dataset)
        cfg.meta.num_gpus = torch.cuda.device_count()
        cfg.meta.total_batch_size = cfg.meta.num_gpus * cfg.training.batch_size * cfg.training.gradient_accumulation_steps
        cfg.meta.epoch_steps = cfg.meta.num_train_examples // cfg.meta.total_batch_size

        print(f"\nTraining on {cfg.meta.num_train_examples} examples with {cfg.meta.num_gpus} GPUs, batch size {cfg.training.batch_size}, gradient accumulation steps {cfg.training.gradient_accumulation_steps}")
        print(f"Total number of steps per epoch: {cfg.meta.epoch_steps}\n")

    if cfg.training.num_epochs is not None:
        if cfg.training.num_steps is not None:
            raise ValueError('num_epochs and num_steps are mutually exclusive')
        cfg.meta.num_steps = int(cfg.meta.epoch_steps * cfg.training.num_epochs)
    else:
        if cfg.training.num_steps is None:
            raise ValueError('must specify either num_epochs or num_steps')
        cfg.meta.num_steps = cfg.training.num_steps