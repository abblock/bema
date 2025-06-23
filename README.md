# Code Repo for EMA Without the Lag: Bias-Corrected Iterate Averaging Schemes

This is the code for [EMA Without the Lag: Bias-Corrected Iterate Averaging Schemes]() by Adam Block and Cyril Zhang.

To run, first make a virtual environment with Python 3.10.12 and install the requirements with `pip install -r requirements.txt`.  All configs are input through [hydra](https://hydra.cc/docs/intro/).  

This repo serves three purposes:

1. Providing an HF callback that allows implementation of our core algorithm, BEMA, to be run while training a model.
2. Providing an offline stabilizer that runs BEMA (as well as other stabilizers discussed in our paper) on an offline directory of precomputed checkpoints of a given model.
3. Easy reproducibility of the core empirical results in the paper.

The paper's results were conducted using the offline approach with precomputed checkpoints.


## Stabilizing while Training with an HF Callback

The central algorithmic intervention, BEMA, is included as an HF callback in `src/core_online.py`.  To use this while training, one could run:
```
python src/run_bema_training.py model.name=<hf-path-to-model> data.name=<hf-path-to-data> stabilizer.ema_power=<ema-power> stabilizer.eta_power=<eta-power> stabilizer.update_freq=<update-freq> ...
```
Some relevant parameters are:

- `model.name` is a Huggingface path to a model, e.g., `'Qwen/Qwen2.5-1.5B'`.
- `data.name` is a Huggingface path to a dataset.  One example would be to run `python src/make_tulu_data.py` when logged into Huggingface with an appropriate username set, then set `data.name=<hf-repo>/tulu-3-sft-mixture-split-seed-1337-filtered`.
- `stabilizer.ema_power` is a float between 0 and 1 determining how aggressively to EMA. (Setting this to `-1` removes EMA.)
- `stabilizer.eta_power` is a float between 0 and 1 determining how aggressively to apply the BEMA correction. (Setting this to `-1` removes the bias correction.)
- `stabilizer.update_freq` is an int determining the number of gradient steps in between BEMA updates

Parameters associated with training such as learning rate, gradient accumulation steps, number of epochs, and many more can be found in `master.yaml` under `training` or `logging`.  Note that `wandb` is used by default and if so, the wandb token needs to be saved in a file called `.wandb_token` for easy logging in.  If wandb is not desired, set `wandb.use=False`.



## Stabilizing Offline with Precomputed Checkpoints

It is significantly more efficient to run repeated stabilization on a single training trajectory with cached checkpoints.  Thus all experiments in the paper were run in this way.  Three stabilizers are available in `src/core_offline.py` including `BEMA`, `OUEMA`, and `DEMA`.  Note that standard `EMA` can be run by using `BEMA` and setting `eta_power=-1`.  We also consider 4 evaluations: `loss`, `boolq`, `gsm8k`, and `mmlu_hs` described in the paper.  To do offline stabilization, run
```
python src/run_eval.py stabilizer=<stabilizer> stabilizer_eval=<eval> stabilizer.ckpts_directory=<path-to-checkpoints> stabilizer.eta_power=<eta-power> stabilizer.ema_power=<ema-power> stabilizer.update_freq=<update-freq> ...
```
Some relevant parameters are:

- `stabilizer`: one of `bema`, `oueama`, or `dema`.  Which stabilizer to apply.
- `stabilizer_eval`: one of `loss`, `boolq`, `gsm8k`, or `mmlu_hs`.  Which evaluation task to consider.
- `stabilizer.ckpts_directory` is a path to a directory output by the training script, `src/run_bema_training.py`, i.e., it should have folders of the form `checkpoint-<ckpt>` as well as a `results.pkl` file which loads into a dict with the key `cfg`, which contains the relevant Hydra configs.  In order to run vanilla training without stabilization, run `src/run_bema_training.py training.use_bema=False`.

There are several stabilizer-specific hyperparameters described above and in the paper that are documented in `src/core_offline.py` and can be found in `hydra_configs/stabilizers/<stabilizer>.yaml`.