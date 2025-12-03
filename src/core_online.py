# # bema_callback_and_trainer.py
import copy
import torch
from typing import Dict, List, Tuple, Optional
from transformers import TrainerCallback, TrainerState, TrainingArguments, TrainerControl, Trainer




def _unwrap_model(model):

    return model.module if hasattr(model, "module") else model

class BEMACallback(TrainerCallback):
    """
    Bias-Corrected EMA (BEMA):
        BEMA_t(θ) = EMA_t + α_t (θ_t - θ_0)
        EMA_t     = β_t θ_t + (1 - β_t) EMA_{t-1}
        β_t = (1 + γ t)^(-ema_power),  α_t = (1 + γ t)^(-eta_power)

    Usage patterns:
    - This callback updates internal EMA/BEMA buffers during training.
    - To EVALUATE with BEMA weights, call `callback.swap_to_bema(trainer)` before eval,
      then `callback.swap_to_live(trainer)` to restore training weights. (You can do it
      around your own `trainer.evaluate()` or `trainer.predict()`.)
    - On checkpoint save, this writes a `bema.pt` with BEMA weights.
    """

    def __init__(
        self,
        update_freq: int = 100,
        ema_power: float = 0.5,
        eta_power: float = 0.2,
        update_after: int = 0,
        scaling_lag: int = 10,
        ema_gamma: float = 1.0,
        min_ema_multiplier: float = 0.0,
        device: str = "cpu",
        update_on_cpu: bool = True,
    ):
        self.update_freq = int(update_freq)
        self.ema_power = float(ema_power)
        self.eta_power = float(eta_power)
        self.update_after = int(update_after) if update_after is not None else 0
        self.scaling_lag = int(scaling_lag)
        self.ema_gamma = float(ema_gamma)
        self.min_ema_multiplier = float(min_ema_multiplier)
        self.device = device

        self.do_bema = (self.update_freq > 0) and (self.eta_power > 0)
        self.do_ema = (self.update_freq > 0) and (self.ema_power > 0)

        self.update_on_cpu = update_on_cpu
        self._saved_live_state = None  # for swapping

        # internal state
        self.initialized = False

        # Internal state
        self.bema = {}  # bias-corrected EMA weights
        self.ema = {}  # standard EMA weights
        self.theta0 = {}  # initial weights

        # Bookkeeping for schedule
        self._last_update_step: int = -1

    # --------- Utilities ---------
    def _global_step(self, state: TrainerState) -> int:
        # HF global_step is >= 0; we use 1-based 't' for schedules (more natural).
        t = int(state.global_step)
        return max(0, t)

    def _t_eff(self, t: int) -> int:
        # Effective time index for schedule after a lag; keep at least 0
        return max(0, t - self.scaling_lag)

    def _beta_t(self, t: int) -> float:
        if not self.do_ema:
            return 0.0
        te = self._t_eff(t)
        beta = self.ema_gamma * (1.0 +  te) ** (-self.ema_power)
        if self.min_ema_multiplier > 0.0:
            beta = max(beta, self.min_ema_multiplier)
        return float(beta)

    def _alpha_t(self, t: int) -> float:
        if not self.do_bema:
            return 0.0
        te = self._t_eff(t)
        alpha = self.ema_gamma * (1.0 +  te) ** (-self.eta_power)
        return float(alpha)

    def _initialize_buffers(self, model):
        base_model = _unwrap_model(model)
        for name, p in base_model.named_parameters():
            if not p.requires_grad:
                continue
            self.bema[name] = p.detach().clone().float()
            self.ema[name] = p.detach().clone().float()
            self.theta0[name] = p.detach().clone().float()
            if self.update_on_cpu:
                self.bema[name] = self.bema[name].to('cpu')
                self.ema[name] = self.ema[name].to('cpu')
                self.theta0[name] = self.theta0[name].to('cpu')
        self.initialized = True

    @torch.no_grad()
    def _update_buffers(self, t: int, model):

        base_model = _unwrap_model(model)
        beta_t = self._beta_t(t)
        alpha_t = self._alpha_t(t)

        for name, p in base_model.named_parameters():
            if not p.requires_grad:
                continue
            if self.update_on_cpu:
                p_data = p.detach().to('cpu').float()
            else:
                p_data = p.detach().float()

            if self.do_ema:
                # Update EMA
                self.ema[name].mul_(1 - beta_t).add_(p_data, alpha=beta_t)
            else:
                self.ema[name] = p_data.clone()

            if self.do_bema:
                # Update BEMA
                bias_correction = p_data - self.theta0[name]
                self.bema[name].copy_(self.ema[name] + alpha_t * bias_correction)
            else:
                self.bema[name].copy_(self.ema[name])



    # --------- HF callback hooks ---------
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        model = kwargs['model']
        self._initialize_buffers(model)
        return control


    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']

        t = self._global_step(state) 
        if t < self.update_after: # skip updates until after this step
            return control
        if self.update_freq <= 0: # Disables updates
            return control
        if t == self._last_update_step:
            return control
        if not self.initialized:
            self._initialize_buffers(model)
            return control


        if ((t - self._last_update_step) % self.update_freq) == 0:
            self._update_buffers(t, model)
            self._last_update_step = t
        return control

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Save a `bema.pt` file alongside the checkpoint with the current BEMA weights.
        """

        if not self.initialized:
            return control

        if not self.update_on_cpu:
            bema = {name: p.detach().cpu().clone() for name, p in self.bema.items()}
        else:
            bema = {name: p.detach().clone() for name, p in self.bema.items()}
        
        output_dir = args.output_dir
        if output_dir:
            path = f"{output_dir}/bema.pt"
            torch.save(bema, path)
        return control

    


# --------- Public helpers for swapping ---------
    @torch.no_grad()
    def swap_to_bema(self, trainer):
        """
        Copy current BEMA buffers into the trainer.model (in-place).
        Call this before eval/inference to use BEMA weights.
        """

        # Save live state_dict once, to restore later
        if self._saved_live_state is None:
            self._saved_live_state = {
                n: p.detach().cpu().clone()
                for n, p in _unwrap_model(trainer.model).named_parameters()
            }

        else:
            raise RuntimeError("swap_to_bema called but live state already saved. Did you forget to call swap_to_live()?")

        # Copy BEMA into live model
        for name, target in _unwrap_model(trainer.model).named_parameters():
            if name in self.bema:
                p_bema = self.bema[name]
                target.copy_(p_bema.to(device=target.device, dtype=target.dtype))


    @torch.no_grad()
    def swap_to_live(self, trainer):
        """
        Restore the trainer.model parameters saved by `swap_to_bema`.
        """
        if self._saved_live_state is None:
            return
        
        for name, target in _unwrap_model(trainer.model).named_parameters():
            if name in self._saved_live_state:
                tensor = self._saved_live_state[name]
                target.copy_(tensor.to(device=target.device, dtype=target.dtype))
        self._saved_live_state = None