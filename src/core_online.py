import torch
from transformers import TrainerCallback, TrainerState, TrainingArguments, TrainerControl
import copy

def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model




class BEMACallback(TrainerCallback):
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
    ):
        """
        Callback for HF SFTTrainer that implements BEMA (Bias Corrected Exponential Moving Average).  The returned model weights scale like 
            θₜ' = αₜ·(θₜ - θ₀) + EMAₜ
        where θₜ is the current model weights, θ₀ is a snapshot of the model weights at the first update_after step, EMAₜ is the exponential moving average of the model weights, and αₜ is a scaling factor that decays with the number of steps t as αₜ = (1 + γ·t)⁻ᵉᵗᵃ.
        The EMA is computed as:
            EMAₜ = βₜ·θₜ + (1 - βₜ)·EMAₜ₋₁
        where βₜ is a decay factor that decays with the number of steps t as βₜ = (1 + γ·t)⁻ᵉᵐᵃ.
        Args:
            update_freq (int): How often to update the BEMA weights (in steps). (default = 100)
            ema_power (float): Power for the EMA decay factor βₜ = (1 + γ·t)⁻ᵉᵐᵃ. (default = 0.5)
            eta_power (float): Power for the BEMA scaling factor αₜ = (1 + γ·t)⁻ᵉᵗᵃ. (default = 0.2)
            update_after (int): Number of steps to wait before starting to update the BEMA weights. (default = 0)
            scaling_lag (int): Number of steps to lag the scaling factor αₜ. (default = 10)
            ema_gamma (float): Initial value for the EMA decay factor. (default = 1.0)
            min_ema_multiplier (float): Minimum value for the EMA decay factor. (default = 0.0)
            device (str): Device to use for the BEMA buffers, e.g. "cpu" or "cuda".  Note that in most cases, this device SHOULD BE DIFFERENT from the device used for training in order to avoid OOM. (default = "cpu")
        """        

        
        # user-provided hyperparams
        self.update_freq = update_freq
        self.ema_power = ema_power
        self.eta_power = eta_power
        self.update_after = update_after if update_after is not None else 0
        self.scaling_lag = scaling_lag
        self.ema_gamma = ema_gamma
        self.min_ema_multiplier = min_ema_multiplier
        self.device = device

        # internal state
        self.initialized = False
        self.param_names = []    # references to training model params
        self.thetat_params = []  # references to training model params
        self.theta0_params = []  # θ₀ buffers (on self.device)
        self.ema_params = []      # EMA buffers (on self.device)
        self.running_model = None # a copy of the model to run BEMA on


    @torch.no_grad()
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = _unwrap_model(kwargs["model"])

        self.running_model = copy.deepcopy(model).to(self.device)
        # cache params once in a fixed order
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.param_names.append(name)
            self.thetat_params.append(p)

            # clone θ₀ and EMA on device
            theta0_buf = p.data.detach().to(self.device).clone()
            ema_buf = theta0_buf.clone()  # initialize EMA with θ₀
            self.theta0_params.append(theta0_buf)
            self.ema_params.append(ema_buf)

           

        self.initialized = False

    def _ema_beta(self, t: int) -> float:
        if self.ema_power < 0:
            return 1.0 # no EMA, just BEMA
        beta = (1 + self.ema_gamma * t) ** (-self.ema_power)
        return max(beta, self.min_ema_multiplier)

    def _bema_alpha(self, t: int) -> float:
        if self.eta_power < 0:
            return 0.0 # no BEMA, just EMA
        return (1 + self.ema_gamma * t) ** (-self.eta_power)
    
    @staticmethod
    def update_weight_(
        first_tensor: torch.Tensor,
        second_tensor: torch.Tensor,
        alpha: float,
        beta: float,
    ):
        """
        Updates the first tensor in place using the second tensor and the given alpha and beta values, i.e., first_tensor = alpha * first_tesnor + beta * second_tensor.
        """
        first_tensor.mul_(alpha)  # first_tensor = alpha * first_tensor
        first_tensor.add_(second_tensor, alpha=beta)  # first_tensor += beta * second_tensor
        return first_tensor

    @torch.no_grad()
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        step = state.global_step
        if step is None:
            return
        

        # 1) When we first cross update_after, snapshot θ₀
        if not self.initialized and step >= self.update_after:
            for thetat_param, theta0_param, ema_param in zip(
                self.thetat_params, self.theta0_params, self.ema_params
            ):
                model = _unwrap_model(kwargs["model"])
                # copy θₜ to θ₀
                theta0_param.copy_(thetat_param.data.detach().to(self.device))
                # initialize EMA with θ₀
                ema_param.copy_(theta0_param)
            self.initialized = True

        elif step >= self.update_after and step % self.update_freq == 0:
            model = _unwrap_model(kwargs["model"])

        # 2) skip until after update_after
        if not self.initialized:
            return

        # 3) frequency check
        if step % self.update_freq != 0:
            return


        # 4) compute shared t (with scaling lag)
        t = max(step - self.update_after + self.scaling_lag, 1)
        beta = self._ema_beta(t)
        alhpa = self._bema_alpha(t)

        # 5) gather the current θₜ from GPU → CPU once, non‐blocking
        new_thetats = [p.data.detach().to(self.device, non_blocking=True) for p in self.thetat_params]

        # 6) EMA update:  ema ← β·θₜ + (1–β)·ema
        torch._foreach_mul_(self.ema_params, 1 - beta)                      # ema = (1 -β) * ema
        torch._foreach_add_(self.ema_params, new_thetats, alpha=beta)      # ema += β*θₜ
        
        # 7) BEMA:  corr = (θₜ - θ₀)*α + ema
        #    we do this in three foreach calls to avoid Python loops:
        deltas = [t.clone() for t in new_thetats]                         # a) δ = θₜ.clone()
        torch._foreach_sub_(deltas, self.theta0_params)                   # b) δ.sub_(θ₀)
        torch._foreach_mul_(deltas, alhpa)                                 # c) δ.mul_(α)
        torch._foreach_add_(deltas, self.ema_params)                       # d) δ.add_(ema)
        
        # 8) write back into a *single* flat state_dict and load
        sd_run = self.running_model.state_dict()
        for name, corr in zip(self.param_names, deltas):
            sd_run[name].copy_(corr)
        self.running_model.load_state_dict(sd_run, strict=False)



    @torch.no_grad()
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            # assemble a final state_dict from the EMA+OU buffers
            final_sd = {}
            for name, buf in zip(self.param_names, self.ema_bufs):
                final_sd[name] = buf.clone()
            path = f"{args.output_dir}/bema.pt"
            torch.save(final_sd, path)
            print(f"Saved BEMA to {path}")