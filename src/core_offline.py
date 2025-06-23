import os
import torch
import copy

from safetensors.torch import load_file



def clamp(val, max_val=1.0, min_val=0.0):
    """
    Clamps a value between max_val and min_val
    """
    return max(min(val, max_val), min_val)

class BaseStabilizer():
    def __init__(self, device):
        """ 
        Base stabilizer class for postprocessing checkpoints along training.
        Args:
            model: Huggingface model object
        """
        self.device=device

    @staticmethod
    def _get_ckpt(path):
        """
        Given a path to a checkpoint saved as `checkpoint-{step}`, returns the step number.
        Args:
            path: 
        """
        fname = os.path.basename(path)
        try:
            return int(fname.split('-')[1])
        except Exception as e:
            return None

    @staticmethod
    def get_sorted_paths(parent, min_ckpt=None, max_ckpt=None):
        """
        Given a path to training checkpoints from HF run with each checkpoint folder named `checkpoint-{step}`, returns sorted list of paths.
        Args:
            parent: str, path to folder containing checkpoints
            min_ckpt: int, minimum checkpoint to consider. default None
            max_ckpt: int, maximum checkpoint to consider. default None
        """

        unsorted_fnames = os.listdir(parent)
        parsed_fnames = [(fname, BaseStabilizer._get_ckpt(fname)) for fname in unsorted_fnames if BaseStabilizer._get_ckpt(fname) is not None]
        if min_ckpt is not None:
            parsed_fnames = [(fname, ckpt) for fname, ckpt in parsed_fnames if ckpt >= min_ckpt]
        if max_ckpt is not None:
            parsed_fnames = [(fname, ckpt) for fname, ckpt in parsed_fnames if ckpt <= max_ckpt]
            
        sorted_fnames = sorted(parsed_fnames, key=lambda x: x[1])
        sorted_paths = [(os.path.join(parent, fname), ckpt) for fname, ckpt in sorted_fnames]
        return sorted_paths


    def load_weights(self, path):
        """
        Given a path to a checkpoint, loads the weights in pytorch format as a dictionary of tensors.

        Args:
            path: str, path to checkpoint
        """
        
        weights_files = [f for f in os.listdir(path) if f.endswith('.safetensors') and f.startswith('model')]
        
        weights = {}
        for wf in weights_files:
            weights.update(load_file(os.path.join(path, wf), device=self.device))
        return weights
    
    
    @staticmethod
    def _weight_update(out_tensor, first_tensor, mult1, second_tensor, mult2):
        """
        Does a linear interpolation so that out_tensor = mult1 * first_tensor + mult2 * second_tensor
        Args:
            out_tensor: torch.Tensor, the tensor to be updated
            first_tensor: torch.Tensor, the first tensor to be multiplied by mult1
            mult1: float, the multiplier for first_tensor
            second_tensor: torch.Tensor, the second tensor to be multiplied by mult2
            mult2: float, the multiplier
        """
        return out_tensor.copy_(first_tensor * mult1 + second_tensor * mult2)
        # return first_tensor * mult1 + second_tensor * mult2



    @staticmethod
    def _assign_new_weights_(temp_weights, weights_dict):
        """
        Assigns the weights in weights_dict to the model temp_model
        Args:
            temp_model: transformers.PreTrainedModel, the model to assign the weights to
            weights_dict: dict, a dictionary of tensors to assign to the model
        """
        for key, value in temp_weights.items():
            if key in weights_dict:
                value.copy_(weights_dict[key])


    def update(self):
        """
        Subclasses must implement this method to update the model weights
        """
        raise NotImplementedError(f"Update must be implemented in subclass {self.__class__.__name__}")


    @staticmethod
    def get_ema_weights(time, ema_power, ema_gamma=1.0, min_multiplier=0.0):
        """
        Returns the weights for exponential moving average.  returns weight beta such that we update ema by theta_{EMA,t+1} = beta * theta_{t} + (1 - beta) * theta_{EMA,t}
        Computes beta = (1 + gamma * time)^(-ema_power) * ema_gamma and floors out at min_multiplier.
        Args:
            time: int, the time from the start of training
            ema_power: float, the power to which the time is raised
            ema_gamma: float, the gamma parameter for EMA
            min_multiplier: float, the minimum value of the multiplier
        """
        multiplier = max((1 + ema_gamma *time)**(-ema_power), min_multiplier)
        return multiplier
    

    def get_ema_time(self, step):
        """
        Returns the time from the start of the training for the EMA correction.
        """
        update_time = max(step - self.ema_update_after + self.scaling_lag, 1)
        return update_time



    def vanilla_update(self, update_model, new_weights):
        """
        Updates the model with the new weights
        """
        # if not (hasattr(new_weights, 'keys') and callable(getattr(new_weights, 'keys'))):
        #     new_weights = new_weights.state_dict()
        running_state_dict = update_model.state_dict()
        for key, running_weight in running_state_dict.items():
            if key in new_weights.keys():
                running_state_dict[key] = new_weights[key]
        update_model.load_state_dict(running_state_dict)


    def ema_update(self, update_model, new_weights, step):
        """
        Updates the model with an EMA of the new weights
        """

        ema_time_from_start = self.get_ema_time(step)
        beta = self.get_ema_weights(ema_time_from_start, self.ema_power, self.ema_gamma, self.min_ema_multiplier)
        running_state_dict = update_model.state_dict()
        for key, running_weight in running_state_dict.items():
            if key in new_weights.keys():
                running_state_dict[key] = self._weight_update(out_tensor=running_weight, first_tensor=running_weight, mult1=1 - beta, second_tensor=new_weights[key], mult2=beta)
        update_model.load_state_dict(running_state_dict)



    def check_if_update(self, step):
        """
        Returns True if the model should be updated at this step.
        """
        return step >= min(self.update_after, self.ema_update_after)
    


    def get_stabilizer_time(self, step):
        """
        Returns the time from the start of the training for the OU-EMA correction.
        """
        update_time = max(step - self.update_after + self.scaling_lag , 1)
        return update_time

    



class OUEMA(BaseStabilizer):

    def __init__(
        self,
        model,
        ema_power=0.5,
        update_after=0,
        eta_power=0.2,
        ema_gamma=1.0,
        scale_ou_term=True,
        max_weight=10.0,
        scaling_lag=0,
        min_ema_multiplier=0.0,
        ema_update_after=None,
        device='cpu',
    ):
        """
        Class for implementing OU-EMA correction to stabilize training post-hoc on saved checkpoints.  The OU-correction returns
        theta_t' = (1 - (1 + t)^-eta_power)^-1 * theta_t - (1 - (1 + t)^-eta_power)^-1 * (1 + t)^-eta_power * theta_0
        The EMA correction returns theta_{EMA,t+1} = beta_t * theta_{t} + (1 - beta_t) * theta_{EMA,t}

        Args:
            model: transformers.PreTrainedModel, the model to be stabilized
            ema_power: float, the power to which the time is raised for EMA
            update_after: int, the number of steps after which to start the OU-EMA correction (default 0)
            eta_power: float, the power to which the time is raised for OU (default 0.2)
            ema_gamma: float, the gamma parameter for EMA
            scale_ou_term: bool, whether to scale the OU term (default True)
            max_weight: float, the maximum weight for the OU term (default 10.0)
            scaling_lag: int, the lag for the OU term (default 0)
            min_ema_multiplier: float, the minimum value of the EMA multiplier (default 0.0)
            ema_update_after: int, the number of steps after which to start the EMA correction if None then is set to the OU update after (default None)
            device: str, the device to use (default 'cuda')
        """
        super().__init__(device)
        self.ema_power = ema_power
        self.ema_gamma = ema_gamma
        if self.ema_power >= 1 or self.ema_power <= 0 or self.ema_power is None:
            self.do_ema = False
            print("\nNot using EMA\n")
        else:
            self.do_ema = True
            print("\nUsing EMA\n")

        self.update_after = update_after
        if ema_update_after is None:
            self.ema_update_after = update_after
        else:
            self.ema_update_after = ema_update_after

        self.eta_power = eta_power
        if self.eta_power < 0 or self.eta_power is None:
            self.do_ou = False
            print("\nNot using OU Debiasing\n")
        else:
            self.do_ou = True
            print("\nUsing OU Debiasing\n")

        self.scale_ou_term = scale_ou_term
        self.max_weight = max_weight
        self.scaling_lag = scaling_lag
        self.min_ema_multiplier = min_ema_multiplier
        

        self.initial_weights = copy.deepcopy(model.state_dict()) ## Keep theta_0
        if self.update_after == 0:
            
            self.initialized = True
        else:
            self.initialized = False
        
        self.running_model = model ## Running model that will be OU-EMA corrected


    @staticmethod
    def get_polynomial_multipliers(time_from_start, eta_power, max_weight=10.0):
        """
        returns weights for polynomially decaying bias correction.  Leads to
        theta_t' = (1 - (1 + t)**-eta_power)**-1 * theta_t - (1 - (1 + t)**-eta_power)**-1 * (1 + t)**-eta_power * theta_0
        """
        mult_1 = 1/(1 - (1 + time_from_start)**-eta_power)
        mult_2 = ((1 + time_from_start)**eta_power - 1)**-1 

        return mult_1, mult_2
    




    @torch.no_grad()
    def update(self, new_weights, step):
        """
        Updates the model with the OU-EMA correction.
        """
        do_update = self.check_if_update(step)
        if do_update:
            if not self.initialized:
                initial_weights = self.running_model.state_dict()
                for key, weight in initial_weights.items():
                    if key in new_weights.keys():
                        initial_weights[key] = new_weights[key]
                self.initial_weights = initial_weights
                self.initialized = True

            if self.do_ou:
                ## Getting unbiased estimate
                ouema_time_from_start = self.get_stabilizer_time(step)
                mult_1, mult_2 = self.get_polynomial_multipliers(ouema_time_from_start, self.eta_power, self.max_weight)
                mult_2 = -mult_2


                for key, new_weight in new_weights.items():
                    new_weights[key] = self._weight_update(out_tensor=new_weights[key], first_tensor=new_weight, mult1=mult_1, second_tensor=self.initial_weights[key], mult2=mult_2)

            if self.do_ema:
                ## EMA Update

                

                self.ema_update(self.running_model, new_weights, step)
            else:
                ## Standard Update
                self.vanilla_update(self.running_model, new_weights)         
        else:
            self.vanilla_update(self.running_model, new_weights)
            





class BEMA(BaseStabilizer):

    def __init__(
        self,
        model,
        ema_power=0.5,
        update_after=0,
        eta_power=0.2,
        ema_gamma=1.0,
        scaling_lag=0,
        min_ema_multiplier=0.0,
        ema_update_after=None,
        device='cuda',
    ):
        """
        Class for implementing BEMA correction to stabilize training post-hoc on saved checkpoints.  The estimate satisfies
        
        mu^_t = (theta_t - theta_0) / t + 1/t \int_0^t theta_s ds
        
        We implement this as a running average of the weights plus a correction term, i.e.,

        mu^t = (theta_t - theta_0) / (1 + t)^eta_power + theta_{EMA, t}
        
        where theta_{EMA,t+1} = beta_t * theta_{t} + (1 - beta_t) * theta_{EMA,t} and beta_t = gamma * (1 + t)^-ema_power

        Args:
            model: transformers.PreTrainedModel, the model to be stabilized
            ema_power: float, the power to which the time is raised for EMA (default 0.5)
            update_after: int, the number of steps after which to start the OU-EMA correction (default 0)
            eta_power: float, the power to which the time is raised for debiasing (default 0.2)
            ema_gamma: float, the gamma parameter for EMA
            scaling_lag: int, the lag for the OU term (default 0)
            min_ema_multiplier: float, the minimum value of the EMA multiplier (default 0.0)
            ema_update_after: int, the number of steps after which to start the EMA correction if None then is set to the OU update after (default None)
            device: str, the device to use (default 'cuda')
        """
        super().__init__(device)
        self.ema_power = ema_power
        self.ema_gamma = ema_gamma
        if self.ema_power >= 1 or self.ema_power <= 0 or self.ema_power is None:
            self.do_ema = False
            print("\nNot using EMA\n")
        else:
            self.do_ema = True
            print("\nUsing EMA\n")

        self.update_after = update_after
        if ema_update_after is None:
            self.ema_update_after = update_after
        else:
            self.ema_update_after = ema_update_after

        self.eta_power = eta_power
        if self.eta_power < 0 or self.eta_power is None:
            self.do_bema = False
            print("\nNot using BEMA\n")
        else:
            self.do_bema = True
            print("\nUsing BEMA\n")



        self.scaling_lag = scaling_lag
        self.min_ema_multiplier = min_ema_multiplier

        self.initial_weights = copy.deepcopy(model.state_dict()) ## Keep theta_0
        if self.update_after == 0:
            self.initialized = True
        else:
            self.initialized = False
        
        self.running_model = model ## Running model that will be BEMA corrected
        self.ema_model = copy.deepcopy(model) ## EMA model for BEMA correction
    

    @staticmethod
    def get_bema_correction_weights(time, eta_power):
        """
        Returns the weights for the BEMA correction.  Returns weight beta such that we update mu^t = (theta_t - theta_0) / (1 + t)^eta_power + theta_{EMA, t}
        """
        mult_1 = (1 + time)**-eta_power
        mult_2 = mult_1
        return mult_1, mult_2




    @torch.no_grad()
    def update(self, new_weights, step):
        """
        Updates the model with the BEMA correction.
        """
        do_update = self.check_if_update(step)

        if do_update:
            if not self.initialized:
                initial_weights = copy.deepcopy(self.running_model.state_dict())
                for key, weight in initial_weights.items():
                    if key in new_weights.keys():
                        initial_weights[key] = new_weights[key]
                self.initial_weights = initial_weights
                self.ema_model = copy.deepcopy(self.running_model)
                self.initialized = True

            if self.do_ema: ## Do EMA first on ema running model
                self.ema_update(self.ema_model, new_weights, step)
            else:
                self.vanilla_update(self.ema_model, new_weights)

            if self.do_bema: ## Do BEMA correction on running model
                
                bema_time_from_start = self.get_stabilizer_time(step)
                mult_1, mult_2 = self.get_bema_correction_weights(bema_time_from_start, self.eta_power)
                mult_2 = -mult_2 ## Subtracting theta_0, i.e. get (theta_t - theta_0) / (1 + t)^eta_power

                for key, new_weight in new_weights.items():
                    new_weights[key] = self._weight_update(out_tensor=new_weights[key], first_tensor=new_weight, mult1=mult_1, second_tensor=self.initial_weights[key], mult2=mult_2)

                running_state_dict = self.running_model.state_dict()
                ema_state_dict = self.ema_model.state_dict()
                for key, running_weight in running_state_dict.items():
                    if key in new_weights.keys(): ## get new_weight + ema_model, yielding (theta_t - theta_0) /t + theta_{EMA, t}
                        running_state_dict[key] = self._weight_update(out_tensor=running_state_dict[key], first_tensor=new_weights[key], mult1=1.0, second_tensor=ema_state_dict[key], mult2=1.0)

                self.running_model.load_state_dict(running_state_dict)
            else:
                self.vanilla_update(self.running_model, self.ema_model.state_dict())

        else:
            self.vanilla_update(self.running_model, new_weights)




class DEMA(BaseStabilizer):
    
    def __init__(self, model, ema_power, update_after, ema_gamma=1.0, scaling_lag=10, min_ema_multiplier=0.0, ema_update_after=None, device='cpu'):
        """
        Class for implementing Double EMA correction to stabilize training post-hoc on saved checkpoints.  The DEMA estimate satisfies

        theta_{DEMA,t+1} = 2 * theta_{EMA,t} - EMA(theta_{EMA})_{t-1})

        Args:
            model: transformers.PreTrainedModel, the model to be stabilized
            ema_power: float, the power to which the time is raised for EMA
            update_after: int, the number of steps after which to start the OU-EMA correction
            ema_gamma: float, the gamma parameter for EMA
            scaling_lag: int, the lag for the OU term (default 0)
            min_ema_multiplier: float, the minimum value of the EMA multiplier (default 0.0)
            ema_update_after: int, the number of steps after which to start the EMA correction if None then is set to the OU update after (default None)
            device: str, the device to use (default 'cpu')
        """
        super().__init__(device)
        self.ema_power = ema_power
        self.ema_gamma = ema_gamma
        if self.ema_power >= 1 or self.ema_power <= 0 or self.ema_power is None:
            raise ValueError("EMA power must be in (0, 1) for DEMA correction")
        else:
            self.do_ema = True
            print("\nUsing EMA\n")

        self.update_after = update_after
        if ema_update_after is None:
            self.ema_update_after = update_after
        else:
            self.ema_update_after = ema_update_after

        self.scaling_lag = scaling_lag
        self.min_ema_multiplier = min_ema_multiplier

        
        
        self.initialized = False
        
        self.running_model = model ## Running model that will be DEMA corrected
        self.single_ema_model = copy.deepcopy(model) ## Intermediate model for single EMA correction
        self.double_ema_model = copy.deepcopy(model) ## Intermediate model for double EMA correction




    @torch.no_grad()
    def update(self, new_weights, step):
        """
        Updates the model with the DEMA correction.
        """
        do_update = self.check_if_update(step)

        if do_update:
            if not self.initialized:
                self.single_ema_model = copy.deepcopy(self.running_model)
                self.double_ema_model = copy.deepcopy(self.running_model)
                self.initialized = True


            self.ema_update(self.single_ema_model, new_weights, step) # Update Single EMA model
            self.ema_update(self.double_ema_model, self.single_ema_model.state_dict(), step) # Update Double EMA model

            mult1, mult2 = 2.0, -1.0

            single_ema_weights = self.single_ema_model.state_dict()
            double_ema_weights = self.double_ema_model.state_dict()
            for key, new_weight in new_weights.items():
                new_weights[key] = self._weight_update(out_tensor=new_weights[key], first_tensor=single_ema_weights[key], mult1=mult1, second_tensor=double_ema_weights[key], mult2=mult2)

            self.vanilla_update(self.running_model, new_weights) # Update running model with DEMA corrected weights
        else:
            self.vanilla_update(self.running_model, new_weights) # Change model weights without DEMA correction