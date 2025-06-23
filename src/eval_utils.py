import numpy as np
import os
import torch
from omegaconf import OmegaConf
import torch.distributed.tensor
from vllm import LLM, SamplingParams
import transformers
from vllm.utils import get_ip, get_open_port
import datasets

import torch.nn.functional as F



############### Loss Computation Utils #############




def compute_loss_and_accuracy(model, batch, tokenizer, device='cuda'):
    """
    Computes log loss and mean token accuracy for a batch efficiently on GPU with minimal memory.

    Args:
        model: Pretrained transformer model.
        batch: Dictionary containing 'input_ids' and 'attention_mask'.
        tokenizer: Tokenizer object to get pad token id.
        device: Compute device (default is 'cuda').

    Returns:
        loss (float): Cross-entropy loss.
        accuracy (float): Mean token accuracy.
    """
    pad_token_id = tokenizer.pad_token_id  

    with torch.no_grad():  
        # Move batch to GPU
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        
        # Forward pass - use half precision if OOM persists
        with torch.amp.autocast('cuda'):  
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        # Shift for loss calculation
        shift_logits = logits[:, :-1, :]  # (batch, seq-1, vocab)
        shift_labels = input_ids[:, 1:]  # (batch, seq-1)
        valid_mask = shift_labels != pad_token_id  # Exclude pad tokens

        # Compute loss in a memory-efficient way (batch-wise processing)
        loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), 
                               shift_labels.reshape(-1), 
                               ignore_index=pad_token_id, 
                               reduction='sum') / valid_mask.sum()  # Normalize over valid tokens

        # Compute accuracy with minimal memory footprint
        predictions = shift_logits.argmax(dim=-1)  
        correct = (predictions == shift_labels) & valid_mask  
        accuracy = correct.sum().float() / valid_mask.sum().float()

        # Free up memory
        del logits, shift_logits, shift_labels, correct, predictions
        torch.cuda.empty_cache()

    return loss.item(), accuracy.item()





def eval_model_loss(model, dataloaders, tokenizer, device='cuda', do_eval=True):
    """
    Given a model and a dataloader, evaluates the model on the dataloader
    Args:
        model: transformers.PreTrainedModel, the model to evaluate
        dataloaders: dict str:torch.utils.data.Dataloader, dictionary of the dataloaders to evaluate on
        tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use
        device: str, the device to use (default 'cuda')
        do_eval: bool, whether to evaluate the model (default True)
    """
    if do_eval:
        model.eval()
        old_device = model.device

        if old_device.type == 'cpu' and device == 'cuda':
            model.cuda()
            changed_device = True
        else:
            changed_device = False

        outs = {}
        for split, dataloader in dataloaders.items():
            losses = []
            correct_tokens = []
            total_tokens = []

            with torch.no_grad():
                for batch in dataloader:

                    loss, accuracy = compute_loss_and_accuracy(model, batch, tokenizer, device=device)
                    losses.append(loss)
                    correct_tokens.append(accuracy)
                    total_tokens.append(batch['attention_mask'].sum().item())

            
            avg_loss = sum(losses) / len(losses)
            avg_accuracy = sum(correct_tokens) / sum(total_tokens)
            if split == 'eval':
                key = 'eval/loss'
            else:
                key = f'eval/{split}_loss'
            outs[key] = avg_loss
            
        if changed_device:
            model.cpu()
        return outs
    else:
        outs = {}
        return outs



############# Updating vLLM weights #############

def stateless_init_process_group(master_address, master_port, rank, world_size,
                                 device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes) 
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(host=master_address,
                                      port=master_port,
                                      rank=rank,
                                      world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl

class WorkerExtension:
    """
    Adapted from https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_utils.py#L24C7-L24C22
    The class for vLLM's worker to inherit from.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """
    def init_weight_update_group(self, master_address, master_port,
                                 rank_offset, world_size):
        from vllm.distributed.parallel_state import get_world_group
        rank = get_world_group().rank + rank_offset
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

    def update_weight(self, name, weight, device='cuda'):
        weight = weight.to(device)
        self.model_update_group.broadcast(weight,
                                          src=0,
                                          stream=torch.cuda.current_stream())

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
    
    def update_weights(self, weights, device='cuda'):
        weights = {k: v.to(device) for k, v in weights.items()}

        self.model_update_group.broadcast(weights,
                                        src=0,
                                        stream=torch.cuda.current_stream())
        self.model_runner.model.load_weights(weights=list(weights.items()))
        del weights        




class WorkerExtensionSingleGPU:
    def update_weights(self, weights, device='cuda'):

        weights = {k: torch.tensor(v).to(device) for k, v in weights.items()}
        self.model_runner.model.load_weights(weights=list(weights.items()))
        del weights




def make_updateable_vllm(path,rank=0, world_size=1, gpu_memory_utilization=0.9, **kwargs):
    """
    Creates a vLLM model from a path.  The model is updateable in that we can update the weights with new weights
    """
    llm = LLM(path, worker_extension_cls='eval_generations_utils.WorkerExtensionSingleGPU', gpu_memory_utilization=gpu_memory_utilization, **kwargs)

    return llm




def update_vllm_weights(llm, hf_model, device='cuda'):
    """
    Updates the weights of the vLLM model with the given weights
    Args:
        llm: vllm.LLM, the model to update
        hf_model: transformers.AutoModelForCausalLM, the model to get the weights from
        device: str, the device to use (default 'cuda')
    """

    weights = {}
    for k, v in hf_model.named_parameters():
        if type(v) == torch.distributed.tensor.DTensor:
            # Redistribute to replicated and convert to NumPy
            device_mesh, placements = v.device_mesh, v.placements
            local_tensor = v.to_local()
            weights[k] = local_tensor.cpu().contiguous().numpy()
            v = v.redistribute(device_mesh, placements)
            
        else:
            weights[k] = v.detach().cpu().numpy()

    llm.llm_engine.collective_rpc('update_weights', args=(weights, device))




############# Generations Utils #############



def get_oai_chat_format(instruction, prefixes=None):
    """
    Given an instruction, returns the chat format for the instruction
    Args:
        instruction: str, the instruction to format
        prefixes: list of str, the prefixes to use for the instruction, e.g. multi-shot examples
    """
    if prefixes is None:
        if type(instruction) == str:
            return [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": instruction
                }
            ]
        elif type(instruction) == list:
            return [[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": current_instruction
                }
            ] for current_instruction in instruction]
    else:
        if type(instruction) == str:
            system_prompt = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                }
            ]
            instruction = [
                {
                    "role": "user",
                    "content": instruction
                }
            ]
            return system_prompt + prefixes + instruction
        elif type(instruction) == list:
            system_prompt = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                }
            ]
            return [
                [system_prompt + prefixes + [{
                    "role": "user",
                    "content": current_instruction
                }] for current_instruction in instruction]
            ]
            


def get_vllm_response_texts(response):
    return [output.text for output in response.outputs]



class BaseGenerationsEvaluator:

    def __init__(
            self,
            alias,
            tokenizer,
            idxs,
            save_gens=True,            
    ):
        """
        Base class for all generations evaluators
        Args:
            alias: str, the alias for the evaluator
            tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use to apply chat format
        """
        self.alias = alias
        self.tokenizer = tokenizer
        self.idxs = idxs
        self.save_gens = save_gens
        print(f"Using {self.alias} evaluator (save_gens={self.save_gens})")
    


    def get_prompts(self):
        """
        Returns the prompts for the evaluator
        """
        raise NotImplementedError("Subclass must implement get_prompts method")
    
    def get_rule(self):
        """
        Returns the rule for the evaluator
        """
        raise NotImplementedError("Subclass must implement get_rule method")
    
    def check_responses(self, responses):
        """
        Given the responses, returns the accuracy and standard error of the responses.  Returns list of strings corresponding to the responses
        Args:
            responses: list of str, the responses to check
        """
        raise NotImplementedError("Subclass must implement check_answers method")
    

    def save_responses(self, responses, outputs_dir, fname):
        """
        Given the responses, saves the responses to the outputs directory
        Args:
            responses: list of str, the responses to save where the list corresponds to the responses per prompt
            outputs_dir: str, the directory to save the responses to
            fname: str, the filename to save the responses to
        """
        if not self.save_gens:
            return
        sep_token = '<<<|||sep|||>>>'
        output_parent = os.path.join(outputs_dir, self.alias)
        os.makedirs(output_parent, exist_ok=True)
        with open(os.path.join(output_parent, f'{fname}.txt'), 'w') as f:
            f.write(sep_token.join(responses))



    @staticmethod
    def _get_response_vllm(string):
        """
        Given the output of a language model SFT'd on Tulu, returns the response
        """

        end_str = '<|im_end|>'
        if end_str in string:
            response_end = string.index(end_str)
            return string[:response_end]
        end_str = '\nuser\n'
        if end_str in string:
            response_end = string.index(end_str)
            return string[:response_end]
        else:
            return string
    

    @torch.no_grad()
    @staticmethod
    def _get_generations_vllm(model, tokenizer, prompts, device='cuda', prefixes=None, **generation_kwargs):
        """
        Uses vllm to generate responses for the given prompts
        Args:
            model: vllm.LLM, the model to use
            tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use
            prompts: list of str, the prompts to generate responses for
            device: str, the device to use (default 'cuda')
            generation_kwargs: dict, the kwargs to pass to vllm.SamplingParams
        """
        assert type(model) == LLM

        chat_formatted = [get_oai_chat_format(prompt, prefixes=prefixes) for prompt in prompts]
        chat_formatted = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in chat_formatted]
        sampling_params = SamplingParams(**generation_kwargs)
        model_outputs = model.generate(chat_formatted, sampling_params=sampling_params)
        responses = [get_vllm_response_texts(output) for output in model_outputs]
        return responses



    def get_generations_base(
            self,
            prompts,
            model,
            sampling_kwargs,
            apply_chat_format=True,
            use_vllm=True,
            prefixes=None
        ):
        """
        Uses a model to generate responses for the given prompts
        Args:
            prompts: list of str, the prompts to generate responses for
            model: vllm.LLM, the model to use
            sampling_kwargs: dict, the kwargs to pass to the model's generate method
            apply_chat_format: bool, whether to apply chat format to the prompts (default True)
            use_vllm: bool, whether to use vllm for generation.  If not, then uses HF (default True)
            prefixes: list of str, the prefixes to use for the instruction, e.g. multi-shot examples.  Defaults to None, no multi-shot examples
        """
        if use_vllm:
            responses = BaseGenerationsEvaluator._get_generations_vllm(model, self.tokenizer, prompts, device='cuda', prefixes=prefixes, **sampling_kwargs)
        else:
            raise NotImplementedError("HF generation not implemented.")
        return responses
    




class GSM8KEvaluator(BaseGenerationsEvaluator):

    def __init__(
            self,
            tokenizer,
            num_examples,
            sampling_args,
            use_vllm=True,
            selection_seed=37771,
            use_cot=False,
            num_shots=0,
            flexible_match = True,
            **kwargs
    ):
        """
        Class for evaluating generations using the GSM8K dataset
        Args:
            tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use
            num_examples: int, number of examples to use for evaluation, must be less than 1319
            prompt_parent: str, the parent directory of the prompts
            sampling_args: dict, the kwargs to pass to the model's generate method
            use_vllm: bool, whether to use vllm for generation. (default True)
            selection_seed: int, the seed to use for selecting the examples
            use_cot: bool, whether to use CoT for generation (default False)
            num_shots: int, the number of multi-shot examples to use for generation (default 0)
            flexible_match: bool, whether to use flexible matching for the answer (default True)

        """
        super().__init__('GSM8K', tokenizer, None, **kwargs)

        self.use_vllm = use_vllm
        self.sampling_kwargs = sampling_args
        
        self.num_shots = num_shots
        if self.num_shots > 0:
            self.use_cot = False
        else:
            self.use_cot = use_cot
        
        self.flexible_match = flexible_match
        
        self.get_prompts_and_rules(num_examples, selection_seed)



    @staticmethod
    def _format_prompt(row):
        """
        Given a row of an MMLU dataset, formats the prompt.  Formatting comes from eleuther's lm-evaluation-harness by adding CoT to https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_default_template_yaml#L7C14-L7C121
        Args:
            row: dict, the row of the dataset to format
        """
        question = row['question']
        ADDED_PROMPT = "Think step by step and write your final answer at the end of your response, separated by '####' from the rest of your response.  For example, if your final response is '42', then your response should end in '#### 42'."
        prompt = ADDED_PROMPT + '\nQuestion: ' + question.strip()
        
        return {'prompt': prompt}

    @staticmethod
    def _format_multishot_example(prompt, answer):
        """
        Given a row of an MMLU dataset, formats the prompt and anaswer.
        """
        answer_text = f"#### {answer}"
        return [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': answer_text}
        ]

    @staticmethod
    def _check_gsm8k_answer(llm_output, answer, flexible_match=True):
        """
        Given the output of a language model, checks if the answer is correct
        """
        numerical_answer = answer.split('####')[-1].strip()
        if flexible_match:
            return numerical_answer in llm_output
        else:
            llm_output = llm_output.split('####')[-1].strip()
            if numerical_answer == llm_output:
                return True
            else:
                return False

    def get_prompts_and_rules(self, num_examples, selection_seed):
        """
        Adds prompts and rules to the class for generation and checking
        """

        data = datasets.load_dataset('openai/gsm8k', name='main', split='test')
        if num_examples > len(data):
                print(f"Warning: num_examples {num_examples} is greater than the number of examples in the dataset {len(data)}.  Using all examples.")
                num_examples = len(data) - self.num_shots
                idxs = list(range(num_examples))
        else:
            if self.num_shots == 0:
                idxs = sorted(list(np.random.RandomState(selection_seed).choice(len(data), num_examples, replace=False)))
                self.prefixes = None
            else:
                idxs = list(np.random.RandomState(selection_seed).choice(len(data), num_examples + self.num_shots, replace=False))
                examples = idxs[num_examples:]
                idxs = sorted(idxs[:num_examples])
                examples = data.select(examples)
                example_prompts = [GSM8KEvaluator._format_prompt(row)['prompt'] for row in examples]
                example_answers = data.select(idxs[num_examples:])['answer']
                raw_prefixes = [GSM8KEvaluator._format_multishot_example(prompt, answer) for prompt, answer in zip(example_prompts, example_answers)]
                prefixes = []
                for prefix in raw_prefixes:
                    prefixes = prefixes + prefix
                
                self.prefixes = prefixes

            data = data.select(idxs)
        
        self.idxs = [int(idx) for idx in idxs]
        

        prompts = data.map(lambda row:GSM8KEvaluator._format_prompt(row))['prompt']
        self.prompts = prompts
        answers = data['answer']
        self.answers = answers
        self.rules = []
        if self.use_vllm:
            extract_answer = BaseGenerationsEvaluator._get_response_vllm
        else:
            extract_answer = BaseGenerationsEvaluator._get_response
        
        for answer in answers:
            rule = lambda x: GSM8KEvaluator._check_gsm8k_answer(extract_answer(x), answer, flexible_match=self.flexible_match)
            self.rules.append(rule)


    def get_generations(self, prompts, model, apply_chat_format=True):
        return self.get_generations_base(
            prompts,
            model,
            self.sampling_kwargs,
            apply_chat_format=apply_chat_format,
            use_vllm=self.use_vllm,
        )


    def check_responses(self, responses):
        """
        Given responses which is a list of lists, where the outer index corresponds to self.prompts and self.rules and the inner list is a list of sampled responses, computes the score of each response
        """
        scores, stds = [], []
        for i, rule in enumerate(self.rules):
            response_scores = []
            temp_responses = responses[i]
            for response in temp_responses:
                if rule(response):
                    response_scores.append(1)
                else:
                    response_scores.append(0)
            scores.append(float(np.mean(response_scores)))
            stds.append(float(np.std(response_scores) / np.sqrt(len(response_scores))))
        return scores, stds
    


    def evaluate_model(self, model, outputs_dir, step, apply_chat_format=True):
        """
        Given a model, evaluates the model on the prompts and rules
        Args:
            model: vllm.LLM, the model to evaluate or transformers.PreTrainedModel
            sampling_kwargs: dict, the kwargs to pass to the model's generate method
            outputs_dir: str, the directory to save the responses to
            step: int, the step to save the responses at
        """
        # get generations

        responses = self.get_generations(self.prompts, model, apply_chat_format=apply_chat_format)
        # check responses
        scores, stds = self.check_responses(responses)
        if outputs_dir is not None and self.save_gens:
            # save responses
            for i, prompt in enumerate(self.prompts):
                idx = self.idxs[i]
                prompt_fname = f'prompt_{idx}'
                response_fname = f'responses_{idx}_{step}'
                prompt_responses = responses[i]
                self.save_responses(prompt_responses, outputs_dir, response_fname)
                with open(os.path.join(outputs_dir, self.alias, f'{prompt_fname}.txt'), 'w') as f:
                    f.write(prompt)
        
        output_dict = {}
        for i, idx in enumerate(self.idxs):
            
            output_dict[f'eval/{self.alias}/{idx}/acc'] = scores[i]
            output_dict[f'eval/{self.alias}/{idx}/std_err'] = stds[i]
        
        ## Calculating the average score
        avg_score = np.mean(scores)
        avg_std = np.sqrt(np.mean(np.array(stds) ** 2))
        output_dict[f'eval/{self.alias}/avg/acc'] = float(avg_score)
        output_dict[f'eval/{self.alias}/avg/std_err'] = float(avg_std)

        return output_dict









class MMLUEvaluator(BaseGenerationsEvaluator):

    def __init__(
            self,
            subset,
            tokenizer,
            num_examples,
            sampling_args,
            use_vllm=True,
            selection_seed=37771,
            use_cot=False,
            num_shots=0,
            flexible_match = True,
            **kwargs
    ):
        """
        Class for evaluating generations using MMLU dataset
        Args:
            subset: str, the subset of MMLU to use, e.g. `high_school_us_history`
            tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use
            num_examples: int, number of examples to use for evaluation, must be less than 1319
            prompt_parent: str, the parent directory of the prompts
            sampling_args: dict, the kwargs to pass to the model's generate method
            use_vllm: bool, whether to use vllm for generation. (default True)
            selection_seed: int, the seed to use for selecting the examples
            use_cot: bool, whether to use CoT for generation (default False)
            num_shots: int, the number of multi-shot examples to use for generation (default 0)
            flexible_match: bool, whether to use flexible matching for the answer (default True)
        """

        alias = f'MMLU_{subset}'
        super().__init__(alias, tokenizer, None, **kwargs)

        self.subset = subset
        self.use_vllm = use_vllm
        self.sampling_kwargs = sampling_args
        
        self.num_shots = num_shots
        if self.num_shots > 0:
            self.use_cot = False
        else:
            self.use_cot = use_cot
        
        self.flexible_match = flexible_match
        self.get_prompts_and_rules(num_examples, selection_seed)
        
    
    @staticmethod
    def _format_prompt(row, use_cot=False):
        """
        Given a row of an MMLU dataset, formats the prompt.  Formatting comes from eleuther's lm-evaluation-harness by adding CoT to https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_default_template_yaml#L7C14-L7C121
        Args:
            row: dict, the row of the dataset to format
        """
        question = row['question']
        choices = row['choices']
        if use_cot:
             PREFIX = "Think step by step to answer the following multiple choice question.  "
        else:
            PREFIX = "Answer the following multiple choice question.  "
        TEMPLATE = PREFIX + f"Write your final answer at the end of your response.  For example, if your final response is 'A', then your response should end in 'Answer: A'.  \n\nQuestion: {question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
        prompt = TEMPLATE
        
        return {'prompt': prompt}

    @staticmethod
    def _format_multishot_example(prompt, answer):
        """
        Given a row of an MMLU dataset, formats the prompt and anaswer.
        """
        answer_text = f"Answer: {answer}"
        return [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': answer_text}
        ]





    @staticmethod
    def _check_mmlu_answer(extracted_answer, answer, flexible_match=True):
        """
        Given a response and an answer, checks if the response is correct
        Args:
            response: str, the response from the model
            answer: int, the answer to check against
        """
        ANSWERS = {
                0: 'A',
                1: 'B',
                2: 'C',
                3: 'D'
            }
        answer = ANSWERS[answer]
        if flexible_match:

            if extracted_answer.strip() == answer:
                return True
            elif f'Answer: {answer}'.lower() in extracted_answer.lower():
                return True
            elif 'The correct answer is {answer}'.lower() in extracted_answer.lower():
                return True
            elif f"Final answer: {answer}.".lower() in extracted_answer.lower():
                return True
            elif f"Answer: ({answer})".lower() in extracted_answer.lower():
                return True
            elif f"Option {answer}".lower() in extracted_answer.lower():
                return True
            elif f"answer is {answer}".lower() in extracted_answer.lower():
                return True
            elif f"{answer}." in extracted_answer:
                if not ('A.' in extracted_answer and 'B.' in extracted_answer and 'C.' in extracted_answer and 'D.' in extracted_answer):
                    return True
                else:
                    return False
            else:
                return False
        else:

            if extracted_answer.strip() == answer:
                return True
            elif f'Answer: {answer}' in extracted_answer:
                return True
            elif 'The correct answer is {answer}' in extracted_answer:
                return True
            elif f"{answer}." in extracted_answer:
                return True
            else:
                return False



    def get_prompts_and_rules(self, num_examples, selection_seed):
        """
        Adds prompts and rules to the class for generation and checking
        Args:
            num_examples: int, number of examples to use for evaluation
            selection_seed: int, the seed to use for selecting the examples
        """

        data = datasets.load_dataset('cais/mmlu', name=self.subset, split='test')
        if num_examples > len(data):
            print(f"Warning: num_examples {num_examples} is greater than the number of examples in the dataset {len(data)}.  Using all examples.")
            num_examples = len(data) - self.num_shots
            idxs = list(range(num_examples))
        else:
            if self.num_shots == 0:
                idxs = sorted(list(np.random.RandomState(selection_seed).choice(len(data), num_examples, replace=False)))
                self.prefixes = None
            else:
                idxs = list(np.random.RandomState(selection_seed).choice(len(data), num_examples + self.num_shots, replace=False))
                examples = idxs[num_examples:]
                idxs = sorted(idxs[:num_examples])
                examples = data.select(examples)
                example_prompts = [MMLUEvaluator._format_prompt(row, use_cot=False)['prompt'] for row in examples]
                ANSWERS = {
                0: 'A',
                1: 'B',
                2: 'C',
                3: 'D'
                }
                example_answers = [ANSWERS[answer] for answer in examples['answer']]
                raw_prefixes = [MMLUEvaluator._format_multishot_example(prompt, answer) for prompt, answer in zip(example_prompts, example_answers)]
                prefixes = []
                for prefix in raw_prefixes:
                    prefixes = prefixes + prefix
                
                self.prefixes = prefixes

            data = data.select(idxs)
        
        self.idxs = [int(idx) for idx in idxs]
        

        prompts = data.map(lambda row:MMLUEvaluator._format_prompt(row, use_cot=self.use_cot))['prompt']
        self.prompts = prompts
        answers = data['answer']
        self.answers = answers
        self.rules = []
        if self.use_vllm:
            extract_answer = BaseGenerationsEvaluator._get_response_vllm
        else:
            extract_answer = BaseGenerationsEvaluator._get_response
        
        for answer in answers:
            rule = lambda x: MMLUEvaluator._check_mmlu_answer(extract_answer(x), answer, flexible_match=self.flexible_match)
            self.rules.append(rule)
    


    def get_generations(self, prompts, model, apply_chat_format=True):
        return self.get_generations_base(
            prompts,
            model,
            self.sampling_kwargs,
            apply_chat_format=apply_chat_format,
            use_vllm=self.use_vllm,
        )
    
    def check_responses(self, responses):
        """
        Given responses which is a list of lists, where the outer index corresponds to self.prompts and self.rules and the inner list is a list of sampled responses, computes the score of each response
        """
        scores, stds = [], []
        for i, rule in enumerate(self.rules):
            response_scores = []
            temp_responses = responses[i]
            for response in temp_responses:
                if rule(response):
                    response_scores.append(1)
                else:
                    response_scores.append(0)
            scores.append(float(np.mean(response_scores)))
            stds.append(float(np.std(response_scores) / np.sqrt(len(response_scores))))
        return scores, stds
    



    def evaluate_model(self, model, outputs_dir, step, apply_chat_format=True):
        """
        Given a model, evaluates the model on the prompts and rules
        Args:
            model: vllm.LLM, the model to evaluate or transformers.PreTrainedModel
            sampling_kwargs: dict, the kwargs to pass to the model's generate method
            outputs_dir: str, the directory to save the responses to
            step: int, the step to save the responses at
        """
    
        # get generations

        responses = self.get_generations(self.prompts, model, apply_chat_format=apply_chat_format)
        # check responses
        scores, stds = self.check_responses(responses)
        if outputs_dir is not None and self.save_gens:
            # save responses
            for i, prompt in enumerate(self.prompts):
                idx = self.idxs[i]
                prompt_fname = f'prompt_{idx}'
                response_fname = f'responses_{idx}_{step}'
                prompt_responses = responses[i]
                self.save_responses(prompt_responses, outputs_dir, response_fname)
                with open(os.path.join(outputs_dir, self.alias, f'{prompt_fname}.txt'), 'w') as f:
                    f.write(prompt)
        
        output_dict = {}
        for i, idx in enumerate(self.idxs):
            
            output_dict[f'eval/{self.alias}/{idx}/acc'] = scores[i]
            output_dict[f'eval/{self.alias}/{idx}/std_err'] = stds[i]
        
        ## Calculating the average score
        avg_score = np.mean(scores)
        avg_std = np.sqrt(np.mean(np.array(stds) ** 2))
        output_dict[f'eval/{self.alias}/avg/acc'] = float(avg_score)
        output_dict[f'eval/{self.alias}/avg/std_err'] = float(avg_std)

        return output_dict
    




class BoolQEvaluator(BaseGenerationsEvaluator):

    def __init__(
            self,
            tokenizer,
            num_examples,
            sampling_args,
            use_vllm=True,
            selection_seed=37771,
            use_cot=False,
            num_shots=0,
            **kwargs
    ):
        """
        Class for evaluating generations using the BoolQ dataset.
        Args:
            tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use
            num_examples: int, number of examples to use for evaluation, must be less than 1319
            prompt_parent: str, the parent directory of the prompts
            sampling_args: dict, the kwargs to pass to the model's generate method
            use_vllm: bool, whether to use vllm for generation.  If not, then uses HF (default True)
            selection_seed: int, the seed to use for selecting the examples
            use_cot: bool, whether to use CoT for generation (default False)
            num_shots: int, the number of multi-shot examples to use for generation (default 0)
        """
        alias = 'BoolQ'
        super().__init__(alias, tokenizer, None, **kwargs)

        self.use_vllm = use_vllm
        self.sampling_kwargs = sampling_args
        self.num_shots = num_shots
        if self.num_shots > 0:
            self.use_cot = False
        else:
            self.use_cot = use_cot
        
        self.get_prompts_and_rules(num_examples, selection_seed)



    @staticmethod
    def _format_prompt(row, use_cot=False):
        """
        Given a row of the BoolQ dataset, formats the prompt.  Formatting comes from eleuther's lm-evaluation-harness https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/super_glue/boolq/default.yaml
        Args:
            row: dict, the row of the dataset to format
        """
        passage = row['passage']
        question = row['question']
        PREFIX = "Read the following passage and answer the question.  Your answer should be either 'true' or 'false'; thus if the answer is 'true', your response should end in 'Answer: true'; otherwise your response should end in 'Answer: false'." 
        prompt = PREFIX + "\n" + f"Passage: {passage}" + "\n" + f"Question: {question}?\n" 
        return {'prompt': prompt}
    
    @staticmethod
    def _format_multishot_example(prompt, answer):
        """
        Given a row of an MMLU dataset, formats the prompt and anaswer.
        """
        answer_text = f"Answer: {answer}"
        return [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': answer_text}
        ]
    

    @staticmethod
    def _check_boolq_answer(extracted_answer, answer):
        """
        Given a response and an answer, checks if the response is correct
        Args:
            response: str, the response from the model
            answer: int, the answer to check against
        """
        answer = str(answer).lower()
        if extracted_answer.strip().lower() == answer:
            return True
        elif f'Answer: {answer}'.lower() in extracted_answer.lower():
            return True
        elif 'answer is {answer}'.lower() in extracted_answer.lower():
            return True
        elif answer.lower() == 'true' and 'yes' in extracted_answer.lower():
            return True
        elif answer.lower() == 'false' and 'no' in extracted_answer.lower():
            return True
        elif answer.lower() == 'true' and 'correct' in extracted_answer.lower():
            return True
        elif answer.lower() == 'false' and 'incorrect' in extracted_answer.lower():
            return True
        else:
            return False
    

    def get_prompts_and_rules(self, num_examples, selection_seed):
        """
        Adds prompts and rules to the class for generation and checking
        Args:
            num_examples: int, number of examples to use for evaluation
            selection_seed: int, the seed to use for selecting the examples
        """

        data = datasets.load_dataset('google/boolq', split='validation')
        if num_examples > len(data):
            print(f"Warning: num_examples {num_examples} is greater than the number of examples in the dataset {len(data)}.  Using all examples.")
            num_examples = len(data) - self.num_shots
            idxs = list(range(num_examples))
        else:
            if self.num_shots == 0:
                idxs = sorted(list(np.random.RandomState(selection_seed).choice(len(data), num_examples, replace=False)))
                self.prefixes = None
            else:
                idxs = list(np.random.RandomState(selection_seed).choice(len(data), num_examples + self.num_shots, replace=False))
                examples = idxs[num_examples:]
                idxs = sorted(idxs[:num_examples])
                examples = data.select(examples)
                example_prompts = [BoolQEvaluator._format_prompt(row, use_cot=False)['prompt'] for row in examples]
                
                example_answers = [f"Answer: {answer}" for answer in examples['answer']]
                raw_prefixes = [BoolQEvaluator._format_multishot_example(prompt, answer) for prompt, answer in zip(example_prompts, example_answers)]
                prefixes = []
                for prefix in raw_prefixes:
                    prefixes = prefixes + prefix
                
                self.prefixes = prefixes

            data = data.select(idxs)
        
        self.idxs = [int(idx) for idx in idxs]
        

        prompts = data.map(lambda row:BoolQEvaluator._format_prompt(row, use_cot=self.use_cot))['prompt']
        self.prompts = prompts
        answers = data['answer']
        self.answers = [str(answer).lower() for answer in answers]
        self.rules = []
        if self.use_vllm:
            extract_answer = BaseGenerationsEvaluator._get_response_vllm
        else:
            extract_answer = BaseGenerationsEvaluator._get_response
        
        for answer in answers:
            rule = lambda x: BoolQEvaluator._check_boolq_answer(extract_answer(x), answer)
            self.rules.append(rule)


    def get_generations(self, prompts, model, apply_chat_format=True):
        return self.get_generations_base(
            prompts,
            model,
            self.sampling_kwargs,
            apply_chat_format=apply_chat_format,
            use_vllm=self.use_vllm,
        )
    
    def check_responses(self, responses):
        """
        Given responses which is a list of lists, where the outer index corresponds to self.prompts and self.rules and the inner list is a list of sampled responses, computes the score of each response
        """
        scores, stds = [], []
        for i, rule in enumerate(self.rules):
            response_scores = []
            temp_responses = responses[i]
            for response in temp_responses:
                if rule(response):
                    response_scores.append(1)
                else:
                    response_scores.append(0)
            scores.append(float(np.mean(response_scores)))
            stds.append(float(np.std(response_scores) / np.sqrt(len(response_scores))))
        return scores, stds
    
    

    def evaluate_model(self, model, outputs_dir, step, apply_chat_format=True):
        """
        Given a model, evaluates the model on the prompts and rules
        Args:
            model: vllm.LLM, the model to evaluate or transformers.PreTrainedModel
            sampling_kwargs: dict, the kwargs to pass to the model's generate method
            outputs_dir: str, the directory to save the responses to
            step: int, the step to save the responses at
        """

        # get generations
        responses = self.get_generations(self.prompts, model, apply_chat_format=apply_chat_format)
        # check responses
        scores, stds = self.check_responses(responses)
        if outputs_dir is not None and self.save_gens:
            # save responses
            for i, prompt in enumerate(self.prompts):
                idx = self.idxs[i]
                prompt_fname = f'prompt_{idx}'
                response_fname = f'responses_{idx}_{step}'
                prompt_responses = responses[i]
                self.save_responses(prompt_responses, outputs_dir, response_fname)
                with open(os.path.join(outputs_dir, self.alias, f'{prompt_fname}.txt'), 'w') as f:
                    f.write(prompt)
        
        output_dict = {}
        for i, idx in enumerate(self.idxs):
            
            output_dict[f'eval/{self.alias}/{idx}/acc'] = scores[i]
            output_dict[f'eval/{self.alias}/{idx}/std_err'] = stds[i]
        
        ## Calculating the average score
        avg_score = np.mean(scores)
        avg_std = np.sqrt(np.mean(np.array(stds) ** 2))
        output_dict[f'eval/{self.alias}/avg/acc'] = float(avg_score)
        output_dict[f'eval/{self.alias}/avg/std_err'] = float(avg_std)

        return output_dict