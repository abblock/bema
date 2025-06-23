import os
import datasets
import transformers



SEED = 1337
data_source = 'allenai/tulu-3-sft-mixture'
hf_username = ''
data_destination = f'{hf_username}/tulu-3-sft-mixture-split-seed-{SEED}'
split = 'train'
train_split_size = 0.99



def get_question_len(example, tokenizer):
    """
    Given an example, returns the length of the question in tokens
    Args:
        example: dict, the example to get the question length from
    """
    question = example['messages'][0]['content']
    example['question_len'] = len(tokenizer(question)['input_ids'])
    return example
def get_answer_len(example, tokenizer):
    """
    Given an example, returns the length of the answer in tokens
    Args:
        example: dict, the example to get the answer length from
    """
    answer = example['messages'][1]['content']
    example['answer_len'] = len(tokenizer(answer)['input_ids'])
    return example
def get_total_len(example, tokenizer):
    """
    Given an example, returns the total length of the example in tokens
    Args:
        example: dict, the example to get the total length from
    """
    question = example['messages'][0]['content']
    answer = example['messages'][1]['content']
    example['total_len'] = len(tokenizer(question)['input_ids']) + len(tokenizer(answer)['input_ids'])
    return example


def get_all_lens(example, tokenizer):
    """
    Given an example, returns the length of the question, answer and total length in tokens
    Args:
        example: dict, the example to get the lengths from
    """
    example = get_question_len(example, tokenizer)
    example = get_answer_len(example, tokenizer)
    example = get_total_len(example, tokenizer)
    return example







def main():

    ## Load the dataset
    dataset = datasets.load_dataset(data_source)
    if type(dataset) is datasets.DatasetDict:
        dataset = dataset[split]

    ## Split the dataset into train and test sets
    dataset = dataset.train_test_split(train_size=train_split_size, seed=SEED + 100)

    ## Filter the dataset to only include examples with a total length of less than 4096 tokens
    tokenizer = transformers.AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')
    tokenizer.pad_token_id = len(tokenizer)
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    tokenizer.padding_side = 'left'


    train = dataset['train']

    train = train.map(lambda x:get_all_lens(x, tokenizer), num_proc=4)


    train = train.filter(lambda x: x['total_len'] <= 4096)
    eval = dataset['test']
    eval = eval.map(get_all_lens, num_proc=4)
    eval = eval.filter(lambda x: x['total_len'] <= 4096)
    filtered_data = datasets.DatasetDict({
        'train': train,
        'test': eval
    })


    ## Push to Hub

    hf_token_path = "../.hf_token"
    if not os.path.exists(hf_token_path):
        raise FileNotFoundError(f"Please create a file at {hf_token_path} with your Hugging Face token.")
    with open(hf_token_path, "r") as f:
        hf_token = f.read().strip()
    login_command = f"huggingface-cli login --token {hf_token}"
    print("Logging in to Hugging Face with command:", login_command)
    os.system(login_command)

    print("Uploading dataset to Hugging Face...")
    filtered_data.push_to_hub(data_destination)


    print(f"Dataset uploaded to {data_destination}.")


if __name__ == "__main__":
    main()