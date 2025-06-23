#!/bin/bash

# Login to the Hugging Face model hub

/home/aiscuser/.local/bin/huggingface-cli login --token $(cat .hf_token)


# wandb login $(cat .wandb_token) --host  https://microsoft-research.wandb.io