import random
import sys

import torch
from transformers import BitsAndBytesConfig

sys.path.append('code')
if 1:
    from few_shots import few_shots


def is_old_to_new(text):
    return '翻譯成文言文' in text or '古代怎麼說' in text


old2new_few_shots = [x for x in few_shots if is_old_to_new(x['instruction'])]
new2old_few_shots = [x for x in few_shots if not is_old_to_new(x['instruction'])]


def get_prompt(instruction: str, num_few_shot: int = 2, prompt_mode='zero_shot') -> str:
    '''Format the instruction as a prompt for LLM.'''
    if prompt_mode == 'zero_shot':
        return get_prompt_zero_shot(instruction)
    elif prompt_mode == 'few_shot':
        return get_prompt_few_shot(instruction, num_few_shot)
    else:
        raise ValueError(f'Unknown prompt mode: {prompt_mode}')


def get_prompt_zero_shot(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"


def get_prompt_few_shot(instruction: str, num_few_shot: int) -> str:
    '''Format the instruction as a prompt for LLM.'''
    samples = old2new_few_shots if is_old_to_new(instruction) else new2old_few_shots
    samples = random.choices(samples, k=num_few_shot)
    samples = ''.join([f"USER: {x['instruction']} ASSISTANT: {x['output']}\n" for x in samples])
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。{samples} USER: {instruction} ASSISTANT:"


def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
    )


def get_prompt_for_training(sample) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {sample['instruction']} ASSISTANT:{sample['output']}"
