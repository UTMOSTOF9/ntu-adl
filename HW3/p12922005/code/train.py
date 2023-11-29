import os
from pprint import pprint
from typing import Any, Dict

import torch
from datasets import load_dataset
from fire import Fire
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from utils import get_bnb_config, get_prompt, get_prompt_for_training


def format_instruction(sample):
    return {
        'text': get_prompt_for_training(sample),
        'prompt': get_prompt(sample['instruction']),
    }


os.environ["WANDB_PROJECT"] = "adl_hw3"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_num_threads(1)


def main(output_dir="results/Taiwan-LLM-7B-v2.0-chat", data_folder='data'):
    use_flash_attention = True
    model_id = "ckpt/Taiwan-LLM-7B-v2.0-chat"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=get_bnb_config(),
        use_cache=False,
        use_flash_attention_2=use_flash_attention,
        device_map=0,
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # prepare model for training
    model = prepare_model_for_kbit_training(model)

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # prepare training arguments
    train_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8 if use_flash_attention else 4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        dataloader_drop_last=True,
        learning_rate=2e-4,
        max_grad_norm=1,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=False,  # disable tqdm since with packing values are in correct
        dataloader_num_workers=4,
        seed=8888,
        data_seed=8888,
        report_to="wandb",
        remove_unused_columns=False,
    )

    pprint(train_args)

    # Upcast layer for flash attnetion
    if use_flash_attention:
        from code.llama_patch import upcast_layer_for_flash_attention
        torch_dtype = torch.bfloat16 if train_args.bf16 else torch.float16 if train_args.fp16 else torch.float32
        model = upcast_layer_for_flash_attention(model, torch_dtype)

    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    max_seq_length = 1024  # max sequence length for model and packing of the dataset

    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{data_folder}/train.json",
            "valid": f"{data_folder}/public_test.json",
        },
    )
    train_dataset = dataset['train'].map(format_instruction)
    eval_dataset = dataset['valid'].map(format_instruction)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        args=train_args,
        tokenizer=tokenizer,
        dataset_text_field='text',
    )
    # train
    trainer.train()  # there will not be a progress bar since tqdm is disabled

    # save model
    trainer.save_model()


if __name__ == "__main__":
    Fire(main)
