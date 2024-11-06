import json
import os
from functools import partial
from pprint import pprint

import numpy as np
import torch
from datasets import load_dataset
from fire import Fire
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from ppl import perplexity
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, SchedulerType)
from transformers.integrations import TensorBoardCallback
from trl import SFTConfig, SFTTrainer
from utils import get_bnb_config, get_prompt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_num_threads(1)


class PPLCallback(TensorBoardCallback):
    def __init__(self, eval_data, **kwargs):
        super().__init__(**kwargs)
        self.eval_data = eval_data

    def on_evaluate(self, args, state, control, model, tokenizer, logs=None, **kwargs):
        if state.is_world_process_zero:
            with open(self.eval_data, "r") as f:
                data = json.load(f)
            model.eval()
            ppl = perplexity(model, tokenizer, data, prompt_mode='training')
            print("Mean perplexity:", ppl["mean_perplexity"])
            self.tb_writer.add_scalar(
                tag="perplexity",
                scalar_value=ppl["mean_perplexity"],
                global_step=state.global_step
            )


def main(
    model_id: str,
    output_dir: str,
    data_folder: str = 'data',
    prompt_mode: str = 'training',
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-6,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=get_bnb_config(),
        attn_implementation="flash_attention_2",
        device_map=0,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tgt_tokenizer = AutoTokenizer.from_pretrained(model_id)
    tgt_tokenizer.pad_token = tgt_tokenizer.eos_token
    tgt_tokenizer.padding_side = "right"
    # prepare model for training
    model = prepare_model_for_kbit_training(model)
    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=32,
        bias="none",
        task_type="CAUSAL_LM",
    )

    max_seq_length = 1024  # max sequence length for model and packing of the dataset
    # prepare training arguments
    train_args = SFTConfig(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        dataloader_drop_last=True,
        optim="paged_adamw_32bit",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=2,
        warmup_ratio=0.1,
        lr_scheduler_type="constant_with_warmup",
        disable_tqdm=False,  # disable tqdm since with packing values are in correct
        dataloader_num_workers=8,
        seed=8888,
        data_seed=8888,
        report_to="tensorboard",
        remove_unused_columns=['id'],
        max_seq_length=max_seq_length,
        dataset_text_field='instruction',
        bf16=True,
        tf32=True,
        eval_accumulation_steps=1,
        eval_do_concat_batches=False,
    )
    pprint(train_args)

    from llama_patch import upcast_layer_for_flash_attention
    torch_dtype = torch.bfloat16 if train_args.fp16 else torch.float16 if train_args.fp16 else torch.float32
    model = upcast_layer_for_flash_attention(model, torch_dtype)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{data_folder}/train.json",
            "valid": f"{data_folder}/public_test.json",
        },
    )

    def preprocessing(sample):
        return {
            "instruction": get_prompt(sample, prompt_mode=prompt_mode),
            "labels": sample["output"],
        }
    dataset = dataset.map(preprocessing)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        args=train_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
        callbacks=[PPLCallback(eval_data=f"{data_folder}/public_test.json")],
    )
    # train
    trainer.train()  # there will not be a progress bar since tqdm is disabled

    # save model
    trainer.save_model()


if __name__ == "__main__":
    Fire(main)
