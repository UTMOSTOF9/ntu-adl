import logging
import os
from functools import partial

import pandas as pd
import torch
from datasets import load_dataset
from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, GenerationConfig,
                          HfArgumentParser, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, set_seed)

if 1:
    import sys
    sys.path.append('.')

from arguments import DataArguments, ModelArguments
from utils import compute_metrics, preprocess_function

torch.set_num_threads(1)

logger = logging.getLogger(__name__)


def evalation(trainer, dataset, generate_config, data_args):
    metrics = trainer.evaluate(
        dataset,
        metric_key_prefix="eval",
        generation_config=generate_config,
    )
    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(dataset))
    print(metrics)
    return metrics


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed
    set_seed(training_args.seed)

    # Get the datasets
    data_files = {"validation": str(data_args.valid_file)}
    extension = data_args.valid_file.split('.')[-1]

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=str(model_args.cache_dir),
    )

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name is not None else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name is not None else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )

    # Preprocessing the datasets. Need to tokenize inputs and targets.
    eval_dataset = raw_datasets["validation"]
    column_names = eval_dataset.column_names

    with training_args.main_process_first(desc="validation dataset map preprocessing"):
        eval_dataset = eval_dataset.map(
            partial(preprocess_function, tokenizer=tokenizer, data_args=data_args),
            batched=True,
            num_proc=data_args.preprocessing_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(
            compute_metrics,
            tokenizer=tokenizer, data_args=data_args
        ) if training_args.predict_with_generate else None,
    )

    # Evaluation
    if training_args.generation_max_length is not None:
        max_length = training_args.generation_max_length
    else:
        max_length = data_args.val_max_target_length

    generate_configs = {
        'GreedySearch': GenerationConfig(max_length=max_length),
        'BeamSearch=3': GenerationConfig(max_length=max_length, num_beams=3),
        'BeamSearch=5': GenerationConfig(max_length=max_length, num_beams=5),
        'BeamSearch=7': GenerationConfig(max_length=max_length, num_beams=7),
        'Top-k=25': GenerationConfig(max_length=max_length, do_sample=True, top_k=25),
        'Top-k=50': GenerationConfig(max_length=max_length, do_sample=True, top_k=50),
        'Top-k=100': GenerationConfig(max_length=max_length, do_sample=True, top_k=100),
        'Top-p=0.8': GenerationConfig(max_length=max_length, do_sample=True, top_p=0.8),
        'Top-p=0.95': GenerationConfig(max_length=max_length, do_sample=True, top_p=0.95),
        'Top-p=0.95,T=0.5': GenerationConfig(max_length=max_length, do_sample=True, top_p=0.95, temperature=0.5),
        'Top-p=0.95,T=2.0': GenerationConfig(max_length=max_length, do_sample=True, top_p=0.95, temperature=2.0),
    }
    metrics = {
        name: evalation(trainer, eval_dataset, config, data_args)
        for name, config in generate_configs.items()
    }
    output_fpath = os.path.join(training_args.output_dir, 'eval_results.csv')
    pd.DataFrame.from_dict(metrics).T.to_csv(output_fpath)
    return metrics


if __name__ == "__main__":
    results = main()
