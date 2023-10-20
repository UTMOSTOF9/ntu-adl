import json
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import transformers
import yaml
from datasets import load_dataset
from easydict import EasyDict as edict
from fire import Fire
from src.dataset import DataCollatorForMultipleChoice, preprocess_function
from transformers import (AutoConfig, AutoModelForMultipleChoice,
                          AutoTokenizer, Trainer, TrainingArguments,
                          default_data_collator)

transformers.logging.set_verbosity_error()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
    return config


def main(config_path, do_train: bool = False, do_test: bool = False, ckpt_path: Optional[str] = None):

    config = load_config(config_path)

    # Prepare model
    model_config = AutoConfig.from_pretrained(config.model.model_name_or_path)

    if config.model.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.tokenizer_name,
            cache_dir=config.cache_dir,
            use_fast=config.model.use_fast_tokenizer
        )
    elif config.model.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_name_or_path,
            cache_dir=config.cache_dir,
            use_fast=config.model.use_fast_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    model = AutoModelForMultipleChoice.from_pretrained(
        config.model.model_name_or_path,
        from_tf=bool(".ckpt" in config.model.model_name_or_path),
        config=model_config,
    )

    # model.resize_token_embeddings(len(tokenizer))

    # prepare datasets
    data_files = {
        "train": config.data.train_file,
        "validation": config.data.validation_file,
    }

    raw_datasets = load_dataset(
        'json',
        data_files=data_files,
        cache_dir=config.cache_dir,
    )

    with open(config.data.context_file, 'r') as f:
        context_json = json.load(f)

    train_dataset = raw_datasets["train"]
    valid_dataset = raw_datasets["validation"]

    # train dataset
    partial_preprocess_function = partial(
        preprocess_function,
        context_json=context_json,
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        pad_to_max_length=config.data.pad_to_max_length,
    )
    train_dataset = train_dataset.map(
        partial_preprocess_function,
        batched=True,
        num_proc=config.data.preprocessing_num_workers,
        load_from_cache_file=not config.data.overwrite_cache,
    )
    valid_dataset = valid_dataset.map(
        partial_preprocess_function,
        batched=True,
        num_proc=config.data.preprocessing_num_workers,
        load_from_cache_file=not config.data.overwrite_cache,
    )

    # test dataset
    test_dataset = load_dataset(
        'json',
        data_files={"test": config.data.test_file},
        cache_dir=config.cache_dir,
    )['test']
    test_dataset = test_dataset.map(
        partial_preprocess_function,
        batched=True,
        num_proc=config.data.preprocessing_num_workers,
        load_from_cache_file=not config.data.overwrite_cache,
    )

    # Data Collater:
    if config.data.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForMultipleChoice(
            tokenizer,
            pad_to_multiple_of=8 if config.trainer.get('fp16', False) else None
        )

    # Metrics
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

    if ckpt_path is not None:
        config.trainer.resume_from_checkpoint = ckpt_path

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**config.trainer),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if do_train:
        print("*** Train ***")
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Evaluation
        print("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Testing
    if do_test:
        results = trainer.predict(test_dataset)
        predictions = np.argmax(results.predictions, axis=1)
        output_json = []
        for i, prediction in enumerate(predictions):
            element = {
                'id': test_dataset['id'][i],
                'question': test_dataset['question'][i],
                'paragraphs': test_dataset['paragraphs'][i],
                'relevant': test_dataset['paragraphs'][i][prediction]
            }
            output_json.append(element)

        with open(Path(config.trainer.output_dir, "test_results.json"), 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    Fire(main)
