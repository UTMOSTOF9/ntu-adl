from functools import partial

import numpy as np
from datasets import load_dataset
from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, GenerationConfig,
                          HfArgumentParser, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

if 1:
    import sys
    sys.path.append('.')

import jsonlines
from src.arguments import DataArguments, ModelArguments


def preprocess_function_for_test(examples, tokenizer, data_args):
    text_column = data_args.text_column
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    padding = "max_length" if data_args.pad_to_max_length else False

    inputs = []
    for i in range(len(examples[text_column])):
        if examples[text_column][i]:
            inputs.append(examples[text_column][i])

    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

    return model_inputs


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Get the datasets
    data_files = {"test": str(data_args.test_file)}
    extension = data_args.test_file.split('.')[-1]
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
    predict_dataset = raw_datasets["test"]
    column_names = predict_dataset.column_names

    with training_args.main_process_first(desc="prediction dataset map preprocessing"):
        predict_dataset = predict_dataset.map(
            partial(preprocess_function_for_test, tokenizer=tokenizer, data_args=data_args),
            batched=True,
            num_proc=data_args.preprocessing_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if training_args.generation_max_length:
        max_length = training_args.generation_max_length
    else:
        max_length = data_args.val_max_target_length

    predict_results = trainer.predict(
        predict_dataset,
        metric_key_prefix="predict",
        max_length=max_length,
        generation_config=GenerationConfig(max_length=max_length, num_beams=5),
    )

    if training_args.predict_with_generate:
        preds = predict_results.predictions
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(
            preds,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        preds = [pred.strip() for pred in preds]
        ids = raw_datasets['test']['id']
        outputs = [
            {'id': i, 'title': pred}
            for i, pred in zip(ids, preds)
        ]
        with jsonlines.open(data_args.pred_output_file, 'w') as writer:
            writer.write_all(outputs)


if __name__ == "__main__":
    main()
