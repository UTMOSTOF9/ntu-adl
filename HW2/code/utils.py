
import nltk
import numpy as np
from filelock import FileLock
from transformers.utils import is_offline_mode
from tw_rouge import get_rouge

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def preprocess_function(examples, tokenizer, data_args):
    text_column = data_args.text_column
    summary_column = data_args.summary_column
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    padding = "max_length" if data_args.pad_to_max_length else False

    inputs, targets = [], []
    for i in range(len(examples[text_column])):
        if examples[text_column][i] and examples[summary_column][i]:
            inputs.append(examples[text_column][i])
            targets.append(examples[summary_column][i])

    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(
        inputs,
        max_length=data_args.max_source_length,
        padding=padding,
        truncation=True,
    )

    labels = tokenizer(
        text_target=targets,
        max_length=data_args.max_target_length,
        padding=padding,
        truncation=True,
    )

    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [
                (l if l != tokenizer.pad_token_id else -100) for l in label
            ]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds, tokenizer, data_args):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(
        decoded_preds, decoded_labels)
    result = get_rouge(decoded_preds, decoded_labels)
    # result = rouge.get_scores(
    #     hyps=decoded_preds, refs=decoded_labels, avg=True)
    result = {k: round(v['f'] * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)

    # calculate linear combined rouge scores
    baseline = {
        'rouge-1': 22.0,
        'rouge-2': 8.5,
        'rouge-l': 20.5,
    }
    combined_rouge = 0
    for k, v in result.items():
        if k in baseline:
            combined_rouge += v / baseline[k]

    result['rouge_combined'] = combined_rouge
    print(result)
    return result
