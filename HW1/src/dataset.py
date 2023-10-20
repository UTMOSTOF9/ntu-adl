from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union

import torch
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


def preprocess_function(
    examples: dict,
    context_json: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: Optional[int] = None,
    pad_to_max_length: bool = True,
):
    first_sentences = [[context] * 4 for context in examples['question']]

    second_sentences = [
        [context_json[idx] for idx in related_p_idx]
        for related_p_idx in examples['paragraphs']
    ]

    # Flatten out
    first_sentences = list(chain(*first_sentences))
    second_sentences = list(chain(*second_sentences))

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=max_length,
        padding="max_length" if pad_to_max_length else False,
    )

    # Unflatten and group by four
    tokenized_inputs = {
        k: [v[i:i + 4] for i in range(0, len(v), 4)]
        for k, v in tokenized_examples.items()
    }

    if 'relevant' in examples.keys():
        tokenized_inputs['label'] = [
            para.index(p_uid)
            for para, p_uid in zip(examples['paragraphs'], examples['relevant'])
        ]
    else:
        tokenized_inputs['label'] = [0] * len(examples['paragraphs'])

    return tokenized_inputs


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: dict) -> dict:
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [
                {k: v[i] for k, v in feature.items()}
                for i in range(num_choices)
            ]
            for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
