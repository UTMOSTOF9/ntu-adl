import argparse
import json

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_bnb_config, get_prompt


def perplexity(
    model,
    tokenizer,
    data,
    max_length: int = 1024,
    prompt_mode: str = 'zero_shot',
    show_example: bool = False
):
    data_size = len(data)
    instructions = [get_prompt(x, prompt_mode=prompt_mode) for x in data]
    outputs = [x["output"] for x in data]

    # Tokenize data
    tokenized_instructions = tokenizer(instructions, add_special_tokens=False)
    tokenized_instructions2 = tokenizer(instructions, add_special_tokens=False)
    tokenized_outputs = tokenizer(outputs, add_special_tokens=False)
    output_masks = []

    # Format data
    for i in range(data_size):
        instruction_input_ids = [tokenizer.bos_token_id] + tokenized_instructions["input_ids"][i]
        output_input_ids = tokenized_outputs["input_ids"][i] + [tokenizer.eos_token_id]
        tokenized_instructions["input_ids"][i] = instruction_input_ids + output_input_ids
        tokenized_instructions["attention_mask"][i] = [1] * len(tokenized_instructions["input_ids"][i])

        tokenized_instructions2["input_ids"][i] = [tokenizer.bos_token_id] + tokenized_instructions2["input_ids"][i]
        output_mask = [0] * len(instruction_input_ids) + [1] * len(output_input_ids)
        tokenized_instructions["input_ids"][i] = torch.tensor(tokenized_instructions["input_ids"][i])[:max_length]
        tokenized_instructions["attention_mask"][i] = torch.tensor(
            tokenized_instructions["attention_mask"][i][:max_length]
        )
        tokenized_instructions2["input_ids"][i] = torch.tensor(tokenized_instructions2["input_ids"][i])[:max_length]
        output_mask = torch.tensor(output_mask[:max_length])
        output_masks.append(output_mask)

    # Calculate ppl
    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for i in tqdm(range(data_size)):
        input_ids = tokenized_instructions["input_ids"][i].unsqueeze(0).to('cuda')
        input_ids_prompt = tokenized_instructions2["input_ids"][i].unsqueeze(0).to('cuda')
        attn_mask = tokenized_instructions["attention_mask"][i].unsqueeze(0).to('cuda')
        output_mask = output_masks[i].unsqueeze(0).to('cuda')
        label = input_ids

        # with torch.no_grad():
        with torch.inference_mode():
            out_logits = model(input_ids, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_label = label[..., 1:].contiguous()
        shift_output_mask = output_mask[..., 1:].contiguous()
        nlls = loss_fct(shift_logits.transpose(1, 2), shift_label)
        perplexity_batch = torch.exp((nlls * shift_output_mask).sum(1) / shift_output_mask.sum(1))
        ppls += perplexity_batch.tolist()

        if i < 5 and show_example:
            with torch.inference_mode():
                outputs = model.generate(input_ids=input_ids_prompt, max_length=128)
                outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print('\n\n')
            print('pred:', outputs[0])
            print('\n')
            print('gt:', data[i]['output'])
            print('\n')
            print('ppl:', perplexity_batch.tolist())
            print('\n\n')

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        default=None,
        required=False,
        help="Path to the saved PEFT checkpoint."
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="zero_shot",
        required=False,
        help="Prompt mode."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        required=True,
        help="Path to test data."
    )
    args = parser.parse_args()

    # Load model
    bnb_config = get_bnb_config()

    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    else:
        model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
        revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"

    # Load LoRA
    if args.peft_path is not None:
        model = PeftModel.from_pretrained(model, args.peft_path)

    with open(args.test_data_path, "r") as f:
        data = json.load(f)

    model.eval()
    ppl = perplexity(model, tokenizer, data, 1024, args.prompt_mode)
    for i, x in enumerate(data):
        x['ppl'] = ppl['perplexities'][i]
    with open('few_shots.json', "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Mean perplexity:", ppl["mean_perplexity"])
