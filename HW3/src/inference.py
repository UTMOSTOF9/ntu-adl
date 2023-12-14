import argparse
import json

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_bnb_config, get_prompt


def main():
    parser = argparse.ArgumentParser('Inference LLaMav2')
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
        default="data/private_test.json",
        required=True,
        help="Path to test data."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/private_test_outputs.json",
        required=True,
        help="Path to output data."
    )
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=get_bnb_config(),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load LoRA
    if args.peft_path is not None:
        model = PeftModel.from_pretrained(model, args.peft_path)

    dataset = load_dataset("json", data_files={"test": args.test_data_path})['test']

    with open(args.test_data_path, "r") as f:
        raw_data = json.load(f)

    output_data = []
    with torch.inference_mode():
        for i, data in tqdm(enumerate(dataset)):
            prompt = get_prompt(data['instruction'], prompt_mode=args.prompt_mode)
            input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
            outputs = model.generate(input_ids=input_ids)
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_data.append(
                {
                    'id': raw_data[i]['id'],
                    'output': outputs[0].split(prompt + ' ')[-1]
                }
            )

    with open(args.output_path, 'w') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
