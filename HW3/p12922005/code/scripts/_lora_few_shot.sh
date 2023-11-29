python3 src/ppl.py \
    --base_model_path ckpt/Taiwan-LLM-7B-v2.0-chat \
    --peft_path adapter_checkpoiint \
    --prompt_mode few_shot \
    --test_data_path data/public_test.json