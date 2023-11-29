python3 src/gen_few_shots.py \
    --base_model_path ckpt/Taiwan-LLM-7B-v2.0-chat \
    --peft_path adapter_checkpoiint \
    --test_data_path data/train.json \
    --num_samples 1000