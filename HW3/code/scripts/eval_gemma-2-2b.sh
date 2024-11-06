ipython --pdb -- code/ppl.py \
    --base_model_path zake7749/gemma-2-2b-it-chinese-kyara-dpo \
    --peft_path results/gemma-2-2b-it-chinese-kyara-dpo \
    --test_data_path data/public_test.json \
    --prompt_mode training