ipython --pdb -- code/ppl.py \
    --base_model_path yentinglin/Llama-3-Taiwan-8B-Instruct \
    --test_data_path data/public_test.json \
    --prompt_mode training
    # --peft_path results/gemma-2-2b-it-chinese-kyara-dpo \