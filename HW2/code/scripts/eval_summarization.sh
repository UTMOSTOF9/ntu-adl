python code/jsonl2json.py --input="data/raw/public.jsonl" --output="data/processed/public.json"
python code/eval_summarization.py \
--model_name_or_path="results/summerization/best" \
--cache_dir="./cache" \
--valid_file="data/processed/public.json" \
--text_column="maintext" \
--summary_column="title" \
--preprocessing_workers=6 \
--output_dir="./results/summerization" \
--per_device_eval_batch_size=48 \
--dataloader_num_workers=8 \
--predict_with_generate=True \
--pad_to_max_length=True \
--bf16 --tf32=y