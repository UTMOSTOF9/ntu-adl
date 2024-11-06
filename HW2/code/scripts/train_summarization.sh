python code/jsonl2json.py --input="data/raw/train.jsonl" --output="data/processed/train.json"
python code/jsonl2json.py --input="data/raw/public.jsonl" --output="data/processed/public.json"

python code/train_summarization.py \
    --model_name_or_path="google/mt5-small" \
    --cache_dir="./cache" \
    --output_dir="./results/summerization2" \
    --train_file="data/processed/train.json" \
    --valid_file="data/processed/public.json" \
    --text_column="maintext" \
    --summary_column="title" \
    --preprocessing_workers=6 \
    --do_train \
    --do_eval \
    --num_train_epochs=20 \
    --per_device_train_batch_size=12 \
    --per_device_eval_batch_size=12 \
    --gradient_accumulation_steps=4 \
    --learning_rate=5e-5 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=300 \
    --dataloader_num_workers=8 \
    --overwrite_output_dir \
    --evaluation_strategy="steps" \
    --save_strategy="steps" \
    --eval_steps=1000 \
    --save_steps=1000 \
    --metric_for_best_model="rouge_combined" \
    --load_best_model_at_end  \
    --report_to="tensorboard" \
    --predict_with_generate=True \
    --pad_to_max_length=True \
    --num_beams=3 \
    --fp16 --tf32=y