python code/jsonl2json.py --input=$1 --output="tmp.json"
python code/pred_summarization.py \
    --model_name_or_path="pretrained" \
    --cache_dir="./cache" \
    --test_file="tmp.json" \
    --pred_output_file=$2 \
    --text_column="maintext" \
    --summary_column="title" \
    --preprocessing_workers=6 \
    --output_dir="./pretrained" \
    --per_device_eval_batch_size=8 \
    --dataloader_num_workers=4 \
    --predict_with_generate=True \
    --pad_to_max_length=True \
    --bf16 --tf32=y