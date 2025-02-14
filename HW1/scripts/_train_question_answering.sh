model_name=$1
python src/question_answering/train_question_answering.py \
    --model_name_or_path $model_name \
    --train_file data/processed/train_squad.json \
    --validation_file data/processed/valid_squad.json \
    --max_length 512 \
    --pad_to_max_length \
    --per_device_train_batch_size 8 \
    --learning_rate 3e-5 \
    --weight_decay 0 \
    --num_train_epochs 6 \
    --num_warmup_steps 100 \
    --gradient_accumulation_steps 6 \
    --lr_scheduler_type linear \
    --output_dir results/$model_name/ckpt/question_answering \
    --seed 0822 \
    --mixed_precision fp16