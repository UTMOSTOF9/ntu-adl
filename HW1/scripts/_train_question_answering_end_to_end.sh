model_name=$1
ipython --pdb --  src/question_answering/train_question_answering.py \
    --model_name_or_path $model_name \
    --train_file data/processed/train_squad_end2end.json \
    --validation_file data/processed/valid_squad_end2end.json \
    --max_length 2048 \
    --pad_to_max_length \
    --per_device_train_batch_size 8 \
    --learning_rate 3e-5 \
    --weight_decay 0 \
    --num_train_epochs 6 \
    --num_warmup_steps 200 \
    --gradient_accumulation_steps 6 \
    --output_dir results/${model_name}_end2end/ckpt/question_answering \
    --seed 0822 \
    --mixed_precision fp16