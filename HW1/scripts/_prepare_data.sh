python src/train_data_preprocess.py \
    --data_dir data/raw \
    --context_file data/raw/context.json \
    --output_folder data/processed

python src/test_data_preprocess.py \
    --context_file data/raw/context.json \
    --test_file data/raw/test.json \
    --output_folder data/processed