
python src/paragraph_selection/test_paragraph_selection.py \
	--model_name_or_path results/$1/ckpt/paragraph_selection \
	--test_file data/processed/test_swag.json \
	--pred_file results/test/paragraph_selection_pred.json \
	--max_length 512 \
	--batch_size 4 \
	--device cuda

python src/question_answering/test_question_answering.py \
	--model_name_or_path results/$2/ckpt/question_answering \
	--test_file results/test/paragraph_selection_pred.json \
	--pred_file results/test/results.csv \
	--max_length 512 \
	--batch_size 4 \
	--device cuda