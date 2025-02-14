context_path=$1
test_path=$2
pred_path=$3

# prepare data
bash scripts/_prepare_test_dataset.sh $context_path $test_path

# run test

bash scripts/_test_for_ta.sh hfl/chinese-macbert-base $pred_path
echo "Finish on variant bert (hfl/chinese-macbert-base)"

# bash scripts/_test.sh bert-base-chinese
# echo "Finish on baseline (bert-base-chinese)"

# bash scripts/_test.sh hfl/chinese-roberta-wwm-ext
# echo "Finish on variant bert (hfl/chinese-roberta-wwm-ext)"

# bash scripts/_test_from-scratch.sh bert-base-chinese
# echo "Finish on from-scratch mode of baseline model (bert-base-chinese)"
