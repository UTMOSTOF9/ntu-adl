set -e
ipython --pdb -- code/train.py \
	--model_id zake7749/gemma-2-2b-it-chinese-kyara-dpo \
	--output-dir results/gemma-2-2b-it-chinese-kyara-dpo