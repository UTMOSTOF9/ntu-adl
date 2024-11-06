set -e
ipython --pdb -- code/train.py \
	--model_id yentinglin/Llama-3-Taiwan-8B-Instruct \
	--output-dir results/Llama-3-Taiwan-8B-Instruct
