# TA Baseline
ipython --pdb -- src/paragraph_selection.py \
    --config-path $1 \
    --do-train \
    --do-test \
    --ckpt-path $2
