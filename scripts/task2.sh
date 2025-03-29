#!/bin/bash

scripts=$(dirname "$0")
base=$(realpath "$scripts/..")

models="$base/models"
data="$base/data"
tools="$base/tools"
logs="$base/logs"

mkdir -p "$logs"

num_threads=4
device=""

SECONDS=0

# list of dropout settings
dropouts=(0 0.25 0.3 0.69 0.8 0.9)

for dropout in "${dropouts[@]}"; do
    (
    cd "$tools/pytorch-examples/word_language_model" &&
    CUDA_VISIBLE_DEVICES="$device" OMP_NUM_THREADS="$num_threads" python3 main.py --data "$data/romjul" \
        --epochs 40 \
        --log-interval 100 \
        --emsize 250 --nhid 250 --dropout "$dropout" --tied \
        --save "$models/model_${dropout}.pt" \
        --mps \
        --log-file "$logs/model_${dropout}.csv"
    )
done

python3 "$scripts/grapher.py"
python3 "$scripts/table_gen.py"
./generate_task2.sh
