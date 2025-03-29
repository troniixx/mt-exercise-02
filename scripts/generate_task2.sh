#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools
samples=$base/samples

mkdir -p $samples

num_threads=4
device=""

for model in $models/*.pt; do
    (
    cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python3 generate.py \
        --data $data/romjul \
        --words 200 \
        --checkpoint "$model" \
        --outf "$samples/sample_$(basename "$model").txt" \
        --mps \
        --temperature 0.6
    )
done
