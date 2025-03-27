#!/bin/bash

# Resolve script and base directories more robustly
scripts="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
base="$(dirname "$scripts")"
data="$base/data"
tools="$base/tools"

# Create necessary directories
mkdir -p "$data"
mkdir -p "$data/wikitext-2"
mkdir -p "$data/romjul/raw"

# Link default training data for easier access
for corpus in train valid test; do
    absolute_path=$(realpath "$tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt")
    ln -snf "$absolute_path" "$data/wikitext-2/$corpus.txt"
done

# Download the Romeo and Juliet text file
curl -L "https://www.gutenberg.org/files/1513/1513-0.txt" -o "$data/romjul/raw/tales.txt"

# Verify file was downloaded successfully
if [ ! -s "$data/romjul/raw/tales.txt" ]; then
    echo "Failed to download the file"
    exit 1
fi

# Preprocess slightly
python3 "$base/scripts/preprocess_raw.py" < "$data/romjul/raw/tales.txt" > "$data/romjul/raw/tales.cleaned.txt"

# Tokenize, fix vocabulary upper bound
python3 "$base/scripts/preprocess.py" --vocab-size 5000 --tokenize --lang "en" --sent-tokenize < "$data/romjul/raw/tales.cleaned.txt" > "$data/romjul/raw/tales.preprocessed.txt"

# Split into train, valid, and test
head -n 440 "$data/romjul/raw/tales.preprocessed.txt" | tail -n 400 > "$data/romjul/valid.txt"
head -n 840 "$data/romjul/raw/tales.preprocessed.txt" | tail -n 400 > "$data/romjul/test.txt"
tail -n 3075 "$data/romjul/raw/tales.preprocessed.txt" | head -n 2955 > "$data/romjul/train.txt"

echo "Data download and preprocessing complete!"