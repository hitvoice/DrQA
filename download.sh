#!/usr/bin/env bash

# check if requirements are met
REQUIRED=(
    "wget"
    "unzip"
    "python3"
    "pip"
)
for ((i=0;i<${#REQUIRED[@]};++i)); do
    if ! [ -x "$(command -v ${REQUIRED[i]})" ]; then
        echo 'Error: ${REQUIRED[i]} is not installed.' >&2
        exit -1
    fi
done

# Download SQuAD & GloVe
SQUAD_DIR=SQuAD
GLOVE_DIR=glove
mkdir -p $SQUAD_DIR
mkdir -p $GLOVE_DIR

URLS=(
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
    "http://nlp.stanford.edu/data/glove.840B.300d.zip"
)
FILES=(
    "$SQUAD_DIR/train-v1.1.json"
    "$SQUAD_DIR/dev-v1.1.json"
    "$GLOVE_DIR/glove.840B.300d.zip"
)
for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    url=${URLS[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download."
    else
        wget $url -O $file
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".zip" ]; then
            unzip $file -d "$(dirname "$file")"
        fi
    fi
done

# Download SpaCy English language models
python3 -m spacy download en

