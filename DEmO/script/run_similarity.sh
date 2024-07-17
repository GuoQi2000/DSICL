#!/bin/bash
GPU=0
TASK="mr"
MODEL_PATH="xxx"
DATA_PATH="xxx"
METHOD="Similarity"
SEED=0
SHOTS=4
EMBEDDING_MODEL_PATH="/home/gq/model/roberta-large"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift ;;
        --task) TASK="$2"; shift ;;
        --model_path) MODEL_PATH="$2"; shift ;;
        --data_path) DATA_PATH="$2"; shift ;;
        --method) METHOD="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --shots) SHOTS="$2"; shift ;;
        --embedding_model_path) EMBEDDING_MODEL_PATH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

SAVING_PATH="./output/$TASK/$METHOD/seed=$SEED"

python Similarity_main.py --gpu "$GPU" \
                    --saving_path "$SAVING_PATH" \
                    --model_path "$MODEL_PATH" \
                    --data_path "$DATA_PATH" \
                    --task "$TASK"  \
                    --seed "$SEED"  \
                    --shots "$SHOTS"  \
                    --embedding_model_path "$EMBEDDING_MODEL_PATH"  \

python evaluate.py --gpu "$GPU" \
                   --saving_path "$SAVING_PATH" \
                   --model_path "$MODEL_PATH" \
                   --data_path "$DATA_PATH" \
                   --task "$TASK"  \
                   --seed "$SEED"  \
                   --method "$METHOD"  \
