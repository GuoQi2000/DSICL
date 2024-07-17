#!/bin/bash
GPU=0
TASK="snli"
MODEL_PATH="xxx"
DATA_PATH="xxx"
METHOD="Random"
SEED=1
SHOTS=4

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift ;;
        --task) TASK="$2"; shift ;;
        --model_path) MODEL_PATH="$2"; shift ;;
        --data_path) DATA_PATH="$2"; shift ;;
        --method) METHOD="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --shots) SHOTS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

SAVING_PATH="./output/$TASK/$METHOD/seed=$SEED"

python Random_main.py --gpu "$GPU" \
                    --saving_path "$SAVING_PATH" \
                    --model_path "$MODEL_PATH" \
                    --data_path "$DATA_PATH" \
                    --task "$TASK"  \
                    --seed "$SEED"  \
                    --shots "$SHOTS"  \

python evaluate.py --gpu "$GPU" \
                   --saving_path "$SAVING_PATH" \
                   --model_path "$MODEL_PATH" \
                   --data_path "$DATA_PATH" \
                   --task "$TASK"  \
                   --seed "$SEED"  \
                   --method "$METHOD"  \
