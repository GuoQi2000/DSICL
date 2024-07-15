#!/bin/bash

# 默认值
GPU=0
TASK="mr"
MODEL_PATH="/home/gq/model/llama_1.3b"
METHOD="Zero"
SEED=0

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift ;;
        --task) TASK="$2"; shift ;;
        --model_path) MODEL_PATH="$2"; shift ;;
        --method) METHOD="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

SAVING_PATH="./output/$TASK/$METHOD/seed=$SEED"

# 主要逻辑
python Zero_main.py --gpu "$GPU" \
                    --saving_path "$SAVING_PATH" \
                    --model_path "$MODEL_PATH" \
                    --task "$TASK"  \
                    --seed "$SEED"  \

python evaluate.py --gpu "$GPU" \
                   --saving_path "$SAVING_PATH" \
                   --model_path "$MODEL_PATH" \
                   --task "$TASK"  \
                   --seed "$SEED"  \
                   --method "$METHOD"  \