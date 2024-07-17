#!/bin/bash

# Method list
methods=(
    "run_random.sh"
    "run_demo.sh"
    "run_best.sh"
    "run_gle.sh"
    "run_mdl.sh"
    "run_mi.sh"
    "run_similarity.sh"
    "run_zero.sh"
)

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift ;;
        --task) TASK="$2"; shift ;;
        --model_path) MODEL_PATH="$2"; shift ;;
        --data_path) DATA_PATH="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Loop through the method list and call each script
for method in "${methods[@]}"; do
    sh script/$method --task $TASK --seed $SEED --model_path $MODEL_PATH --data_path $DATA_PATH --gpu $GPU
done