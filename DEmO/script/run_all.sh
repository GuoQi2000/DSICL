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

# Loop through the method list and call each script
for method in "${methods[@]}"; do

    sh script/$method --task subj --seed 1
done