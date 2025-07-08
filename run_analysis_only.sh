#!/bin/bash

# This script runs only the analysis part for the four novel experiments,
# using the already trained checkpoints.

export PYTHONPATH=$PYTHONPATH:.
export TOKENIZERS_PARALLELISM=false

# --- Experiment Definitions ---
declare -A experiments
experiments=(
    ["novel_freq_noise"]="freq_noise_config.yaml"
    ["novel_self_refine"]="self_refine_config.yaml"
    ["novel_freq_refine"]="freq_refine_config.yaml"
    ["novel_self_distrust_hybrid"]="self_distrust_hybrid_config.yaml"
)

# --- Run Analysis ---
echo "\n======================================================"
echo "           STARTING ANALYSIS OF ALL MODELS"
echo "======================================================"
gpu=0
for exp_id in "${!experiments[@]}"; do
    echo "\n--- Analyzing: $exp_id on GPU $gpu ---"
    
    # For the self-refining models, we need to run the refined diversity analysis
    if [[ "$exp_id" == "novel_self_refine" || "$exp_id" == "novel_freq_refine" ]]; then
        analysis_parts="perplexity diversity refined_diversity"
    else
        analysis_parts="perplexity diversity"
    fi

    python analyze_models.py \
        --exp_id $exp_id \
        --gpu $gpu \
        --parts $analysis_parts
        
    gpu=$((gpu + 1))
    if [ $gpu -gt 7 ]; then
        gpu=0 # Reset GPU counter if we have more experiments than GPUs
    fi
done

echo "\n======================================================"
echo "      ALL NOVEL EXPERIMENTS AND ANALYSES COMPLETE!"
echo "======================================================"
