#!/bin/bash

# This script runs our most promising experiment yet:
# The Frequency-Aware Self-Distrust Hybrid model.

export PYTHONPATH=$PYTHONPATH:.
export TOKENIZERS_PARALLELISM=false

CONFIG_FILE="freq_hybrid_config.yaml"
EXP_ID="final_freq_hybrid"
GPU=0

echo "======================================================"
echo "   STARTING FREQUENCY-AWARE HYBRID EXPERIMENT"
echo "======================================================"

# --- Step 1: Train the model ---
echo "\n--- Training Model: $EXP_ID ---"
python src/trainer.py --config $CONFIG_FILE
if [ $? -ne 0 ]; then
    echo "Error: Training failed for $EXP_ID."
    exit 1
fi
echo "Training complete."

# --- Step 2: Analyze the model ---
echo "\n--- Analyzing Model: $EXP_ID ---"
python analyze_models.py \
    --exp_id $EXP_ID \
    --gpu $GPU \
    --parts perplexity diversity
if [ $? -ne 0 ]; then
    echo "Error: Analysis failed for $EXP_ID."
    exit 1
fi
echo "Analysis complete."


echo "\n======================================================"
echo "      EXPERIMENT AND ANALYSIS COMPLETE!"
echo "======================================================"
echo "Final results are in: analysis_results/$EXP_ID/summary_gs_1.0.yaml"

