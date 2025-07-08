#!/bin/bash

# This script runs the final, best-effort "Self-Distrusting" Hybrid Model.

export PYTHONPATH=$PYTHONPATH:.
export TOKENIZERS_PARALLELISM=false

echo "======================================================"
echo "    STARTING FINAL SELF-DISTRUSTING HYBRID EXPERIMENT"
echo "======================================================"

# --- Train the Model ---
echo "\n--- Training Self-Distrusting Hybrid Model ---"
python src/trainer.py --config self_distrust_hybrid_config.yaml
if [ $? -ne 0 ]; then
    echo "Error: Self-Distrusting Hybrid model training failed."
    exit 1
fi
echo "Training complete."

# --- Analyze the Model ---
echo "\n--- Analyzing Self-Distrusting Hybrid Model ---"
python analyze_models.py \
    --exp_id "final_self_distrust_hybrid" \
    --gpu 0
if [ $? -ne 0 ]; then
    echo "Error: Analysis failed."
    exit 1
fi
echo "Analysis complete."

echo "\n======================================================"
echo "      FINAL EXPERIMENT COMPLETE!"
echo "======================================================"
echo "Results can be found in 'analysis_results/final_self_distrust_hybrid'."

