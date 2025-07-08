#!/bin/bash

# This is the final, definitive script to run the new state-of-the-art
# Gated-Absorbing model.

export PYTHONPATH=$PYTHONPATH:.
export TOKENIZERS_PARALLELISM=false

echo "======================================================"
echo "    STARTING FINAL GATED-ABSORBING EXPERIMENT"
echo "======================================================"

# --- Train the Gated-Absorbing Model ---
echo "\n--- Training Gated-Absorbing Model ---"
python src/trainer.py --config gated_absorbing_config.yaml
if [ $? -ne 0 ]; then
    echo "Error: Gated-Absorbing model training failed."
    exit 1
fi
echo "Gated-Absorbing model training complete."

# --- Analyze the Gated-Absorbing Model ---
echo "\n--- Analyzing Gated-Absorbing Model ---"
python analyze_models.py \
    --exp_id "final_gated_absorbing" \
    --gpu 0
if [ $? -ne 0 ]; then
    echo "Error: Gated-Absorbing model analysis failed."
    exit 1
fi
echo "Gated-Absorbing model analysis complete."


echo "\n======================================================"
echo "      GATED-ABSORBING EXPERIMENT COMPLETE!"
echo "======================================================"
echo "All results can be found in 'analysis_results/final_gated_absorbing'."

