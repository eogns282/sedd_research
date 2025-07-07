#!/bin/bash

# This is the final, definitive script to run the full set of experiments
# from scratch on the stable, refactored codebase.
# It trains both the uniform and absorbing models, then runs a full
# analysis on the best checkpoint of each.

export PYTHONPATH=$PYTHONPATH:.
export TOKENIZERS_PARALLELISM=false

echo "======================================================"
echo "        STARTING FINAL EXPERIMENT PIPELINE"
echo "======================================================"

# --- Step 1: Train the Uniform Model ---
echo "\n--- Training Final Uniform Model ---"
python src/trainer.py --config uniform_config.yaml
if [ $? -ne 0 ]; then
    echo "Error: Uniform model training failed."
    exit 1
fi
echo "Uniform model training complete."

# --- Step 2: Train the Absorbing Model ---
echo "\n--- Training Final Absorbing Model ---"
python src/trainer.py --config absorbing_config.yaml
if [ $? -ne 0 ]; then
    echo "Error: Absorbing model training failed."
    exit 1
fi
echo "Absorbing model training complete."


# --- Step 3: Analyze the Uniform Model ---
echo "\n--- Analyzing Final Uniform Model ---"
python analyze_models.py \
    --exp_id "final_uniform" \
    --gpu 0
if [ $? -ne 0 ]; then
    echo "Error: Uniform model analysis failed."
    exit 1
fi
echo "Uniform model analysis complete."


# --- Step 4: Analyze the Absorbing Model ---
echo "\n--- Analyzing Final Absorbing Model ---"
python analyze_models.py \
    --exp_id "final_absorbing" \
    --gpu 1
if [ $? -ne 0 ]; then
    echo "Error: Absorbing model analysis failed."
    exit 1
fi
echo "Absorbing model analysis complete."


echo "\n======================================================"
echo "      FINAL EXPERIMENT PIPELINE COMPLETE!"
echo "======================================================"
echo "All results can be found in the 'analysis_results' directory."
echo "You can compare 'final_uniform/summary.yaml' and 'final_absorbing/summary.yaml'."
