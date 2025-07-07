#!/bin/bash

# This script runs the final, comprehensive analysis on the best-performing
# uniform and absorbing models.

export PYTHONPATH=$PYTHONPATH:.
export TOKENIZERS_PARALLELISM=false

echo "======================================================"
echo "      RUNNING FINAL ANALYSIS: UNIFORM MODEL"
echo "======================================================"
python analyze_models.py \
    --exp_id "uniform_experiment_continued" \
    --gpu 1

echo "

======================================================"
echo "    RUNNING FINAL ANALYSIS: RETRAINED ABSORBING MODEL"
echo "======================================================"
python analyze_models.py \
    --exp_id "absorbing_retrained" \
    --gpu 1

echo "
--- Final Analysis Complete ---"
echo "All results have been saved to the 'analysis_results' directory."

