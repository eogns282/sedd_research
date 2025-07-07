#!/bin/bash

# This script runs a quick, end-to-end verification of the framework
# to ensure all components are working correctly.

export PYTHONPATH=$PYTHONPATH:.
export TOKENIZERS_PARALLELISM=false

CONFIG_FILE="test_config.yaml"
EXP_ID="verification_test"

echo "======================================================"
echo "         STARTING FRAMEWORK VERIFICATION TEST"
echo "======================================================"

# --- Step 1: Run a short training session ---
echo "\n--- Testing Training Pipeline (Debug Mode) ---"
python src/trainer.py --config $CONFIG_FILE --debug
if [ $? -ne 0 ]; then
    echo "Error: Training pipeline failed."
    exit 1
fi
echo "Training pipeline test successful."


# --- Step 2: Run a full analysis ---
echo "\n--- Testing Analysis Pipeline ---"
python analyze_models.py \
    --exp_id $EXP_ID \
    --gpu 0 \
    --num_samples 2 # Just 2 samples for a quick test
if [ $? -ne 0 ]; then
    echo "Error: Analysis pipeline failed."
    exit 1
fi
echo "Analysis pipeline test successful."


echo "\n======================================================"
echo "      FRAMEWORK VERIFICATION TEST PASSED!"
echo "======================================================"
# Clean up the test artifacts
rm -rf checkpoints/$EXP_ID
rm -rf analysis_results/$EXP_ID
rm $CONFIG_FILE
echo "Test artifacts have been cleaned up."

