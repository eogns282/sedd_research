#!/bin/bash
# A simple script to run the training process.

# This script trains a model using the settings defined in the specified
# YAML configuration file.

# It's recommended to copy and modify this script for different experiments.
# For example, you could create 'train_uniform.sh' and 'train_absorbing.sh'
# and point them to different config files or override parameters.

export PYTHONPATH=$PYTHONPATH:.
export TOKENIZERS_PARALLELISM=false

echo "======================================================"
echo "           STARTING TRAINING RUN"
echo "======================================================"

python src/trainer.py --config config.yaml

echo "\n--- Training complete ---"