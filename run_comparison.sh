#!/bin/bash

# This script runs a comparison between the uniform and absorbing graph types in parallel.
# It trains a model for each type on a separate GPU and saves the results in separate experiment directories.

export PYTHONPATH=$PYTHONPATH:.

# --- Run Experiments in Parallel ---
echo "--- Running Uniform and Absorbing Graph Experiments in Parallel ---"

python src/trainer.py --exp_id "uniform_experiment" --gpu 1 --graph_type "uniform" &
PID1=$!
echo "Started Uniform Experiment on GPU 1 (PID: $PID1)"

python src/trainer.py --exp_id "absorbing_experiment" --gpu 2 --graph_type "absorbing" &
PID2=$!
echo "Started Absorbing Experiment on GPU 2 (PID: $PID2)"

# Wait for both experiments to complete
wait $PID1
echo "Uniform Experiment (PID: $PID1) has finished."
wait $PID2
echo "Absorbing Experiment (PID: $PID2) has finished."

echo "--- Comparison experiments complete ---"
echo "You can find the results in the 'checkpoints/uniform_experiment' and 'checkpoints/absorbing_experiment' directories."
echo "You can also view the logs in the 'wandb' directory."
