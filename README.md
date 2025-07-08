# Score-Entropy Discrete Diffusion (SEDD) Research Framework

This repository provides a research-grade framework for experimenting with Score-Entropy Discrete Diffusion models, based on the paper ["Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution"](https://arxiv.org/abs/2310.16834).

The framework is designed to be clear, extensible, and easy to use for comparing different diffusion strategies (e.g., Uniform vs. Absorbing states) and for exploring new ideas in discrete diffusion.

## Project Structure

```
/
├─── checkpoints/      # Saved model checkpoints
├─── analysis_results/ # Detailed analysis outputs (metrics, samples)
├─── src/
│   ├─── data.py         # Data loading and preprocessing
│   ├─── diffusion/      # Core diffusion logic (process, graphs, schedules)
│   ├─── model.py        # Transformer model architecture
│   ├─── utils/
│   │   └─── config_loader.py # Handles loading YAML configs
│   ├─── losses.py       # Training loss function
│   └─── trainer.py      # Main training script
├─── uniform_config.yaml # Configuration for the Uniform model
├─── absorbing_config.yaml # Configuration for the Absorbing model
├─── run_final_experiments.sh # End-to-end script to train and analyze both models
├─── run_verification_test.sh # Quick script to verify the environment
└─── requirements.txt  # Python dependencies
```

## Setup

1.  **Create and activate a conda environment:**
    ```bash
    conda create -n text-diffusion-poc python=3.9
    conda activate text-diffusion-poc
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run Experiments

This framework is configured using `.yaml` files. The primary models are the original `Absorbing` model and the new state-of-the-art `Gated-Absorbing` model.

### 1. Verifying the Framework

Before running long experiments, you can run a quick verification test to ensure all components are working correctly.
```bash
./run_verification_test.sh
```

### 2. Reproducing the Final Results

To train and evaluate the final, best-performing model, use the `run_gated_absorbing_experiment.sh` script. This will train the model from scratch and run a full analysis.
```bash
./run_gated_absorbing_experiment.sh
```
This script uses the `gated_absorbing_config.yaml` file.

To compare this to the original baselines, you can still use the `run_final_experiments.sh` script, which trains the original `Uniform` and `Absorbing` models.

### 3. Analyzing Results

After the experiments are complete, all results will be saved in the `analysis_results/` directory under their respective `exp_id`s.

You can compare the summary files:
- `analysis_results/final_absorbing/summary.yaml`
- `analysis_results/final_gated_absorbing/summary.yaml`

This provides a clean and organized way to compare the performance of the models.