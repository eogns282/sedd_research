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

This framework is configured using `.yaml` files. We provide `uniform_config.yaml` and `absorbing_config.yaml` as starting points.

### 1. Verifying the Framework

Before running long experiments, you can run a quick verification test to ensure all components are working correctly.
```bash
./run_verification_test.sh
```

### 2. Running the Full Experiment Pipeline

The main script for reproducing the core results is `run_final_experiments.sh`. This script will:
1.  Train the **Uniform** model from scratch using `uniform_config.yaml`.
2.  Train the **Absorbing** model from scratch using `absorbing_config.yaml`.
3.  Run a full analysis (perplexity, diversity, infilling) on the best checkpoint of each model.

To run the entire pipeline, simply execute:
```bash
./run_final_experiments.sh
```

### 3. Analyzing Results

After the final experiments are complete, all quantitative and qualitative results will be saved in the `analysis_results/` directory, under the `exp_id` specified in the config files (`final_uniform` and `final_absorbing`).

You can directly compare the summary files:
- `analysis_results/final_uniform/summary.yaml`
- `analysis_results/final_absorbing/summary.yaml`

This provides a clean and organized way to compare the performance of the two models.