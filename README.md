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
├─── config.yaml       # Main configuration file for experiments
├─── run_train.sh      # Example script to start a training run
├─── run_final_analysis.sh # Runs a full analysis on specified models
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

## How to Run

This framework is configured using `.yaml` files. The main configuration is `config.yaml`. You can create new config files for different experiments.

### 1. Training a Model

To train a model, you can use the provided `run_train.sh` script.

1.  **Edit `config.yaml`** to define your experiment. Key parameters to change include:
    - `exp_id`: A unique name for your experiment (e.g., `absorbing_v2`).
    - `gpu`: The GPU index to use.
    - `diffusion.graph_type`: The main variable for your research (`uniform` or `absorbing`).
    - Other hyperparameters in the `model`, `diffusion`, and `training` sections.

2.  **Run the training script:**
    ```bash
    ./run_train.sh
    ```
    The script will use the settings in `config.yaml` to run the training. Checkpoints and logs will be saved to the `checkpoints/` and `wandb/` directories under the `exp_id`.

### 2. Analyzing a Model

Once a model is trained, you can run a comprehensive analysis on it using the `run_final_analysis.sh` script.

1.  **Edit `run_final_analysis.sh`** to specify the `exp_id` of the model(s) you want to analyze.

2.  **Run the analysis script:**
    ```bash
    ./run_final_analysis.sh
    ```
    This will perform the following analyses:
    - **Perplexity:** Calculated on the test set.
    - **Generation Diversity:** Measured with Self-BLEU and Distinct-N metrics.
    - **Infilling:** Assesses the model's contextual understanding.

    All results, including generated samples and a YAML summary of metrics, will be saved to the `analysis_results/<your_exp_id>/` directory. This provides a clean and organized way to compare different experiments.
