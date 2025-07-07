"""
This module handles loading and managing configurations from YAML files.
"""

import yaml
import argparse
from typing import Any, Dict

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_config() -> Any:
    """
    Parses command-line arguments to get the config path, loads the config,
    and returns it as an object.

    This allows for easy attribute access (e.g., `config.model.d_model`)
    and is the main entry point for accessing configuration in the project.

    Returns:
        argparse.Namespace: An object containing the configuration.
    """
    parser = argparse.ArgumentParser(description="SEDD Research Framework")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    
    # Load the config from the specified YAML file
    cfg_dict = load_config(args.config)
    
    # Convert the nested dictionary to a Namespace object for attribute access
    return dict_to_namespace(cfg_dict)

def dict_to_namespace(d: Dict[str, Any]) -> argparse.Namespace:
    """
    Recursively converts a dictionary to a Namespace object.
    """
    namespace = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

if __name__ == '__main__':
    # Example of how to use the config loader
    config = get_config()
    print("--- Loaded Configuration ---")
    print(f"Experiment ID: {config.exp_id}")
    print(f"Model Dimension: {config.model.d_model}")
    print(f"Learning Rate: {config.training.learning_rate}")
    print(f"Noise Schedule: {config.diffusion.noise_schedule.name}")
