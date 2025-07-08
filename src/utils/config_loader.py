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

def get_config(config_path: str) -> Any:
    """
    Loads the config from a given path and returns it as a namespace object.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        Any: An object containing the configuration.
    """
    # Load the config from the specified YAML file
    cfg_dict = load_config(config_path)
    
    # Convert the nested dictionary to a Namespace object for attribute access
    config = dict_to_namespace(cfg_dict)
    
    return config

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
    # This part is for standalone testing of the loader
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    config = get_config(args.config)
    print("--- Loaded Configuration ---")
    print(f"Experiment ID: {config.exp_id}")
    print(f"Model Dimension: {config.model.d_model}")
    print(f"Learning Rate: {config.training.learning_rate}")
    print(f"Noise Schedule: {config.diffusion.noise_schedule}")
