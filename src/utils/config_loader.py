import os
import yaml
from typing import Dict

def load_config(config_name: str = "hyperparameters.yaml") -> Dict:
    """
    Load a YAML configuration file from the `config/` directory.

    This function constructs the absolute path to the configuration file,
    verifies its existence, loads its contents, and adds `root_dir` and `src_dir`
    paths dynamically for reference in configurations.

    Args:
        config_name (str): The name of the configuration file to load.
                           Defaults to "hyperparameters.yaml".

    Returns:
        Dict: The loaded configuration as a Python dictionary.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        ValueError: If the file cannot be parsed as valid YAML.
    """
    # Get the absolute path to the `src/` directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/ level
    config_path = os.path.join(base_dir, "config", config_name)

    # Verify that the configuration file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        # Load the YAML configuration file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError(f"Invalid YAML format in {config_name}")

        # Dynamically add directory paths to the configuration
        config.setdefault("export", {})  # Ensure 'export' key exists
        config["export"]["root_dir"] = os.path.dirname(base_dir)  # Root project directory
        config["export"]["src_dir"] = base_dir  # Path to `src/` directory

        return config

    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {config_name}: {e}")
