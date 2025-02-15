import json
import os
import pandas as pd
from src.utils.logger import train_logger as logger


def load_data(data_folder: str = "raw", data_filename: str = "stages-votes.json") -> pd.DataFrame:
    """
    Load a dataset from the `data/` directory.

    This function constructs an absolute path to the dataset based on the provided
    folder and filename, verifies its existence, and loads it into a Pandas DataFrame.

    Args:
        data_folder (str): The subdirectory within `data/` where the dataset is located.
                           Defaults to "raw".
        data_filename (str): The name of the dataset file. Defaults to "stages-votes.json".

    Returns:
        pd.DataFrame: The loaded dataset as a Pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the dataset cannot be loaded into a DataFrame.
    """
    # Get the absolute path of the data directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # makefile level
    data_path = os.path.join(base_dir, "data", data_folder, data_filename)

    # Verify that the file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    try:
        # Load the dataset
        # Load JSON file into a DataFrame
        with open(data_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        df = pd.DataFrame(data)
        print(df.head())
        logger.info(f'[DATA LOAD] loaded a dataframe of shape {df.shape}')
        return df
    except Exception as e:
        raise ValueError(f"Failed to load dataset {data_filename}: {e}")

if __name__=='__main__':
    load_data()
