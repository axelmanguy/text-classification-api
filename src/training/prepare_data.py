import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import train_logger as logger
from typing import Tuple, List
from src.utils.config_loader import load_config
from src.utils.data_loader import load_data

train_config = load_config("hyperparameters.yaml")
def data_preparation(data_filename: str) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Prepares the dataset for training by loading a CSV file, validating required columns,
    and splitting the data into training and testing sets.

    Args:
        data_filename (str): Path to the CSV file containing the dataset.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        - x_train: Training data (text).
        - x_test: Testing data (text).
        - y_train: Training labels.
        - y_test: Testing labels.

    Raises:
        ValueError: If the CSV file does not contain the required 'text' and 'label' columns.
    """
    logger.info("[TRAIN] Starting the training data preparation process.")

    # Load dataset from CSV file
    df = load_data('raw','stages-votes.json')
    # Count occurrences of different labels for the same phrase_text
    conflict_mask = df.groupby("phrase_text")["sol"].nunique() > 1
    # Separate conflicting and clean data
    conflict_phrases = df[
        df["phrase_text"].isin(df.groupby("phrase_text").filter(lambda x: x["sol"].nunique() > 1)["phrase_text"])]
    clean_data = df[~df.index.isin(conflict_phrases.index)]

    # Log and print discarded data count
    discarded_count = len(conflict_phrases)
    logger.info(f"Discarded {discarded_count} rows due to label conflicts.")
    print(f"Discarded {discarded_count} rows due to label conflicts.")

    x, y = df["phrase_text"], df["sol"]
    logger.info(f"[TRAIN] Dataset loaded successfully with {len(df)} records.")

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=train_config["training"]["test_size"],
        random_state=train_config["training"]["random_state"],
        stratify=y if train_config["training"]["stratify"] else None
    )

    logger.info(f"[TRAIN] Data split into {len(x_train)} training and {len(x_test)} testing samples.")
    return x_train, x_test, y_train, y_test


if __name__=="__main__":
    x_train, x_test, y_train, y_test=data_preparation("stages-votes.json")