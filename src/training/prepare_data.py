import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import train_logger as logger
from typing import Tuple, List
from src.utils.config_loader import load_config
from src.utils.data_loader import load_data

train_config = load_config("hyperparameters.yaml")


def apply_majority_vote(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolves label conflicts using a majority vote approach. If a tie occurs, the phrase is discarded.

    Args:
        df (pd.DataFrame): Raw dataframe containing 'phrase_text' and 'sol' columns.

    Returns:
        pd.DataFrame: Consolidated dataframe with 'text' and 'label' columns (binary: ok=1, ko=0).
    """
    logger.info("[DATA PROCESSING] Starting majority vote resolution.")

    # Compute initial statistics
    total_phrases = df["phrase_text"].nunique()
    total_users = df["user"].nunique()

    # Identify conflicting labels
    conflict_counts = df.groupby("phrase_text")["sol"].nunique()
    conflicting_phrases = df[df["phrase_text"].isin(conflict_counts[conflict_counts > 1].index)]
    clean_data = df[~df["phrase_text"].isin(conflicting_phrases["phrase_text"])]

    total_conflicting_rows = len(conflicting_phrases)  # Total rows before processing
    total_conflicting_phrases = conflicting_phrases["phrase_text"].nunique()  # Unique phrases with conflicts

    logger.info(f"[DATA PROCESSING] Unique phrases: {total_phrases}")
    logger.info(f"[DATA PROCESSING] Unique annotators: {total_users}")
    logger.info(f"[DATA PROCESSING] Total conflicting phrases: {total_conflicting_phrases}")
    logger.info(f"[DATA PROCESSING] Total conflicting rows: {total_conflicting_rows}")

    resolved_conflicts = []
    discarded_phrases = 0

    for phrase, group in conflicting_phrases.groupby("phrase_text"):
        majority_label = group["sol"].mode()

        if len(majority_label) == 1:
            resolved_conflicts.append({"text": phrase, "label": 1 if majority_label.iloc[0] == "ok" else 0})
        else:
            discarded_phrases += 1  # Count phrases discarded

    # Convert resolved conflicts to DataFrame
    resolved_conflicts_df = pd.DataFrame(resolved_conflicts)
    resolved_phrases = resolved_conflicts_df["text"].nunique()
    # Log statistics
    logger.info(f"[DATA PROCESSING] Resolved {resolved_phrases} phrases via majority vote.")
    logger.info(f"[DATA PROCESSING] Discarded {discarded_phrases} rows due to unresolved ties.")

    # Convert clean data to the required format
    clean_data_df = clean_data.rename(columns={"phrase_text": "text", "sol": "label"})[["text", "label"]]
    clean_data_df["label"] = clean_data_df["label"].apply(lambda x: 1 if x == "ok" else 0)  # Convert to binary

    # Append resolved conflicts to the clean data
    final_df = pd.concat([clean_data_df, resolved_conflicts_df], ignore_index=True)
    logger.info(f"[DATA PROCESSING] Final dataset size: {len(final_df)} rows.")

    # Validate correct accounting of conflicts
    assert (resolved_phrases + discarded_phrases) == total_conflicting_phrases, \
        "Mismatch in total conflicting rows, resolved, and discarded counts!"
    return final_df



def data_preparation(data_filename: str) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Prepares the dataset for training by loading a CSV file, validating required columns,
    applying majority vote resolution, and splitting the data into training and testing sets.

    Args:
        data_filename (str): Path to the dataset file.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        - x_train: Training data (text).
        - x_test: Testing data (text).
        - y_train: Training labels.
        - y_test: Testing labels.

    Raises:
        ValueError: If required columns are missing.
    """
    logger.info("[TRAIN] Starting the training data preparation process.")

    # Load dataset
    df = load_data("raw", data_filename)
    get_stats(df)

    # Apply majority vote and consolidate the dataset
    consolidated_df = apply_majority_vote(df)

    logger.info(f"[TRAIN] Dataset consolidated successfully with {len(consolidated_df)} records.")

    # Extract features and labels
    x, y = consolidated_df["text"], consolidated_df["label"]

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=train_config["training"]["test_size"],
        random_state=train_config["training"]["random_state"],
        stratify=y if train_config["training"]["stratify"] else None
    )

    logger.info(f"[TRAIN] Data split into {len(x_train)} training and {len(x_test)} testing samples.")
    return x_train, x_test, y_train, y_test


def get_stats(dataframe: pd.DataFrame) -> None:
    """
    Logs key statistics about the dataset.

    Args:
        dataframe (pd.DataFrame): Corpus dataset.
    """
    total_rows = len(dataframe)
    total_unique_users = dataframe["user"].nunique()
    total_unique_phrases = dataframe["phrase_text"].nunique()

    # Identify conflicting data
    conflict_mask = dataframe.groupby("phrase_text")["sol"].nunique() > 1
    conflict_phrases = dataframe[dataframe["phrase_text"].isin(conflict_mask[conflict_mask].index)]

    # Compute statistics
    discarded_count = len(conflict_phrases)
    conflict_percentage = (discarded_count / total_rows) * 100 if total_rows > 0 else 0

    # Log statistics
    logger.info(f"[DATA PREPARATION] Total rows processed: {total_rows}")
    logger.info(f"[DATA PREPARATION] Total unique users: {total_unique_users}")
    logger.info(f"[DATA PREPARATION] Total unique phrases: {total_unique_phrases}")
    logger.info(f"[DATA PREPARATION] Conflicting phrases: {discarded_count} ({conflict_percentage:.2f}%)")



if __name__ == "__main__":
    x_train, x_test, y_train, y_test = data_preparation("stages-votes.json")
