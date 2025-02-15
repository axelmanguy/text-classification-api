import logging
import os
from datetime import datetime
from pathlib import Path

# Get the directory of the current script
# Ensure the log directory exists
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # makefile level
LOG_DIR = os.path.join(base_dir,"logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Generate a timestamp for the log filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# Define log file paths
train_log_file = os.path.join(LOG_DIR, f"{timestamp}_NP_train.log")
test_log_file = os.path.join(LOG_DIR, f"{timestamp}_NP_test.log")
inference_log_file = os.path.join(LOG_DIR, f"{timestamp}_NP_runtime.log")

# Create and configure loggers
def setup_logger(name, log_file):
    """Setup a logger with a specific file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Avoid duplicate handlers in case of multiple calls
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger

# Define separate loggers for training and serving
train_logger = setup_logger("train_logger", train_log_file)
inference_logger = setup_logger("inference_logger", inference_log_file)
test_logger = setup_logger("test_logger", test_log_file)
