import os
from datetime import datetime

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, RocCurveDisplay, auc

from src.training.prepare_data import data_preparation
from src.utils.logger import train_logger as logger
from typing import Tuple, List
from src.utils.config_loader import load_config
from src.utils.data_loader import load_data

train_config = load_config("hyperparameters.yaml")


def build_pipeline(max_k: int) -> Pipeline:
    """
    Constructs a machine learning pipeline for text classification using TF-IDF vectorization,
    feature selection, and a linear SVM classifier.

    Args:
        max_k (int): The maximum number of top features to select.

    Returns:
        Pipeline: A scikit-learn pipeline with preprocessing, feature selection, and classification.

    Raises:
        ValueError: If `max_k` is not a positive integer.
    """
    if max_k <= 0:
        raise ValueError("max_k must be a positive integer.")

    logger.info("[PIPELINE] Building the machine learning pipeline.")

    # Define the pipeline with preprocessing, feature selection, and classification steps
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=tuple(train_config["pipeline"]['tfidf']["ngram_range"]),
            min_df=train_config["pipeline"]['tfidf']["min_df"],
            dtype=train_config["pipeline"]['tfidf']["dtype"],
            strip_accents=train_config["pipeline"]['tfidf']["strip_accents"],
            decode_error=train_config["pipeline"]['tfidf']["decode_error"],
            analyzer=train_config["pipeline"]['tfidf']["token_mode"]
        )),
        ("select_kbest", SelectKBest(
            score_func=f_classif,
            k=min(train_config["pipeline"]['kbest']["top_k"], max_k)  # Ensure k does not exceed max_k
        )),
        ("classifier", LinearSVC())
    ])

    logger.info("[PIPELINE] Pipeline construction completed successfully.")

    return pipeline


def kfold_pipeline_validation(model: Pipeline, x: pd.Series, y: pd.Series, output_filepath:str) -> None:
    """
    Performs 5-fold cross-validation on a classification pipeline and evaluates its performance using ROC curves.
    adapted script from :
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

    Args:
        model (Pipeline): A trained scikit-learn pipeline for classification.
        X (pd.Series): Feature matrix containing input data.
        y (pd.Series): Target labels corresponding to the input data.
        output_filepath (str) : output filepath

    Returns:
        None: The function saves the ROC curve plot and logs AUC scores for each fold.

    Raises:
        ValueError: If `X` and `y` have inconsistent lengths.
    """
    if len(x) != len(y):
        raise ValueError("Feature matrix X and target labels y must have the same length.")

    logger.info("[KFOLD] Starting cross-validation.")

    # Convert input data to NumPy arrays (ensures compatibility with scikit-learn)
    x, y = np.array(x), np.array(y)

    # Define stratified 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5)

    # Lists to store true positive rates and AUC scores
    tprs: List[np.ndarray] = []
    aucs: List[float] = []
    mean_fpr = np.linspace(0, 1, 100)

    # Initialize figure for ROC curve plotting
    fig, ax = plt.subplots(figsize=(6, 6))

    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y)):
        # Train the model on the training set
        model.fit(x[train_idx], y[train_idx])

        # Compute ROC curve and AUC for the current fold
        roc_viz = RocCurveDisplay.from_estimator(
            model,
            x[test_idx],
            y[test_idx],
            name=f"ROC Fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
            plot_chance_level=(fold == 4),
        )

        # Log AUC score for the current fold
        logger.info(f'[KFOLD] Fold {fold} - ROC AUC: {roc_viz.roc_auc:.4f}')

        # Interpolate true positive rates to align different curves
        interp_tpr = np.interp(mean_fpr, roc_viz.fpr, roc_viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_viz.roc_auc)

    # Compute mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure the final point reaches (1,1)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Plot mean ROC curve
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=rf"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.4f})",
        lw=2,
        alpha=0.8,
    )

    # Compute standard deviation of true positive rates
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    # Plot confidence interval around the mean ROC curve
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    # Set plot labels and title
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Mean ROC Curve with Variability\n(Positive Label)",
    )
    ax.legend(loc="lower right")

    # Save the ROC curve plot
    plt.savefig(output_filepath)

    logger.info(f"[KFOLD] Cross-validation completed successfully. ROC curve saved as '{output_filepath}'.")

def train_pipeline(pipeline: Pipeline, x_train: pd.Series, y_train: pd.Series) -> Pipeline:
    """
    Trains a machine learning pipeline for text classification

    Args:
        pipeline (Pipeline): Untrained scikit learn pipeline
        x_train (pd.Series): Training data containing text samples.
        y_train (pd.Series): Corresponding labels for the training data.

    Returns:
        Pipeline: A trained scikit-learn pipeline ready for predictions.
    """
    logger.info("[TRAIN] Starting model training.")
    pipeline.fit(x_train, y_train)
    logger.info("[TRAIN] Model training completed successfully.")
    return pipeline

def test_pipeline(pipeline: Pipeline, x_test: pd.Series, y_test: pd.Series, output_report_filepath:str) -> str:
    """
    Evaluates a trained machine learning pipeline on test data and generates a classification report.

    Args:
        pipeline (Pipeline): A trained scikit-learn pipeline for text classification.
        x_test (pd.Series): Test data containing text samples.
        y_test (pd.Series): True labels corresponding to the test data.
        output_report_filepath (str) : Optional, where to write classification report as txt

    Returns:
        str: A classification report summarizing the model's performance.
    """
    logger.info("[TRAIN] Starting model evaluation.")

    # Generate predictions using the trained pipeline
    y_pred = pipeline.predict(x_test)

    # Compute classification metrics
    report = classification_report(y_test, y_pred)

    # Log the classification report
    if output_report_filepath :
        with open(output_report_filepath,'w') as outfile:
            outfile.write(f'Classification Report:\n {report}')
    logger.info(f"[TRAIN] Classification Report:\n {report}")
    logger.info("[TRAIN] Model evaluation completed successfully.")

    return report

def export_model(pipeline: Pipeline, model_path:str=None) -> None:
    """
    Saves a trained machine learning pipeline to a file for deployment.

    Args:
        pipeline (Pipeline): A trained scikit-learn pipeline to be saved.
        model_path (str): Optional, where to write the joblib file

    Returns:
        None
    """
    # Retrieve the model save path from the deployment configuration
    if model_path is None:
        model_filename=f'{datetime.now()}_{train_config["export"]["model_filename"]}'
        model_path = os.path.join(train_config["export"]['output_folder'],model_filename)

    # Save the trained pipeline to a file
    joblib.dump(pipeline, model_path)

    # Log the save operation
    logger.info(f"[TRAIN] Pipeline saved to '{model_path}'")

if __name__=='__main__':
    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d%H%M")
    #Prepare dataset
    x_traintest, x_val, y_traintest, y_val = data_preparation("stages-votes.json")
    raw_pipeline = build_pipeline(max_k=len(x_traintest))
    logger.info('[SKLEARN] build pipeline')

    #CROSS validate model choice and hyperparameter
    #Grid search can be perfomed here for hyper parameter optimization
    output_eval_filename = f"{formatted_time}_NP_Evaluation_ROC.png"
    output_eval_filepath =os.path.join(train_config["export"]['root_dir'],
                                       train_config["export"]['output_folder'],
                                       output_eval_filename)
    kfold_pipeline_validation(raw_pipeline,x_traintest,y_traintest,output_eval_filepath)
    logger.info('[SKLEARN] KFOLD validation performed')

    #Train final pipeline on full data
    trained_pipeline = train_pipeline(raw_pipeline, x_traintest, y_traintest)
    logger.info('[SKLEARN] final training performed')

    #Perfom evaluation on validation subset
    output_report_filename = f"{formatted_time}_NP_classification_report.txt"
    output_report_filepath =os.path.join(train_config["export"]['root_dir'],
                                         train_config["export"]['output_folder'],
                                         output_report_filename)
    test_pipeline(trained_pipeline, x_val, y_val,output_report_filepath)

    logger.info('[SKLEARN] final test performed, classifcation report in log file')
    #Export model as joblib file
    output_model_filename = f"{formatted_time}_NP_pipeline_model.joblib"
    output_model_filepath = os.path.join(train_config["export"]['root_dir'],
                                         train_config["export"]['output_folder'],
                                         output_model_filename)
    export_model(trained_pipeline,output_model_filepath)
    #Also export last model for APItree
    output_model_lastfilepath = os.path.join(train_config["export"]['src_dir'],'models',
                                         train_config["export"]['model_filename'])
    export_model(trained_pipeline,output_model_lastfilepath)
    logger.info('[SKLEARN] end of process')


