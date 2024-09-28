import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from joblib import load
from utils import Logger, load_data

# Initialize logger
logger = Logger('model_evaluation', logging.INFO)

def load_model(file_path: Path):
    """
    Load a trained model from a file.

    Parameters
    ----------
    file_path : Path
        The path to the model file.

    Returns
    -------
    Trained model
        The loaded model.
    """
    try:
        logger.info(f"Loading model from {file_path}")
        return load(file_path)
    except FileNotFoundError:
        logger.error(f"Model file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load model from {file_path}: {e}")
        raise

from sklearn.preprocessing import label_binarize  # Import this for multiclass AUC

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the trained model using test data.

    Parameters
    ----------
    clf : Trained model
        The trained classifier model.
    X_test : np.ndarray
        The test features.
    y_test : np.ndarray
        The test labels.

    Returns
    -------
    dict
        A dictionary of evaluation metrics.
    """
    try:
        logger.info("Making predictions on the test data.")
        y_pred = clf.predict(X_test)

        # Check if the model has predict_proba method
        if hasattr(clf, "predict_proba"):
            y_pred_proba = clf.predict_proba(X_test)  # Get probabilities for all classes
        else:
            logger.warning("The model does not support probability predictions. AUC will not be computed.")
            y_pred_proba = None

        # Calculate evaluation metrics
        metrics_dict = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),  # Change made here
            'recall': recall_score(y_test, y_pred, average='weighted'),        # Change made here
            'auc': None  # Default to None; we will set this if applicable
        }

        # Calculate AUC if y_pred_proba is not None
        if y_pred_proba is not None:
            if len(np.unique(y_test)) == 2:  # Binary classification
                metrics_dict['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:  # Multiclass classification
                # Binarize the output
                y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
                metrics_dict['auc'] = roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr')

        logger.info("Evaluation metrics calculated successfully.")
        return metrics_dict
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def save_metrics(metrics: dict, file_path: Path) -> None:
    """
    Save evaluation metrics to a JSON file.

    Parameters
    ----------
    metrics : dict
        The metrics to save.
    file_path : Path
        The path where the metrics will be saved.

    Returns
    -------
    None
    """
    try:
        logger.info(f"Saving metrics to {file_path}")
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
    except Exception as e:
        logger.error(f"Error saving metrics to {file_path}: {e}")
        raise

def main() -> None:
    """
    Execute the model evaluation pipeline.

    Returns
    -------
    None
    """
    try:
        # Load the trained model
        clf = load_model(Path('model.joblib'))

        # Load the test data
        test_data = load_data(Path('./data/features/test_bow.csv'), logger)

        # Prepare input features and labels
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        # Evaluate the model
        metrics = evaluate_model(clf, X_test, y_test)

        # Save the evaluation metrics
        save_metrics(metrics, Path('metrics.json'))

        logger.info("Model evaluation pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Model evaluation pipeline failed: {str(e)}")
        raise

# Run the script
if __name__ == "__main__":
    main()