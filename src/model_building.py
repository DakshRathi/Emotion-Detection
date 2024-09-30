import numpy as np
import joblib
import logging
from pathlib import Path
from lightgbm import LGBMClassifier
from utils import Logger, load_params, load_data

# Initialize logger
logger = Logger('model_building', logging.INFO)


def train_model(X: np.ndarray, y: np.ndarray, params: dict) -> LGBMClassifier:
    """
    Train a LightGBM Classifier model.

    Parameters
    ----------
    X : np.ndarray
        The training features.
    y : np.ndarray
        The training labels.
    params : dict
        The parameters for the model.

    Returns
    -------
    LGBMClassifier
        The trained LightGBM model.
    """
    try:
        logger.info("Defining and training the LightGBM model.")
        
        # Define the LightGBM model with additional parameters
        clf = LGBMClassifier(
            n_estimators=params.get('n_estimators', 100),        # Number of boosting rounds
            learning_rate=params.get('learning_rate', 0.1),      # Learning rate
            max_depth=params.get('max_depth', -1),               # Max depth of trees (-1 means no limit)
            num_leaves=params.get('num_leaves', 31),             # Maximum number of leaves in one tree
            min_data_in_leaf=params.get('min_data_in_leaf', 20), # Minimum number of samples in a leaf
            boosting_type=params['boosting_type'],   # Boosting type (gbdt, dart, goss)
            subsample=params.get('subsample', 1.0),              # Fraction of data used for fitting each tree
            colsample_bytree=params.get('colsample_bytree', 1.0),# Fraction of features used for fitting each tree
            reg_lambda=params.get('reg_lambda', 0.0),            # L2 regularization term
            reg_alpha=params.get('reg_alpha', 0.0),              # L1 regularization term
            random_state=params.get('random_state', 42)          # Random state for reproducibility
        )
        
        # Train the model
        clf.fit(X, y)
        
        return clf
    except Exception as e:
        logger.error(f"Error training the model: {e}")
        raise

def save_model(model: LGBMClassifier, file_path: Path) -> None:
    """
    Save the trained model to a file.

    Parameters
    ----------
    model : LGBMClassifier
        The trained model.
    file_path : Path
        The path where the model will be saved.

    Returns
    -------
    None
    """
    try:
        logger.info(f"Saving the trained model to {file_path}")
        with open(file_path, 'wb') as file:
            joblib.dump(model, file)
    except Exception as e:
        logger.error(f"Error saving the model to {file_path}: {e}")
        raise

def main() -> None:
    """
    Execute the model building pipeline.

    Returns
    -------
    None
    """
    try:
        # Load parameters
        params = load_params(Path('params.yaml'), logger)['model_building']

        # Load the training data
        train_data = load_data(Path('./data/processed/train_tfidf.csv'), logger)

        # Prepare input features and labels
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        # Train the model
        clf = train_model(X_train, y_train, params)

        # Save the trained model
        save_model(clf, Path('models/model.joblib'))

        logger.info("Model building pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Model building pipeline failed: {str(e)}")
        raise

# Run the script
if __name__ == "__main__":
    main()