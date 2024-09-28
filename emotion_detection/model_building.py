import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from utils import Logger, load_params, load_data

# Initialize logger
logger = Logger('model_building', logging.INFO)


def train_model(X: np.ndarray, y: np.ndarray, params: dict) -> GradientBoostingClassifier:
    """
    Train a Gradient Boosting Classifier model.

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
    GradientBoostingClassifier
        The trained model.
    """
    try:
        logger.info("Defining and training the Gradient Boosting model.")
        clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
        clf.fit(X, y)
        return clf
    except Exception as e:
        logger.error(f"Error training the model: {e}")
        raise

def save_model(model: GradientBoostingClassifier, file_path: Path) -> None:
    """
    Save the trained model to a file.

    Parameters
    ----------
    model : GradientBoostingClassifier
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
        train_data = load_data(Path('./data/features/train_bow.csv'), logger)

        # Prepare input features and labels
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        # Train the model
        clf = train_model(X_train, y_train, params)

        # Save the trained model
        save_model(clf, Path('model.joblib'))

        logger.info("Model building pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Model building pipeline failed: {str(e)}")
        raise

# Run the script
if __name__ == "__main__":
    main()