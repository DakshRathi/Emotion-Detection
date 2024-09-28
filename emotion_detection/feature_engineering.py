import numpy as np
import pandas as pd
import yaml
import logging
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from utils import Logger, load_data, load_params, save_data

# Initialize logger
logger = Logger('feature_engineering', logging.INFO)


# Create Bag of Words representation
def create_bow(X: np.ndarray, max_features: int) -> np.ndarray:
    """
    Create a Bag of Words representation of the input data.

    Parameters
    ----------
    X : np.ndarray
        The input data (text).
    max_features : int
        The maximum number of features to consider.

    Returns
    -------
    np.ndarray
        The Bag of Words representation.
    """
    try:
        logger.info("Creating Bag of Words representation.")
        vectorizer = CountVectorizer(max_features=max_features)
        return vectorizer.fit_transform(X)
    except Exception as e:
        logger.error(f"Error creating Bag of Words: {e}")
        raise


# Main function to execute the feature engineering pipeline
def main() -> None:
    """
    Execute the feature engineering pipeline.

    Returns
    -------
    None
    """
    try:
        # Load parameters
        params = load_params(Path('params.yaml'), logger)['feature_engineering']

        # Load the data
        train_data = load_data(Path('./data/processed/train_processed.csv'), logger)
        test_data = load_data(Path('./data/processed/test_processed.csv'), logger)

        # Fill missing values
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)

        # Prepare input features and labels
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        # Create Bag of Words
        X_train_bow = create_bow(X_train, params['max_features'])
        X_test_bow = CountVectorizer(max_features=params['max_features']).fit(X_train).transform(X_test)

        # Create DataFrames for training and testing data
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        # Create data/features directory
        data_path = Path("data") / "features"
        data_path.mkdir(parents=True, exist_ok=True)

        # Save the processed data
        save_data(train_df, data_path, "train_bow.csv", logger)
        save_data(test_df, data_path, "test_bow.csv", logger)

        logger.info("Feature engineering pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Feature engineering pipeline failed: {str(e)}")
        raise

# Run the script
if __name__ == "__main__":
    main()