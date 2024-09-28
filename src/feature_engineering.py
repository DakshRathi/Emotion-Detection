import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import Logger, load_data, load_params, save_data

# Initialize logger
logger = Logger('feature_engineering', logging.INFO)


# Create TF-IDF representation
def create_tfidf_dataframe(X_train: np.ndarray, X_test: np.ndarray, max_features: int) -> tuple:
    """
    Create a TF-IDF representation for both training and testing datasets, returning fitted DataFrames.

    Parameters
    ----------
    X_train : np.ndarray
        The input training data (text).
    X_test : np.ndarray
        The input testing data (text).
    max_features : int
        The maximum number of features to consider.

    Returns
    -------
    tuple
        A tuple containing:
        - The fitted train DataFrame with TF-IDF features.
        - The transformed test DataFrame with the same TF-IDF features as the training data.
    """
    try:
        logger.info("Creating TF-IDF representation for train and test data.")
        
        # Initialize the vectorizer
        vectorizer = TfidfVectorizer(max_features=max_features)

        # Fit on the training data and transform both train and test datasets
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Get feature names to use as column names
        feature_names = vectorizer.get_feature_names_out()

        # Convert to pandas DataFrames
        X_train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=feature_names)
        X_test_df = pd.DataFrame(X_test_tfidf.toarray(), columns=feature_names)

        logger.info("TF-IDF transformation complete.")
        return X_train_df, X_test_df
    except Exception as e:
        logger.error(f"Error creating TF-IDF DataFrame: {e}")
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
        train_data = load_data(Path('./data/interim/train_processed.csv'), logger)
        test_data = load_data(Path('./data/interim/test_processed.csv'), logger)

        # Fill missing values
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)

        # Prepare input features and labels
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

       # Create TF-IDF DataFrames for both train and test data
        train_df, test_df = create_tfidf_dataframe(X_train, X_test, params['max_features'])
        train_df['label'] = y_train
        test_df['label'] = y_test

        # Create data/features directory
        data_path = Path("data") / "processed"
        data_path.mkdir(parents=True, exist_ok=True)

        # Save the processed data
        save_data(train_df, data_path, "train_tfidf.csv", logger)
        save_data(test_df, data_path, "test_tfidf.csv", logger)

        logger.info("Feature engineering pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Feature engineering pipeline failed: {str(e)}")
        raise

# Run the script
if __name__ == "__main__":
    main()