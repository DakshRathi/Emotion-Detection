import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from typing import Tuple, Dict
from pathlib import Path
import logging
from utils import Logger, load_params, save_data

logger = Logger('data_ingestion', logging.INFO)

# Function to load the dataset from a URL
def load_data(url: str) -> pd.DataFrame:
    """
    Load the dataset from a URL.

    Parameters
    ----------
    url : str
        The URL to load the dataset from.

    Returns
    -------
    pd.DataFrame
        The loaded dataset as a pandas DataFrame.
    """
    try:
        logger.info(f"Loading dataset from {url}")
        df = pd.read_csv(url)
        logger.debug(f"Dataset loaded successfully with {len(df)} records.")
        return df
    except pd.errors.EmptyDataError as e:
        logger.error(f"No data found at the URL: {url}")
        raise e
    except Exception as e:
        logger.error(f"Failed to load dataset from {url}: {str(e)}")
        raise e

# Function to preprocess the dataset
def preprocess_data(
    df: pd.DataFrame, 
    columns_to_drop: list, 
    target_column: str, 
    target_mapping: Dict[str, int], 
    sentiments_to_keep: list
) -> pd.DataFrame:
    """
    Preprocess the dataset by dropping columns and filtering sentiments.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to preprocess.
    columns_to_drop : list
        The columns to drop from the dataset.
    target_column : str
        The column containing the target variable.
    target_mapping : Dict[str, int]
        Mapping of sentiment labels to numerical values.
    sentiments_to_keep : list
        The sentiments to keep in the dataset.

    Returns
    -------
    pd.DataFrame
        The preprocessed dataset.
    """
    try:
        logger.info("Preprocessing data")
        logger.debug(f"Dropping columns {columns_to_drop} and filtering sentiments {sentiments_to_keep}")
        df = df.drop(columns=columns_to_drop)
        final_df = df[df[target_column].isin(sentiments_to_keep)]
        final_df[target_column].replace(target_mapping, inplace=True)
        logger.info("Data preprocessing completed successfully.")
        logger.debug(f"Final dataframe shape: {final_df.shape}")
        return final_df
    except KeyError as e:
        logger.error(f"Error in preprocessing: {e}")
        raise e

# Function to split the dataset into train and test sets
def split_data(
    df: pd.DataFrame, 
    test_size: float, 
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to split.
    test_size : float
        The proportion of the dataset to include in the test set.
    random_state : int
        The random state to set for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The train and test sets, respectively.
    """
    logger.debug(f"Splitting data with test size {test_size} and random state {random_state}")
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
    logger.debug(f"Data split into {len(train_data)} train records and {len(test_data)} test records.")
    return train_data, test_data


# Main function to execute the pipeline
def main(params_path: str) -> None:
    """
    Execute the data ingestion pipeline.

    The pipeline involves loading parameters, loading the data, preprocessing the data,
    splitting the data into train and test sets, and saving the datasets to a given directory.

    Parameters
    ----------
    params_path : str
        The path to the YAML file containing the parameters for the pipeline.

    Returns
    -------
    None
    """
    try:
        logger.info("Starting data ingestion pipeline...")
        # Load parameters
        params = load_params(params_path, logger)['data_ingestion']

        # Load the data
        url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
        df = load_data(url)

        # Preprocess the data
        columns_to_drop = ['tweet_id']
        target_column = 'sentiment'
        target_mapping = {'neutral': 1, 'sadness': 0}
        sentiments_to_keep = ['neutral', 'sadness']
        
        final_df = preprocess_data(df, columns_to_drop, target_column, target_mapping, sentiments_to_keep)

        # Split the data into train and test sets
        test_size = params['test_size']
        random_state = 42
        train_data, test_data = split_data(final_df, test_size, random_state)

        # Save the datasets
        data_path = Path("data") / "raw"
        save_data(train_data, data_path, "train.csv", logger)
        save_data(test_data, data_path, "test.csv", logger)

        logger.info("Data ingestion pipeline completed successfully.")

    except Exception as e:
        logger.critical(f"Data ingestion pipeline failed: {str(e)}")
        raise e

# Run the script
if __name__ == "__main__":
    main('params.yaml')