import numpy as np
import pandas as pd
import re
import nltk
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from utils import Logger, load_data, save_data

# Initialize logger
logger = Logger('data_preprocessing', logging.INFO)


# Transform the data
def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by applying various cleaning operations.

    Parameters
    ----------
    text : str
        The input text to preprocess.

    Returns
    -------
    str
        The cleaned and preprocessed text.
    """
    try:
        text = lower_case(text)
        text = remove_stop_words(text)
        text = removing_numbers(text)
        text = removing_punctuations(text)
        text = removing_urls(text)
        text = lemmatization(text)
        return text
    except Exception as e:
        logger.error(f"Error in preprocessing text: {e}")
        raise

def lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    return " ".join([i for i in text.split() if i not in stop_words])

def removing_numbers(text: str) -> str:
    return ''.join([i for i in text if not i.isdigit()])

def lower_case(text: str) -> str:
    return text.lower()

def removing_punctuations(text: str) -> str:
    """
    Remove punctuation from the input text.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    str
        The text without punctuation.
    """
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

def removing_urls(text: str) -> str:
    """
    Remove URLs from the input text.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    str
        The text without URLs.
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df: pd.DataFrame) -> None:
    """
    Remove sentences that are too short.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the text data.

    Returns
    -------
    None
    """
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the text in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the text data.

    Returns
    -------
    pd.DataFrame
        The DataFrame with normalized text.
    """
    try:
        logger.info("Normalizing text data.")
        df.content = df.content.apply(preprocess_text)
        return df
    except Exception as e:
        logger.error(f"Error normalizing text: {e}")
        raise

      
# Main function to execute the processing pipeline
def main() -> None:
    """
    Execute the data processing pipeline.

    Returns
    -------
    None
    """
    try:
        # Load the data
        train_data = load_data(Path('./data/raw/train.csv'), logger)
        test_data = load_data(Path('./data/raw/test.csv'), logger)

        # Normalize the text
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Create data/processed directory
        data_path = Path("data") / "interim"
        data_path.mkdir(parents=True, exist_ok=True)

        # Save the processed data
        save_data(train_processed_data, data_path, "train_processed.csv", logger)
        save_data(test_processed_data, data_path, "test_processed.csv", logger)

        logger.info("Data processing pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Data processing pipeline failed: {str(e)}")
        raise

# Run the script
if __name__ == "__main__":
    nltk.download('wordnet')
    nltk.download('stopwords')
    main()