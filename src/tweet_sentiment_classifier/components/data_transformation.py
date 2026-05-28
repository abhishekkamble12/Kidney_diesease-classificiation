import os
import re
import pandas as pd
from pathlib import Path
from tweet_sentiment_classifier import logger
# import nltk
from tweet_sentiment_classifier.config.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def transform_and_split(self):
        try:
            logger.info(f"Reading dataset from {self.config.data_path}")
            df = pd.read_csv(self.config.data_path)
            
            logger.info("Applying text cleaning (lowercase, URLs, punctuation)")
            df['text'] = df['text'].apply(self.clean_text)
            
            # Remove empty texts after cleaning
            df = df[df['text'] != ""]
            
            test_size = self.config.test_size
            random_state = self.config.random_state
            
            logger.info(f"Splitting data with test_size={test_size} and random_state={random_state}")
            train_df = df.sample(frac=1.0 - test_size, random_state=random_state)
            test_df = df.drop(train_df.index)
            
            train_path = Path(self.config.train_data_path)
            test_path = Path(self.config.test_data_path)
            
            os.makedirs(train_path.parent, exist_ok=True)
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            logger.info(f"Successfully transformed and split dataset")
            logger.info(f"Train set shape: {train_df.shape} saved at {train_path}")
            logger.info(f"Test set shape: {test_df.shape} saved at {test_path}")
            
        except Exception as e:
            logger.exception(f"Error during data transformation: {e}")
            raise e
