import os
import pandas as pd
from pathlib import Path
from kidney_disease_classfication import logger
from kidney_disease_classfication.config.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def split_data(self, ingested_data_path: Path):
        """
        Splits the ingested CSV dataset into train and test sets using pandas,
        and saves them to the configured paths.
        """
        try:
            logger.info(f"Reading ingested dataset from {ingested_data_path} for train-test split")
            df = pd.read_csv(ingested_data_path)
            
            # Perform train-test split using pandas sample
            test_size = self.config.test_size
            random_state = self.config.random_state
            
            logger.info(f"Splitting data with test_size={test_size} and random_state={random_state}")
            train_df = df.sample(frac=1.0 - test_size, random_state=random_state)
            test_df = df.drop(train_df.index)
            
            # Save split data to target CSV files
            train_path = Path(self.config.train_data_path)
            test_path = Path(self.config.test_data_path)
            
            # Ensure directories for the output paths exist
            os.makedirs(train_path.parent, exist_ok=True)
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            logger.info(f"Successfully split dataset into train and test sets")
            logger.info(f"Train set shape: {train_df.shape} saved at {train_path}")
            logger.info(f"Test set shape: {test_df.shape} saved at {test_path}")
            
        except Exception as e:
            logger.exception(f"Error during train-test split: {e}")
            raise e
