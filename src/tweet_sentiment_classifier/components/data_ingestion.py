import os
import urllib.request as request
import zipfile
import shutil
from pathlib import Path
from tweet_sentiment_classifier import logger
from tweet_sentiment_classifier.utils.common import get_size
from tweet_sentiment_classifier.config.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> Path:
        """
        Fetches the dataset file from either a remote URL or a local path.
        """
        local_data_file_path = Path(self.config.local_data_file)
        if len(local_data_file_path.parts) == 1:
            local_data_path = Path(self.config.root_dir) / local_data_file_path
        else:
            local_data_path = local_data_file_path

        if not os.path.exists(local_data_path):
            source_url = self.config.source_URL
            
            # Check if source_url is a remote URL or local path
            if source_url.startswith("http://") or source_url.startswith("https://"):
                logger.info(f"Downloading dataset from remote URL: {source_url} to {local_data_path}")
                filename, headers = request.urlretrieve(
                    url=source_url,
                    filename=str(local_data_path)
                )
                logger.info(f"{filename} downloaded successfully with following info: \n{headers}")
            else:
                # Treat as local path
                logger.info(f"Copying dataset from local path: {source_url} to {local_data_path}")
                source_path = Path(source_url)
                if source_path.exists():
                    shutil.copy2(source_path, local_data_path)
                    logger.info(f"Dataset copied successfully to {local_data_path}")
                else:
                    raise FileNotFoundError(f"Local source dataset file not found at: {source_path}")
        else:
            logger.info(f"File already exists at: {local_data_path} (size: {get_size(Path(local_data_path))})")

        return local_data_path

    def extract_zip_file(self, local_data_path: Path):
        """
        Extracts the zip file into the unzip directory.
        """
        unzip_path = Path(self.config.unzip_dir)
        os.makedirs(unzip_path, exist_ok=True)
        
        # Check if it is a zip file
        if zipfile.is_zipfile(local_data_path):
            logger.info(f"Extracting zip archive {local_data_path} to {unzip_path}")
            with zipfile.ZipFile(local_data_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"Zip extraction completed successfully to {unzip_path}")
        else:
            logger.info(f"File {local_data_path} is not a zip archive. Bypassing unzip step.")