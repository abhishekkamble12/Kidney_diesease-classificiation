import sys
import os
# Insert 'src' to system path to enable absolute imports from kidney_disease_classfication directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from kidney_disease_classfication import logger
from kidney_disease_classfication.config.entity_confirguation import ConfigurationManager
from kidney_disease_classfication.components.data_ingestion import DataIngestion
from kidney_disease_classfication.components.data_transformation import DataTransformation
from kidney_disease_classfication.components.model_trainer import ModelTrainer
from kidney_disease_classfication.components.model_evaluation import ModelEvaluation

STAGE_1_NAME = "Data Ingestion Stage"
STAGE_2_NAME = "Data Transformation Stage"
STAGE_3_NAME = "Model Training Stage"
STAGE_4_NAME = "Model Evaluation Stage"

try:
    config_manager = ConfigurationManager()

    # --- DATA INGESTION ---
    logger.info(f">>>>>> Stage {STAGE_1_NAME} started <<<<<<")
    data_ingestion_config = config_manager.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    local_data_path = data_ingestion.download_file()
    data_ingestion.extract_zip_file(local_data_path)
    logger.info(f">>>>>> Stage {STAGE_1_NAME} completed successfully <<<<<<\n\nx==========x")

    # --- DATA TRANSFORMATION ---
    logger.info(f">>>>>> Stage {STAGE_2_NAME} started <<<<<<")
    data_transformation_config = config_manager.get_data_transformation_config()
    data_transformation = DataTransformation(config=data_transformation_config)
    data_transformation.split_data(ingested_data_path=local_data_path)
    logger.info(f">>>>>> Stage {STAGE_2_NAME} completed successfully <<<<<<\n\nx==========x")

    # --- MODEL TRAINING ---
    logger.info(f">>>>>> Stage {STAGE_3_NAME} started <<<<<<")
    model_trainer_config = config_manager.get_model_trainer_config()
    model_trainer = ModelTrainer(config=model_trainer_config)
    model_trainer.train()
    logger.info(f">>>>>> Stage {STAGE_3_NAME} completed successfully <<<<<<\n\nx==========x")

    # --- MODEL EVALUATION ---
    logger.info(f">>>>>> Stage {STAGE_4_NAME} started <<<<<<")
    model_evaluation_config = config_manager.get_model_evaluation_config()
    model_evaluation = ModelEvaluation(config=model_evaluation_config)
    model_evaluation.log_into_mlflow()
    logger.info(f">>>>>> Stage {STAGE_4_NAME} completed successfully <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e