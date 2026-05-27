import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tweet_sentiment_classifier import logger
from tweet_sentiment_classifier.config.configuration import ConfigurationManager
from tweet_sentiment_classifier.components import (
    DataIngestion,
    DataValidation,
    DataTransformation,
    ModelTrainer,
    ModelEvaluation,
    BestModelSelector
)

try:
    config_manager = ConfigurationManager()

    # --- DATA INGESTION ---
    logger.info(">>>>>> Stage 1: Data Ingestion started <<<<<<")
    data_ingestion_config = config_manager.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    local_data_path = data_ingestion.download_file()
    data_ingestion.extract_zip_file(local_data_path)
    logger.info(">>>>>> Stage 1: Data Ingestion completed successfully <<<<<<\n\nx==========x")

    # --- DATA VALIDATION ---
    logger.info(">>>>>> Stage 2: Data Validation started <<<<<<")
    data_validation_config = config_manager.get_data_validation_config()
    data_validation = DataValidation(config=data_validation_config)
    status = data_validation.validate_all_data()
    if not status:
        raise Exception("Data Validation Failed. Check status file or logs.")
    logger.info(">>>>>> Stage 2: Data Validation completed successfully <<<<<<\n\nx==========x")

    # --- DATA TRANSFORMATION ---
    logger.info(">>>>>> Stage 3: Data Transformation started <<<<<<")
    data_transformation_config = config_manager.get_data_transformation_config()
    data_transformation = DataTransformation(config=data_transformation_config)
    data_transformation.transform_and_split()
    logger.info(">>>>>> Stage 3: Data Transformation completed successfully <<<<<<\n\nx==========x")

    # --- MODEL ITERATIVE TRAINING AND EVALUATION LOOP ---
    models_to_train = config_manager.params.get("models_to_train", ["LogisticRegression"])
    logger.info(f"Retrieved list of models to train: {models_to_train}")

    for idx, model_name in enumerate(models_to_train, 1):
        logger.info(f"===========================================================")
        logger.info(f"Executing Model {idx}/{len(models_to_train)}: {model_name}")
        logger.info(f"===========================================================")

        # Model Training Stage
        logger.info(f">>>>>> Stage 4a: Model Training - {model_name} started <<<<<<")
        model_trainer_config = config_manager.get_model_trainer_config(model_name)
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()
        logger.info(f">>>>>> Stage 4a: Model Training - {model_name} completed successfully <<<<<<")

        # Model Evaluation Stage
        logger.info(f">>>>>> Stage 4b: Model Evaluation - {model_name} started <<<<<<")
        model_evaluation_config = config_manager.get_model_evaluation_config(model_name)
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.log_into_mlflow()
        logger.info(f">>>>>> Stage 4b: Model Evaluation - {model_name} completed successfully <<<<<<\n\nx==========x")

    # --- BEST MODEL SELECTION ---
    logger.info(">>>>>> Stage 5: Best Model Selection started <<<<<<")
    best_model_config = config_manager.get_best_model_config()
    best_model_selector = BestModelSelector(config=best_model_config)
    best_model_selector.select_best_model()
    logger.info(">>>>>> Stage 5: Best Model Selection completed successfully <<<<<<\n\nx==========x")

    logger.info(">>>>>> Complete Multi-Model Sentiment Classification Pipeline completed successfully! <<<<<<")

except Exception as e:
    logger.exception(e)
    raise e