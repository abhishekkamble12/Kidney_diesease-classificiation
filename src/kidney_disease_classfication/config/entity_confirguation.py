from kidney_disease_classfication.constants import *
from kidney_disease_classfication.utils.common import read_yaml, create_directories
from kidney_disease_classfication.config.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            source_type=config.source_type
        )

        return data_ingestion_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            test_size=config.test_size,
            random_state=config.random_state
        )

        return data_transformation_config

    def get_model_trainer_config(self, model_name: str) -> ModelTrainerConfig:
        config = self.config.model_trainer
        data_transform_config = self.config.data_transformation
        
        # Load hyperparameters dynamically for the selected model
        hyperparams = dict(self.params.get(model_name, {}))
        random_state = int(self.params.get("random_state", 42))
        max_features = int(self.params.get("max_features", 15000))

        create_directories([config.root_dir])

        # Generate a model-specific artifact path to prevent overwriting
        trained_model_path = Path(config.root_dir) / f"model_{model_name}.pkl"

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            trained_model_path=trained_model_path,
            train_data_path=data_transform_config.train_data_path,
            test_data_path=data_transform_config.test_data_path,
            model_name=model_name,
            random_state=random_state,
            max_features=max_features,
            hyperparams=hyperparams
        )

        return model_trainer_config

    def get_model_evaluation_config(self, model_name: str) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        
        # Load hyperparameters dynamically for the selected model
        hyperparams = dict(self.params.get(model_name, {}))
        random_state = int(self.params.get("random_state", 42))

        create_directories([config.root_dir])

        # Generate a model-specific loaded path mapping to the trainer's output
        model_path = Path(self.config.model_trainer.root_dir) / f"model_{model_name}.pkl"

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=model_path,
            metrics_file=config.metrics_file,
            threshold=config.threshold,
            stage=config.stage,
            model_name=model_name,
            random_state=random_state,
            hyperparams=hyperparams
        )

        return model_evaluation_config