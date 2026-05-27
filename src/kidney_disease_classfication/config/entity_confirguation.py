from kidney_disease_classfication.constants import *
from kidney_disease_classfication.utils.common import read_yaml, create_directories
from kidney_disease_classfication.config.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)

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

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        data_transform_config = self.config.data_transformation
        params = self.params

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            trained_model_path=config.trained_model_path,
            train_data_path=data_transform_config.train_data_path,
            test_data_path=data_transform_config.test_data_path,
            model_name=params.model_name,
            epochs=int(params.epochs),
            batch_size=int(params.batch_size),
            learning_rate=float(params.learning_rate),
            max_features=int(params.max_features),
            embedding_dim=int(params.embedding_dim),
            max_len=int(params.max_len)
        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            metrics_file=config.metrics_file,
            threshold=config.threshold,
            stage=config.stage,
            model_name=params.model_name
        )

        return model_evaluation_config