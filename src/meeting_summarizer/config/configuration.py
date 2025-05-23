import os
from pathlib import Path
from src.meeting_summarizer.constants import *
from src.meeting_summarizer.utils.common import read_yaml, create_directories
from src.meeting_summarizer.entity.config_entity import (DataIngestionConfig, 
                                                         DataPreprocessingConfig)

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
            root_dir=Path(config.root_dir),
            source_dataset=config.source_dataset,
            train_data_path=Path(config.train_data_path),
            validation_data_path=Path(config.validation_data_path),
            test_data_path=Path(config.test_data_path),
            use_vector_db=config.use_vector_db,
            vector_db_path=Path(config.vector_db_path),
            embedding_model_name=config.embedding_model_name
        )

        return data_ingestion_config

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing

        create_directories([config.root_dir])

        # Get the actual paths (not the template strings)
        train_path = Path(os.path.expandvars(config.train_data_path))
        val_path = Path(os.path.expandvars(config.validation_data_path))
        test_path = Path(os.path.expandvars(config.test_data_path))

        # In your configuration.py
        # In your configuration.py
        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            validation_data_path=Path(config.validation_data_path),
            test_data_path=Path(config.test_data_path),
            model_name=config.model_name,
            max_input_length=int(config.get('max_input_length', 512)),  # Default fallback
            max_target_length=int(config.get('max_target_length', 128))  # Default fallback
        )

        return data_preprocessing_config