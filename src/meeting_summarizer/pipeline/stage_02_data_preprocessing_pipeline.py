from src.meeting_summarizer.config.configuration import ConfigurationManager
from src.meeting_summarizer.components.data_preprocessing import DataPreprocessing
from src.meeting_summarizer import logger
from typing import Dict

class DataPreprocessingPipeline:
    def __init__(self):
        self.stage_name = "Data Preprocessing Pipeline"
    
    def main(self) -> Dict[str, str]:
        try:
            logger.info(f">>>>>> {self.stage_name} started <<<<<<")
            
            config = ConfigurationManager()
            data_preprocessing_config = config.get_data_preprocessing_config()
            
            # Check if input files exist
            for path in [data_preprocessing_config.train_data_path,
                        data_preprocessing_config.validation_data_path,
                        data_preprocessing_config.test_data_path]:
                if not path.exists():
                    raise FileNotFoundError(
                        f"Input file {path} not found. Run data ingestion first."
                    )
            
            data_preprocessor = DataPreprocessing(config=data_preprocessing_config)
            preprocessed_data_paths = data_preprocessor.initiate_data_preprocessing()
            
            logger.info(f">>>>>> {self.stage_name} completed <<<<<<")
            return preprocessed_data_paths
            
        except Exception as e:
            logger.exception(f"{self.stage_name} failed with exception: {e}")
            raise e