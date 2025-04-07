from src.meeting_summarizer import logger
from src.meeting_summarizer.pipeline.stage_01_data_ingestion_pipeline import DataIngestionPipeline
from src.meeting_summarizer.pipeline.stage_02_data_preprocessing_pipeline import DataPreprocessingPipeline

# STAGE_NAME = "Data Ingestion stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = DataIngestionPipeline()
#    data_ingestion.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e

STAGE_NAME = "Data Preprocessing stage"
try:
   logger.info(">>>>>> stage {STAGE_NAME} started <<<<<<")
   obj = DataPreprocessingPipeline()
   preprocessed_data_paths = obj.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
   logger.info(f"Preprocessed data paths: {preprocessed_data_paths}")
except Exception as e:
   logger.exception(e)
   raise e