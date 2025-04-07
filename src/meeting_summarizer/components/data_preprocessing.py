import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoTokenizer

from src.meeting_summarizer.entity.config_entity import DataPreprocessingConfig
from src.meeting_summarizer import logger
from src.meeting_summarizer.utils.common import create_directories

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        """
        Initialize data preprocessing component with configuration.
        
        Args:
            config (DataPreprocessingConfig): Configuration parameters
        """
        self.config = config
        self.tokenizer = self._initialize_tokenizer()
        
    def _initialize_tokenizer(self) -> AutoTokenizer:
        try:
            logger.info(f"Initializing tokenizer from {self.config.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="right",
                truncation_side="right"
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            return tokenizer
        except Exception as e:
            logger.error(f"Tokenizer initialization failed: {e}")
            raise e  # Ensure the exception is raised to notify the caller of the failure

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load train, validation, and test datasets from CSV files.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing dataframes for each split
        """
        logger.info("Loading datasets from CSV files")
        data_splits = {}

        required_columns = {'dialogue', 'summary'}
        
        try:
             for split in ['train', 'validation', 'test']:
                file_path = getattr(self.config, f"{split}_data_path")
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Required {split} data file not found: {file_path}")
                
                df = pd.read_csv(file_path)
                
                # Validate columns
                if not required_columns.issubset(df.columns):
                    missing = required_columns - set(df.columns)
                    raise ValueError(f"{split} data missing required columns: {missing}")
                
                data_splits[split] = df
                logger.info(f"Loaded {split} data with {len(df)} examples")
                
             return data_splits
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise e

    def _tokenize_text(self, texts: list, is_target: bool = False, 
                  max_length: int = None) -> Dict[str, np.ndarray]:
        """
        Tokenize text sequences with proper parameter handling for TinyLlama
        """
        try:
            # For TinyLlama tokenizer, we need to call it directly rather than through as_target_tokenizer
            if is_target:
                # For target sequences, we need to handle this differently
                return self.tokenizer(
                    text=texts,
                    max_length=max_length or self.config.max_target_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="np",
                    add_special_tokens=True
                )
            else:
                # For input sequences
                return self.tokenizer(
                    text=texts,
                    max_length=max_length or self.config.max_input_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="np"
                )
                
        except Exception as e:
            logger.error(f"Tokenization failed for sample text: {texts[0] if texts else 'Empty input'}")
            logger.error(f"Tokenization parameters - max_length: {max_length}, is_target: {is_target}")
            raise ValueError(f"Tokenization error: {str(e)}") from e

    def preprocess_data(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Preprocess datasets by tokenizing dialogues and summaries.
        """
        logger.info("Preprocessing datasets")
        processed_data = {}
        
        try:
            for split, df in dataframes.items():
                logger.info(f"Processing {split} split with {len(df)} examples")
                
                # Clean and validate data
                df['dialogue'] = df['dialogue'].astype(str).str.strip()
                df['summary'] = df['summary'].astype(str).str.strip()
                
                # Filter out empty strings
                df = df[(df['dialogue'].str.len() > 0) & (df['summary'].str.len() > 0)]
                
                if len(df) == 0:
                    raise ValueError(f"No valid data in {split} split after cleaning")
                
                # Tokenize dialogues
                input_encodings = self._tokenize_text(
                    df['dialogue'].tolist(),
                    max_length=self.config.max_input_length
                )
                
                # Tokenize summaries (as targets)
                target_encodings = self._tokenize_text(
                    df['summary'].tolist(),
                    is_target=True,
                    max_length=self.config.max_target_length
                )
                
                processed_data[split] = {
                    "input_ids": input_encodings.input_ids,
                    "attention_mask": input_encodings.attention_mask,
                    "labels": target_encodings.input_ids
                }
                
            return processed_data
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise e
        
    def initiate_data_preprocessing(self) -> Dict[str, str]:
        """
        Execute the complete data preprocessing pipeline.
        
        Returns:
            Dictionary with paths to processed data directories
        """
        try:
            logger.info("Starting data preprocessing pipeline")
            
            # 1. Load raw data
            dataframes = self.load_data()
            
            # 2. Preprocess data
            processed_data = self.preprocess_data(dataframes)
            
            # 3. Save processed data
            self.save_processed_data(processed_data)
            
            # 4. Return paths to saved data
            return {
                "train_dir": os.path.join(self.config.root_dir, "train"),
                "validation_dir": os.path.join(self.config.root_dir, "validation"),
                "test_dir": os.path.join(self.config.root_dir, "test")
            }
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            raise e
        
    def save_processed_data(self, processed_data: Dict[str, Dict[str, np.ndarray]]) -> None:
        """
        Save processed data as numpy arrays in split-specific directories.
        
        Args:
            processed_data: Dictionary containing processed data for each split
        """
        try:
            logger.info("Saving processed data to disk")
            
            # Create root directory if it doesn't exist
            os.makedirs(self.config.root_dir, exist_ok=True)
            
            for split, data in processed_data.items():
                split_dir = os.path.join(self.config.root_dir, split)
                os.makedirs(split_dir, exist_ok=True)
                
                # Save each array separately
                for key, array in data.items():
                    np.save(os.path.join(split_dir, f"{key}.npy"), array)
                    logger.debug(f"Saved {key} array for {split} split")
                
                logger.info(f"Saved processed {split} data to {split_dir}")
                
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise e