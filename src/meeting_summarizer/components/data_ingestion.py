import os
import sys
import pandas as pd
from dataclasses import dataclass
from datasets import load_dataset
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.meeting_summarizer import logger
from src.meeting_summarizer.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize data ingestion with the configuration parameters
        
        Args:
            config (DataIngestionConfig): Configuration for data ingestion
        """
        self.config = config

    def download_dataset(self):
        """
        Download the SamSum dataset from Hugging Face
        """
        logger.info("Downloading SamSum dataset from Hugging Face")
        
        try:
            dataset = load_dataset(self.config.source_dataset, trust_remote_code=True)
            return dataset
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise e

    def convert_to_dataframe(self, dataset) -> Dict[str, pd.DataFrame]:
        """
        Convert Hugging Face dataset to pandas DataFrames
        
        Args:
            dataset: Hugging Face dataset
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with train, validation, and test DataFrames
        """
        logger.info("Converting dataset to pandas DataFrames")
        
        try:
            dataframes = {}
            
            # Extract each split from the dataset and convert to DataFrame
            for split in dataset:
                # Convert dataset to pandas DataFrame
                df = pd.DataFrame({
                    'id': dataset[split]['id'],
                    'dialogue': dataset[split]['dialogue'],
                    'summary': dataset[split]['summary']
                })
                dataframes[split] = df
                
            return dataframes
        except Exception as e:
            logger.error(f"Error converting to dataframe: {e}")
            raise e

    def save_data(self, dataframes: Dict[str, pd.DataFrame]):
        """
        Save DataFrames to disk
        
        Args:
            dataframes (Dict[str, pd.DataFrame]): Dictionary with train, validation, test DataFrames
        """
        logger.info("Saving data to disk")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.config.root_dir, exist_ok=True)
            
            # Save each split to its respective path
            if 'train' in dataframes:
                dataframes['train'].to_csv(self.config.train_data_path, index=False)
                logger.info(f"Train data saved at: {self.config.train_data_path}")
                
            if 'validation' in dataframes:
                dataframes['validation'].to_csv(self.config.validation_data_path, index=False)
                logger.info(f"Validation data saved at: {self.config.validation_data_path}")
                
            if 'test' in dataframes:
                dataframes['test'].to_csv(self.config.test_data_path, index=False)
                logger.info(f"Test data saved at: {self.config.test_data_path}")
                
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise e

    def setup_vector_db(self, dataframes: Dict[str, pd.DataFrame]):
        """
        Set up ChromaDB vector database with the dataset
        
        Args:
            dataframes (Dict[str, pd.DataFrame]): Dictionary with dataframes for each split
        """
        logger.info("Setting up ChromaDB vector database")
        
        try:
            # Create directory for vector DB if it doesn't exist
            os.makedirs(self.config.vector_db_path, exist_ok=True)
            
            # Initialize ChromaDB client
            client = chromadb.PersistentClient(path=str(self.config.vector_db_path))
            
            # Use all-MiniLM-L6-v2 model for embeddings (lightweight for 8GB RAM)
            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config.embedding_model_name
            )
            
            # Create collections for dialogues and summaries
            dialogue_collection = client.get_or_create_collection(
                name="dialogues",
                embedding_function=sentence_transformer_ef
            )
            
            summary_collection = client.get_or_create_collection(
                name="summaries",
                embedding_function=sentence_transformer_ef
            )
            
            # Add training data to the collections in batches
            if 'train' in dataframes:
                df = dataframes['train']
                
                # Define batch size (less than the max limit of 5461)
                batch_size = 1000
                
                # Process dialogue batches
                logger.info(f"Adding dialogue data to vector DB in batches of {batch_size}")
                for i in tqdm(range(0, len(df), batch_size), desc="Processing dialogue batches"):
                    end_idx = min(i + batch_size, len(df))
                    batch_df = df.iloc[i:end_idx]
                    
                    dialogue_collection.add(
                        documents=batch_df['dialogue'].tolist(),
                        metadatas=[{"source": "train", "id": id_} for id_ in batch_df['id'].tolist()],
                        ids=[f"train_dialogue_{i + j}" for j in range(len(batch_df))]
                    )
                
                # Process summary batches
                logger.info(f"Adding summary data to vector DB in batches of {batch_size}")
                for i in tqdm(range(0, len(df), batch_size), desc="Processing summary batches"):
                    end_idx = min(i + batch_size, len(df))
                    batch_df = df.iloc[i:end_idx]
                    
                    summary_collection.add(
                        documents=batch_df['summary'].tolist(),
                        metadatas=[{"source": "train", "id": id_} for id_ in batch_df['id'].tolist()],
                        ids=[f"train_summary_{i + j}" for j in range(len(batch_df))]
                    )
                
            logger.info(f"Vector database created at: {self.config.vector_db_path}")
            
        except Exception as e:
            logger.error(f"Error setting up vector database: {e}")
            raise e

    def initiate_data_ingestion(self) -> Dict[str, Path]:
        """
        Initiate the data ingestion process
        
        Returns:
            Dict[str, Path]: Dictionary with paths to train, validation, and test data
        """
        logger.info("Initiating data ingestion")
        
        try:
            # Download dataset
            dataset = self.download_dataset()
            
            # Convert to DataFrames
            dataframes = self.convert_to_dataframe(dataset)
            
            # Save data to CSV
            self.save_data(dataframes)
            
            # Set up vector database if enabled
            if self.config.use_vector_db:
                self.setup_vector_db(dataframes)
            
            # Return paths
            data_paths = {
                "train_data_path": self.config.train_data_path,
                "validation_data_path": self.config.validation_data_path,
                "test_data_path": self.config.test_data_path,
                "vector_db_path": self.config.vector_db_path if self.config.use_vector_db else None
            }
            
            return data_paths
            
        except Exception as e:
            logger.error(f"Exception occurred during data ingestion: {e}")
            raise e