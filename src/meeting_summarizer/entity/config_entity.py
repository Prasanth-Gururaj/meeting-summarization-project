from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_dataset: str
    train_data_path: Path
    validation_data_path: Path
    test_data_path: Path
    use_vector_db: bool
    vector_db_path: Path
    embedding_model_name: str

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    train_data_path: Path
    validation_data_path: Path
    test_data_path: Path
    model_name: str
    max_input_length: int
    max_target_length: int