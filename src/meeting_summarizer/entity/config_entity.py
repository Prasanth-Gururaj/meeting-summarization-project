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