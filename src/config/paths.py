"""File to store all the paths used in the project."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "src/models"


class Paths:
    """Configuration for file paths."""

    RAW_DATA = DATA_DIR / "raw_data/normalization_assesment_dataset_10k.csv"
    MODEL_SAVE = os.path.join(DATA_DIR, "models/t5_model")
    TOKENIZER_SAVE = os.path.join(DATA_DIR, "models/t5_tokenizer")
    TRAIN_DATA = os.path.join(DATA_DIR, "train.csv")
    VAL_DATA = os.path.join(DATA_DIR, "val.csv")
    TEST_DATA = os.path.join(DATA_DIR, "test.csv")
    CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
