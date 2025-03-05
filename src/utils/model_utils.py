""" Utility functions for model training and evaluation. """

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.model_params import ModelParams
from src.config.paths import Paths


def split_dataset(
    df: pd.DataFrame, params: ModelParams
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training, validation, and test sets while ignoring rows with missing values.

    Args:
        df (pd.DataFrame): Input DataFrame containing the dataset.
        params (ModelParams): An instance of ModelParams containing split proportions.

    Returns:
        tuple: A tuple containing (train_df, val_df, test_df).
    """
    df_clean = df.dropna(subset=["raw_comp_writers_text", "CLEAN_TEXT"])

    train_val_df, test_df = train_test_split(
        df_clean, test_size=params.TEST_SIZE, random_state=42
    )

    # Split the remaining data into training and validation sets
    val_size_adjusted = params.VALIDATION_SIZE / (
        1 - params.TEST_SIZE
    )  # Adjust validation size relative to train_val_df
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size_adjusted, random_state=42
    )

    return train_df, val_df, test_df


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    Parameters:
        train (pd.DataFrame): The training dataset to be saved.
        val (pd.DataFrame): The validation dataset to be saved.
        test (pd.DataFrame): The test dataset to be saved.

        Returns: None
    """
    train.to_csv(Paths.TRAIN_DATA, index=False)
    val.to_csv(Paths.VAL_DATA, index=False)
    test.to_csv(Paths.TEST_DATA, index=False)
