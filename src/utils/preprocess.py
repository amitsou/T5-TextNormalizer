""" This module contains utility functions for loading and cleaning data from a CSV file. """

from pathlib import Path
from typing import Union

import chardet
import ftfy
import pandas as pd


def load_and_clean_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from a CSV file, fix encoding issues, and clean the data.
    Args:
        file_path (Union[str, Path]): The path to the CSV file.
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    The function performs the following steps:
    1. Detects the encoding of the file.
    2. Loads the CSV file into a DataFrame using the detected encoding.
    3. Fixes encoding issues in the 'raw_comp_writers_text' and 'CLEAN_TEXT' columns.
    4. Drops rows where both 'raw_comp_writers_text' and 'CLEAN_TEXT' columns are empty.
    5. Removes rows where 'raw_comp_writers_text' is entirely numeric.
    """
    encoding = detect_encoding(file_path)
    df = pd.read_csv(file_path, encoding=encoding)

    # Fix encoding for both columns
    df["raw_comp_writers_text"] = df["raw_comp_writers_text"].apply(
        lambda x: fix_encoding(x) if isinstance(x, str) else x
    )
    df["CLEAN_TEXT"] = df["CLEAN_TEXT"].apply(
        lambda x: fix_encoding(x) if isinstance(x, str) else x
    )
    df = df.dropna(
        subset=["raw_comp_writers_text", "CLEAN_TEXT"], how="all"
    )  # Drop rows where both columns are empty
    df = df[
        ~df["raw_comp_writers_text"].str.fullmatch(r"\d+", na=False)
    ]  # Remove rows where raw text is entirely numeric

    return df


def detect_encoding(file_path: Union[str, Path]) -> str:
    """
    Detect the encoding of a given file.

    Args:
        file_path (Union[str, Path]):
        The path to the file whose encoding needs to be detected.

    Returns (str):
        The detected encoding of the file.
        If the encoding cannot be detected, 'utf-8' is returned as the default.
    """
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
        return result["encoding"] or "utf-8"


def fix_encoding(text: str) -> str:
    """
    Fixes the encoding of a given text string.
    This function uses the `ftfy` library to
    correct any encoding issues in the input text.

    If the input is not a string, it returns the input unchanged.

    Args:
        text (str): The text string to be fixed.
    Returns:
        str: The text string with encoding issues corrected.
    """

    if isinstance(text, str):
        return ftfy.fix_text(text)
    return text
