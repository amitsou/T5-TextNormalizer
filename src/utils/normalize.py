""" Text normalization utilities."""

import re
from typing import Optional

from transformers import pipeline


class DataNormalizerConfig:
    """Configuration for text normalization."""

    def __init__(self):
        self.removal_patterns = [
            r"\bCopyright Control\b",
            r"\bSONY/ATV MUSIC PUBLISHING\b",
            r"\b\(Publishing\)\b",
            r"\bLtd\.\b",
            r"\b- ASCAP\b",
            r"^<Unknown>/",
        ]


class DataNormalizer:
    """Handles text normalization operations."""

    def __init__(self, config: Optional[DataNormalizerConfig] = None):
        self.config = config or DataNormalizerConfig()
        self.generator = pipeline(
            "text2text-generation", model="t5-small", tokenizer="t5-small"
        )

    def normalize_text(self, raw_text: str) -> str:
        """
        Normalize the given raw text by applying removal patterns
        and converting name formats.

        Args:
            raw_text (str): The raw text to be normalized.

        Returns (str):
            The normalized text. If the input is not a string or
            is an empty string, returns an empty string.
        """
        if not isinstance(raw_text, str) or raw_text.strip() == "":
            return ""
        normalized_text = self._apply_removal_patterns(raw_text)

        return self._convert_name_format(normalized_text)

    def _apply_removal_patterns(self, text: str) -> str:
        """
        Applies a series of removal patterns to the given text.

        Args:
            text (str): The input text to be processed.

        Returns:
            str: The text after applying all removal patterns.
        """
        result = text
        for pattern in self.config.removal_patterns:
            result = re.sub(pattern, "", result)
        return result

    def _convert_name_format(self, text: str) -> str:
        """
        Converts a given text containing names separated by slashes into a standardized format.
        The function processes each name variant separated by slashes ("/").

        If a name variant contains a comma, it assumes the format is "Last Name,
        First Name" and converts it to  "First Name Last Name".

        If no comma is found, it leaves the name as is.

        Args:
            text (str): The input string containing name variants separated by slashes.

        Returns (str):
            A string with the names converted to the "First Name Last Name" format,
            separated by slashes.
        """
        name_variants = text.split("/")
        cleaned_names = []

        for name in name_variants:
            name_parts = name.strip().split(", ")
            if len(name_parts) == 2:
                cleaned_names.append(f"{name_parts[1]} {name_parts[0]}")
            else:
                cleaned_names.append(name.strip())

        return "/".join(cleaned_names)

    def enhance_normalization(self, raw_text: str) -> str:
        """
        Enhance the normalization of the given raw text.
        This method first applies a base normalization to
        the raw text using the `normalize_text` method.

        It then generates a prompt to further correct and
        normalize the text using a text generation model.

        Args:
            raw_text (str): The raw text to be normalized.

        Returns:
            str: The enhanced normalized text.
        """
        base_normalized = self.normalize_text(raw_text)

        prompt = f"Correct and normalize: {base_normalized}"
        result = self.generator(prompt, max_length=128, truncation=True)
        return result[0]["generated_text"]
