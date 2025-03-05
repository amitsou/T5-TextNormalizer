import random
from typing import List

import pandas as pd
from textattack.augmentation import WordNetAugmenter


class DataAugmentor:
    """Handles text data augmentation using WordNet and noise injection."""

    def __init__(
        self, noise_prob: float = 0.1, augment_synonyms_prob: float = 0.5
    ) -> None:
        self.synonym_augmenter = WordNetAugmenter()
        self.noise_prob = noise_prob  # Initialize noise probability
        self.augment_synonyms_prob = (
            augment_synonyms_prob  # Initialize synonym augmentation probability
        )

    def add_noise(self, text: str) -> str:
        """
        Adds random noise to the given text with a specified probability.

        Parameters:
        text (str):
            The input text to which noise will be added.
        noise_prob (float):
            The probability of adding noise to the text. Default is 0.1.

        eturns (str):
            The text with added noise if the random condition is met, otherwise the original text.
        """
        if random.random() < self.noise_prob:
            return text + random.choice(["/", " & ", " ", ", "])
        return text

    def augment_text(self, text: str) -> List[str]:
        """
        Augments the given text by either applying synonym-based augmentation or adding noise.
        Args:
            text (str): The input text to be augmented.
        Returns:
            List[str]: A list of augmented text strings.
        """
        augmented_texts = []

        # Decide if we do a synonym-based augmentation for this text
        if random.random() < self.augment_synonyms_prob:
            synonyms = self.synonym_augmenter.augment(text)
            for syn in synonyms:
                augmented_texts.append(self.add_noise(syn))
        else:
            augmented_texts.append(self.add_noise(text))

        return augmented_texts

    def augment_dataset(self, df: pd.DataFrame, n_augment: int = 2) -> pd.DataFrame:
        # Drop rows with missing ground truth
        df = df.dropna(subset=["CLEAN_TEXT"])

        augmented_rows = []
        for _, row in df.iterrows():
            for _ in range(n_augment):
                for aug_text in self.augment_text(row["raw_comp_writers_text"]):
                    augmented_rows.append(
                        {
                            "raw_comp_writers_text": aug_text,
                            "CLEAN_TEXT": row["CLEAN_TEXT"],  # Propagate ground truth
                        }
                    )
        return pd.DataFrame(augmented_rows)
