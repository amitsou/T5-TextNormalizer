""" T5 model prediction with dynamic normalization. """

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.config.model_params import ModelParams
from src.config.paths import Paths
from src.utils.normalize import DataNormalizer


class T5Predictor:
    """Handles T5 model predictions with dynamic normalization."""

    def __init__(self):
        self.model = T5ForConditionalGeneration.from_pretrained(Paths.MODEL_SAVE)
        self.tokenizer = T5Tokenizer.from_pretrained(Paths.TOKENIZER_SAVE)
        self.normalizer = DataNormalizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, text: str) -> str:
        """
        Generates a prediction based on the provided text input.
        Args:
            text (str): The input text to be normalized and processed.
        Returns:
            str: The normalized and enhanced output text.
        """
        input_text = f"normalize: {text}"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(
            self.device
        )
        output_ids = self.model.generate(input_ids, max_length=ModelParams.MAX_LENGTH)
        rule_based_output = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )
        return self.normalizer.enhance_normalization(rule_based_output)
