"""Model training module for T5 model."""

from typing import List

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.config.model_params import ModelParams


class TextDataset(Dataset):
    """Dataset class for T5 training."""

    def __init__(self, inputs: List[str], targets: List[str], tokenizer: T5Tokenizer):
        self.inputs = [f"normalize: {text}" for text in inputs]
        self.targets = targets
        self.tokenizer = tokenizer

    def __getitem__(self, idx: int):
        """
        Retrieves the input and target text at the specified index,
        tokenizes them, and returns a dictionary containing the tokenized
        input IDs, attention mask, and tokenized target IDs as labels.

        Args:
            idx (int): The index of the input and target text to retrieve.
        Returns:
            dict: A dictionary containing the following keys:
            - "input_ids" (torch.Tensor): The tokenized input IDs.
            - "attention_mask" (torch.Tensor): The attention mask for the input IDs.
            - "labels" (torch.Tensor): The tokenized target IDs.
        """
        input_text = self.inputs[idx]
        target_text = self.targets[idx]

        inputs = self.tokenizer(
            input_text,
            max_length=ModelParams.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        targets = self.tokenizer(
            target_text,
            max_length=ModelParams.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze(),
        }

    def __len__(self) -> int:
        """
        Returns the number of input elements.

        Returns:
            int: The number of elements in the inputs.
        """
        return len(self.inputs)


class T5Trainer(pl.LightningModule):
    """T5 model trainer with early stopping and checkpointing."""

    def __init__(self):
        """
        Initializes the training model and tokenizer.

        Attributes:
            model (T5ForConditionalGeneration): The T5 model for conditional generation.
            tokenizer (T5Tokenizer): The tokenizer for the T5 model.
        """
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(ModelParams.MODEL_NAME)
        self.tokenizer = T5Tokenizer.from_pretrained(ModelParams.MODEL_NAME)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the model.

        Args:
            input_ids (torch.Tensor): Tensor containing the input IDs.
            attention_mask (torch.Tensor): Tensor containing the attention mask.
            labels (torch.Tensor, optional): Tensor containing the labels. Defaults to None.

        Returns:
            torch.Tensor: The output tensor from the model.
        """
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configures and returns the AdamW optimizer for the model.

        This method sets up the AdamW optimizer with the learning
        rate and weight decay specified in the ModelParams class.

        Returns:
            torch.optim.AdamW: The configured AdamW optimizer.
        """
        return torch.optim.AdamW(
            self.parameters(),
            lr=ModelParams.LEARNING_RATE,
            weight_decay=ModelParams.WEIGHT_DECAY,
        )
