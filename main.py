""" Main script for running the text normalization workflow. """

import argparse
import os

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import T5Tokenizer

from src.config.model_params import ModelParams
from src.config.paths import Paths
from src.eval.evaluation import Evaluator
from src.models.predict import T5Predictor
from src.models.train import T5Trainer, TextDataset
from src.utils.augment import DataAugmentor
from src.utils.general_utils import timeit
from src.utils.model_utils import save_splits, split_dataset
from src.utils.normalize import DataNormalizer
from src.utils.preprocess import load_and_clean_data


def parse_args():
    """
    Parse command-line arguments for controlling the data pipeline and model flow.
    """
    parser = argparse.ArgumentParser(description="Text Normalization Workflow")

    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Load, clean, and optionally augment data, then split into train/val/test.",
    )
    parser.add_argument(
        "--augment",
        type=int,
        default=0,
        help="Number of augmentations per row during data preparation. (Requires --prepare.)",
    )

    parser.add_argument(
        "--train", action="store_true", help="Train the model on the train/val splits."
    )

    parser.add_argument(
        "--test", action="store_true", help="Evaluate the model on the test split."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples for testing/evaluation. (Requires --test.)",
    )
    parser.add_argument(
        "--inference", action="store_true", help="Run inference on example inputs."
    )

    return parser.parse_args()


@timeit
def prepare_data(n_augment: int = 0):
    """
    Loads the raw CSV, cleans it, optionally augments it, and splits into
    train, validation, and test sets. The resulting splits are saved as CSVs.

    Args:
        n_augment (int): Number of augmented samples per row. If 0, no augmentation.
    """
    print("Loading and cleaning data...")
    df = load_and_clean_data(Paths.RAW_DATA)

    rb_normalizer = DataNormalizer()
    df["raw_comp_writers_text"] = df["raw_comp_writers_text"].apply(
        rb_normalizer.normalize_text
    )

    if n_augment > 0:
        print(f"Augmenting data with n_augment={n_augment} ...")
        augmentor = DataAugmentor(noise_prob=0.1, augment_synonyms_prob=0.5)
        augmented_df = augmentor.augment_dataset(df, n_augment=n_augment)
        df = pd.concat([df, augmented_df], ignore_index=True)

    df = df.dropna(subset=["raw_comp_writers_text", "CLEAN_TEXT"], how="any")

    print("Splitting dataset into train/val/test...")
    train_df, val_df, test_df = split_dataset(df, ModelParams)

    print("Saving split files to disk...")
    save_splits(train_df, val_df, test_df)
    print("Data preparation complete!")


@timeit
def train_model():
    """
    Trains the T5 model on the prepared train/val splits with early stopping and
    checkpointing. Logs validation loss at intervals.
    """
    print("Loading train and validation splits from CSV...")

    if not os.path.exists(Paths.TRAIN_DATA):
        raise FileNotFoundError(
            f"Train data file not found at {Paths.TRAIN_DATA}. Please run with --prepare first."
        )
    if not os.path.exists(Paths.VAL_DATA):
        raise FileNotFoundError(
            f"Validation data file not found at {Paths.VAL_DATA}. Please run with --prepare first."
        )

    train_df = pd.read_csv(Paths.TRAIN_DATA)
    val_df = pd.read_csv(Paths.VAL_DATA)
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    tokenizer = T5Tokenizer.from_pretrained(ModelParams.MODEL_NAME)

    train_dataset = TextDataset(
        train_df["raw_comp_writers_text"].tolist(),
        train_df["CLEAN_TEXT"].tolist(),
        tokenizer,
    )
    val_dataset = TextDataset(
        val_df["raw_comp_writers_text"].tolist(),
        val_df["CLEAN_TEXT"].tolist(),
        tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=ModelParams.BATCH_SIZE,
        shuffle=True,
        num_workers=getattr(ModelParams.NUM_WORKERS, "NUM_WORKERS", 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=ModelParams.BATCH_SIZE,
        shuffle=False,
        num_workers=getattr(ModelParams.NUM_WORKERS, "NUM_WORKERS", 0),
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,  # #number of epochs with no improvement to stop
        mode="min",  # #minimize the validation loss
    )

    checkpoint_dir = Paths.CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    print("Initializing PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=ModelParams.NUM_EPOCHS,
        val_check_interval=0.25,  # #validate 4 times per epoch (every 25%)
        enable_model_summary=True,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    model = T5Trainer()
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    print("Training completed.")
    print("Best checkpoint:", checkpoint_callback.best_model_path)

    print("Saving final model weights...")
    model.model.save_pretrained(Paths.MODEL_SAVE)
    model.tokenizer.save_pretrained(Paths.TOKENIZER_SAVE)
    print(f"Model + tokenizer saved to {Paths.MODEL_SAVE} and {Paths.TOKENIZER_SAVE}.")


@timeit
def test_model(num_samples: int = 100):
    """
    Loads the test split, runs a subset of samples through the model, and prints
    evaluation metrics plus sample predictions.

    Args:
        num_samples (int): number of random samples from the test set to evaluate.
    """
    print("Loading test split from CSV...")
    if not os.path.exists(Paths.TEST_DATA):
        raise FileNotFoundError(
            f"Test data file not found at {Paths.TEST_DATA}. Please run with --prepare first."
        )

    test_df = pd.read_csv(Paths.TEST_DATA)

    test_df = test_df.dropna(subset=["CLEAN_TEXT"])
    if len(test_df) == 0:
        print("No test data found or 'CLEAN_TEXT' is empty in the test set.")
        return

    # Randomly sample a portion of the test set
    sample_count = min(num_samples, len(test_df))
    test_samples = test_df.sample(sample_count)

    print("Initializing T5Predictor...")
    predictor = T5Predictor()

    print(f"Generating predictions for {sample_count} test samples...")

    inputs = test_samples["raw_comp_writers_text"].tolist()
    predictions = [predictor.predict(txt) for txt in inputs]
    targets = test_samples["CLEAN_TEXT"].tolist()

    # Evaluate the predictions
    evaluator = Evaluator()
    metrics = evaluator.evaluate(predictions, targets)

    print("\nEvaluation Results on Test Subset:")
    print(f"BLEU Score:                 {metrics['BLEU']:.4f}")
    print(f"Character-level Accuracy:   {metrics['Char_Accuracy']:.4f}")
    print(f"Word-level Accuracy:        {metrics['Word_Accuracy']:.4f}")
    print(f"Normalized Edit Distance:   {metrics['Normalized_Edit_Distance']:.4f}")

    print("\nExample Predictions:")
    for i in range(min(3, sample_count)):
        raw_inp = test_samples.iloc[i]["raw_comp_writers_text"]
        pred = predictions[i]
        actual = targets[i]
        print(f"  Input:     {raw_inp}")
        print(f"  Predicted: {pred}")
        print(f"  Actual:    {actual}\n")


def inference_example():
    predictor = T5Predictor()

    examples = [
        "<Unknown>/Wright, Justyce Kaseem",
        "Pixouu/Abdou Gambetta/Copyright Control",
        "Mike Hoyer/JERRY CHESNUT/SONY/ATV MUSIC PUBLISHING (UK) LIMITED",
    ]

    for raw_text in examples:
        normalized = predictor.predict(raw_text)
        print(f"RAW TEXT: {raw_text}")
        print(f"PREDICTED NORMALIZED: {normalized}")
        print("-" * 50)


def main():
    args = parse_args()

    if args.prepare:
        prepare_data(n_augment=args.augment)

    if args.train:
        train_model()

    if args.test:
        test_model(num_samples=args.samples)

    if args.inference:
        inference_example()


if __name__ == "__main__":
    main()
