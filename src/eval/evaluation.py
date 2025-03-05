"""Module for model evaluation using various metrics."""

import difflib
from typing import Dict, List

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge
from sklearn.model_selection import KFold


class Evaluator:
    """Handles model evaluation with multiple metrics."""

    def __init__(self):
        self.rouge = Rouge()
        self.smoothie = SmoothingFunction().method4

    def evaluate(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate the performance of predictions against references using various metrics.
        Args:
            predictions (List[str]): A list of predicted strings.
            references (List[str]): A list of reference strings.
        Returns:
            Dict[str, float]: A dictionary containing the following evaluation metrics:
                - 'BLEU': The average BLEU score of the predictions.
                - 'Char_Accuracy': The average character-level accuracy of the predictions.
                - 'Word_Accuracy': The average word-level accuracy of the predictions.
                - 'Normalized_Edit_Distance': The average normalized edit distance between predictions and references.
        """
        bleu_scores = []
        char_accs = []
        word_accs = []
        edit_distances = []

        for pred, ref in zip(predictions, references):
            bleu = sentence_bleu(
                [ref.split()], pred.split(), smoothing_function=self.smoothie
            )
            bleu_scores.append(bleu)

            min_len = min(len(pred), len(ref))
            char_matches = sum(1 for i in range(min_len) if pred[i] == ref[i])
            char_accs.append(char_matches / max(len(ref), 1))

            pred_words = pred.split()
            ref_words = ref.split()
            min_words = min(len(pred_words), len(ref_words))
            word_matches = sum(
                1 for i in range(min_words) if pred_words[i] == ref_words[i]
            )
            word_accs.append(word_matches / max(len(ref_words), 1))

            sm = difflib.SequenceMatcher(None, pred, ref)
            edit_distances.append(1 - sm.ratio())

        return {
            "BLEU": np.mean(bleu_scores),
            "Char_Accuracy": np.mean(char_accs),
            "Word_Accuracy": np.mean(word_accs),
            "Normalized_Edit_Distance": np.mean(edit_distances),
        }

    def cross_validate(self, df: pd.DataFrame, k: int = 5) -> Dict[str, float]:
        """
        Perform k-fold cross-validation on the given DataFrame.
        Args:
            df (pd.DataFrame): The input DataFrame containing the data to be evaluated.
            k (int, optional): The number of folds for cross-validation. Default is 5.
        Returns:
            Dict[str, float]: A dictionary containing the average evaluation metrics
                              across all folds.
        """

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_results = []

        for _, test_index in kf.split(df):
            test_fold = df.iloc[test_index]
            predictions = test_fold["CLEAN_TEXT"].tolist()
            references = test_fold["CLEAN_TEXT"].tolist()
            fold_results.append(self.evaluate(predictions, references))

        return {
            key: np.mean([fold[key] for fold in fold_results])
            for key in fold_results[0]
        }
