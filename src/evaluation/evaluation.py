import json
import os
from datetime import datetime
from typing import Any, List, Optional

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from configs.config import ConfigPath, logger


class Evaluator:

    def __init__(
        self,
        ground_truth_data: dict,
        short_answer_results: Optional[List] = None,
        long_answer_results: Optional[List] = None,
    ):
        self.ground_truth_data = ground_truth_data
        self.short_answer_results = short_answer_results
        self.long_answer_results = long_answer_results

    def evaluate(self, to_save: bool = True):
        short_answer_metrics, long_answer_metrics = None, None
        if self.short_answer_results:
            short_answer_metrics = self.compute_short_answer_metrics(to_save)
        if self.long_answer_results:
            long_answer_metrics = self.compute_long_answer_metrics(to_save)
        return short_answer_metrics, long_answer_metrics

    def evaluate_retrieved_context(
        self,
    ):
        pass

    def compute_short_answer_metrics(self, to_save: bool = True) -> dict:
        if self.short_answer_results is None:
            raise ValueError("Results are not generated. Cannot compute metrics.")
        logger.info("Computing metrics...")

        actual_decisions = []
        generated_decisions = []

        for result in self.short_answer_results:
            actual_decisions.append(result["actual_response"])
            generated_decisions.append(result["generated_response"])

        # convert to binary (yes=1, no=0)
        actual_binary = [1 if decision == "yes" else 0 for decision in actual_decisions]
        generated_binary = [
            1 if decision == "yes" else 0 for decision in generated_decisions
        ]

        accuracy = accuracy_score(actual_binary, generated_binary)
        precision = precision_score(actual_binary, generated_binary, zero_division=0)
        recall = recall_score(actual_binary, generated_binary, zero_division=0)
        f1 = f1_score(actual_binary, generated_binary, zero_division=0)

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1: {f1:.4f}")

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        if to_save:
            self.save_metrics(metrics)

        return metrics

    def compute_long_answer_metrics(self, to_save: bool = True) -> dict:
        """to be completed"""
        pass

    def save_metrics(self, metrics: dict):
        """
        Saves metrics dictionary in json file.
        """

        # Define the file path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"answering_metrics_{timestamp}.json"
        file_path = os.path.join(ConfigPath.METRICS_DIR, file_name)
        try:
            with open(file_path, "w") as f:
                json.dump(metrics, f)
            logger.info(f"Results saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving results to {file_path}: {e}")
