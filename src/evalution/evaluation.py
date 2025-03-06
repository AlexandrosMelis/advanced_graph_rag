import json
import os
import re
import time
from datetime import datetime
from typing import Any, Optional

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from configs.config import ConfigPath, logger


class Evaluator:

    def __init__(
        self,
        ground_truth_data: dict,
        generated_results: Optional[dict] = None,
        retriever: Optional[Any] = None,
    ):
        self.ground_truth_data = ground_truth_data
        self.retriever = retriever
        self.generate_results = (
            generated_results
            if generated_results
            else self.generate_results(retriever=retriever)
        )

    def generate_results(self, retriever, to_save: bool = True) -> dict:
        if retriever is None:
            raise ValueError("Retriever is not set. Cannot generate results.")
        logger.info("Generating results...")

        results = []
        for pmid, info in tqdm(
            self.ground_truth_data.items(), desc="Generating results..."
        ):
            try:
                single_result = {}
                single_result["pmid"] = pmid
                single_result["actual_decision"] = info["final_decision"]
                question = info["QUESTION"]
                response = retriever.invoke(question)
                answer = self.clean_answer(response)
                single_result["generated_decision"] = answer
                results.append(single_result)
                time.sleep(4)
            except Exception as e:
                logger.error(f"Error generating results for {pmid}: {e}")

        logger.info(f"Generated {len(results)} results.")

        if to_save:
            self.save_results(results)

        return results

    def clean_answer(self, answer: str) -> str:
        """
        Apply regex pattern that extract only the keywords: "yes" or "no" from the llm's final answer.
        Get rid of any redundant text, delimiters, etc.
        """
        answer = answer.lower()
        match = re.search(r"\b(yes|no)\b", answer)
        if match:
            return match.group(1)
        else:
            logger.warning(
                f"Could not extract 'yes' or 'no' from answer: {answer}. Returning 'no' as default."
            )
            return ""

    def compute_metrics(self, to_save: bool = True) -> dict:
        if self.generate_results is None:
            raise ValueError("Results are not generated. Cannot compute metrics.")
        logger.info("Computing metrics...")

        actual_decisions = []
        generated_decisions = []

        for result in self.generate_results:
            actual_decisions.append(result["actual_decision"])
            generated_decisions.append(result["generated_decision"])

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

    def save_results(self, results: list):
        """
        Saves list in json file.
        """
        results_for_save = {"results": results}

        # Define the file path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"evaluation_results_{timestamp}.json"
        file_path = os.path.join(ConfigPath.RESULTS_DIR, file_name)
        try:
            with open(file_path, "w") as f:
                json.dump(results_for_save, f)
            logger.info(f"Results saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving results to {file_path}: {e}")

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
