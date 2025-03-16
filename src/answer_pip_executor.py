import os
import re
import time
from datetime import datetime
from typing import Any, Literal, Optional

import pandas as pd
from langchain_core.tools import BaseTool
from tqdm import tqdm

from configs.config import ConfigPath, logger
from utils.utils import save_json_file


class AnswerPipExecutor:

    def __init__(self, source_data: dict):
        self.source_data = source_data

    def generate_answers(
        self,
        retriever,
        answer_type: Literal["short", "long"],
        model_name: Optional[str] = "test",
        to_save: bool = True,
    ) -> list:

        short_answer_source_key = "final_decision"
        long_answer_source_key = "LONG_ANSWER"
        answer_source_key = (
            short_answer_source_key
            if answer_type == "short"
            else long_answer_source_key
        )

        logger.info("Starting the answer generation...")
        results = []
        for pmid, info in tqdm(self.source_data.items(), desc="Generating results..."):
            try:
                row = {}
                row["pmid"] = pmid
                row["actual_response"] = info[answer_source_key]
                question = info["QUESTION"]
                response = retriever.invoke(question)
                row["generated_response"] = response["answer"]
                row["contexts"] = [chunk["content"] for chunk in response["context"]]
                row["context_pmids"] = [chunk["pmid"] for chunk in response["context"]]
                # TODO: add time delay to goolge/groq providers
                # time.sleep(2)
                results.append(row)
            except Exception as e:
                logger.error(f"Error generating results for {pmid}: {e}")

        logger.info(f"Generated {len(results)} results.")

        results = self._add_context_comparisons_info(results)

        if to_save:
            self.save_results(
                results=results, answer_type=answer_type, model_name=model_name
            )
        return results

    def _add_context_comparisons_info(self, results: list) -> list:
        results_copy = results.copy()
        results_df = pd.DataFrame(results_copy)

        results_df["context_found"] = results_df.apply(
            lambda row: row["pmid"] in row["context_pmids"], axis=1
        )
        results_df["context_order"] = results_df.apply(
            lambda row: (
                row["context_pmids"].index(row["pmid"])
                if row["pmid"] in row["context_pmids"]
                else -1
            ),
            axis=1,
        )
        return results_df.to_dict(orient="records")

    def save_results(
        self, results: list, answer_type: Literal["short", "long"], model_name: str
    ) -> None:
        """
        Saves list in json file.
        """
        data = {"results": results}

        # Define the file path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{answer_type}_answer_{model_name}_{timestamp}.json"
        file_path = os.path.join(ConfigPath.RESULTS_DIR, file_name)
        save_json_file(file_path=file_path, data=data)
