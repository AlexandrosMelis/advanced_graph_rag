import os
from typing import Any, Tuple

from tqdm import tqdm

from configs.config import ConfigPath
from utils.utils import save_json_file


def collect_retrieved_chunks(
    source_data: list, retriever: Any, output_dir: str = None
) -> dict:
    """Run the retriever against every sample in the source data.
    Prepares the results for the evaluation.

    Returns:
        dict: contains the id, question and the retrieved chunks
    """
    results = {}
    for sample in tqdm(source_data, desc="Collecting retrieved chunks..."):
        sample_id = sample.get("id")
        question = sample.get("question")
        retrieved_data = retriever.invoke(question)
        retrieved_data = [(chunk["pmid"], chunk["score"]) for chunk in retrieved_data]
        results[sample_id] = retrieved_data

    if output_dir:
        file_name = "retrieved_chunks.json"
        file_path = os.path.join(output_dir, file_name)
        save_json_file(file_path=file_path, data=results)

    return results


def collect_generated_answers(
    source_data: list, retriever: Any, to_save: bool = False, output_dir: str = None
) -> Tuple[dict, str]:
    pass
