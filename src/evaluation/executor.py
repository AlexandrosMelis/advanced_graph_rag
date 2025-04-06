import os
from typing import Any, Literal, Tuple

from tqdm import tqdm

from configs.config import ConfigPath
from retrieval_techniques.similarity_search import SimilaritySearchRetriever
from utils.utils import save_json_file


def collect_retrieved_chunks(
    source_data: list,
    retriever: SimilaritySearchRetriever,
    k: int,
    retrieval_technique: Literal["relevant_contexts", "relevant_meshes"],
    output_dir: str = None,
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
        retrieved_data = retriever.retrieve_chunks(
            query=question, k=k, technique=retrieval_technique
        )
        retrieved_data = [(chunk["pmid"], chunk["score"]) for chunk in retrieved_data]
        results[sample_id] = retrieved_data

    if output_dir:
        file_name = f"{k}_{retrieval_technique}_chunks.json"
        file_path = os.path.join(output_dir, file_name)
        save_json_file(file_path=file_path, data=results)

    return results


def collect_generated_answers(
    source_data: list, retriever: Any, output_dir: str = None
) -> list:
    results = []
    for sample in tqdm(source_data, desc="Collecting answers and chunks..."):
        sample_id = sample.get("id")
        user_input = sample.get("question")
        reference = sample.get("answer")
        output = retriever.invoke(user_input)
        if "answer" not in output:
            raise ValueError(
                "The retriever did not return an answer. Check retriever initialization!"
            )
        response = output["answer"]
        contexts = output["context"]
        retrieved_contexts = [context["content"] for context in contexts]

        result = {
            "id": sample_id,
            "user_input": user_input,
            "reference": reference,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
        }
        results.append(result)

    if output_dir:
        file_name = "retrieved_answers.json"
        file_path = os.path.join(output_dir, file_name)
        save_json_file(file_path=file_path, data=results)

    return results
