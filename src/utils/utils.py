import json

import numpy as np
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity

from configs.config import logger


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def read_json_file(file_path: str) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise e
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file: {file_path}")
        raise e


def save_json_file(file_path: str, data: dict) -> None:
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file)
    except Exception as e:
        logger.error(f"Failed to write JSON to {file_path}. Error: {e}")
        raise


def compute_similarity_matrix(self, embeddings) -> np.ndarray:
    """
    Computes a cosine similarity matrix for a given set of embeddings.

    Args:
    - embeddings (numpy.ndarray): An array of embeddings.

    Returns:
    - numpy.ndarray: A cosine similarity matrix for the input embeddings.
    """
    return cosine_similarity(embeddings)
