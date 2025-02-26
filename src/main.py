import json
import os

from tqdm import tqdm

from configs import ConfigPath
from data_preprocessing.text_splitter import TextSplitter
from knowledge_graph.loader import GraphLoader
from llms.embedding_model import EmbeddingModel
from utils.utils import read_json_file


def initialize_graph_loader() -> GraphLoader:
    # required components
    text_splitter = TextSplitter()
    embedding_model = EmbeddingModel()

    graph_loader = GraphLoader(
        text_splitter=text_splitter, embedding_model=embedding_model
    )
    print("Graph loader initialized!")
    return graph_loader


if __name__ == "__main__":
    loader = initialize_graph_loader()
    raw_data = read_json_file(
        file_path=os.path.join(ConfigPath.RAW_DATA_DIR, "pqa_labeled.json")
    )
    data_for_load = loader.prepare_data_for_load(data=raw_data)
