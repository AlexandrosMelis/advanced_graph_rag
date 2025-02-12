import json
from typing import List

from tqdm import tqdm

from data_preprocessing.text_splitter import TextSplitter
from llms.embedding_model import EmbeddingModel


class GraphLoader:

    def __init__(
        self, data: dict, embedding_model: EmbeddingModel, text_splitter: TextSplitter
    ):
        self.text_splitter = text_splitter
        self.embedding_model = embedding_model
        self.data = data

        # schema labels
        self.ARTICLE_LABEL = "ARTICLE"
        self.CONTEXT_LABEL = "CONTEXT"
        self.MESH_LABEL = "MESH"

    @staticmethod
    def from_json_file(path: str) -> "GraphLoader":
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return GraphLoader(data=data)

    def create_schema_node_data(self):
        node_data = []
        for pmid, info in tqdm(self.data.items()):
            graph_data = {}
            graph_data["article"] = self._create_article_node_properties(
                pmid=pmid, info=info
            )
            graph_data["context"] = self._create_context_node_properties(info=info)
            graph_data["meshes"] = self._create_meshes_node_properties(info=info)
            node_data.append(graph_data)
            break
        return node_data

    def _create_article_node_properties(self, pmid: str, info: dict) -> dict:
        return {"properties": {"pmid": pmid, "year": info["YEAR"]}}

    def _create_context_node_properties(self, info: dict) -> dict:
        return {"properties": {"text_content": " ".join(info["CONTEXTS"])}}

    def _create_meshes_node_properties(self, info: dict) -> List[dict]:
        return [
            self._create_mesh_node_properties(mesh_term=mesh) for mesh in info["MESHES"]
        ]

    def _create_mesh_node_properties(self, mesh_term: str) -> dict:
        return {"properties": {"name": mesh_term}}

    def _create_chunk_node_properties(self, text: str) -> List[dict]:
        """
        Splits text into chunks and embeds them.
        Returns a list of node properties.
        """
        chunks = self._get_text_chunks(text)
        embeddings = self.embedding_model.embed_documents(chunks)
        node_properties = [
            {"properties": {"text_content": chunk, "embedding": embedding}}
            for chunk, embedding in zip(chunks, embeddings)
        ]
        return node_properties

    def _get_text_chunks(self, text: str) -> list:
        return self.text_splitter.split_text(text)
