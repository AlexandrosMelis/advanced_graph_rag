import json
from typing import List

from tqdm import tqdm

from data_preprocessing.text_splitter import TextSplitter
from llms.embedding_model import EmbeddingModel


class GraphLoader:
    """
    Class to prepare and load data into the graph database.
    Receives a dictionary with the following structure:
    {
        "PMID": {
            "YEAR": int,
            "QUESTION": str,
            "CONTEXTS": List[str],
            "MESHES": List[str],
            "LABELS": List[str],
            "reasoning_required_pred": str,
            "reasoning_free_pred": str,
            "final_decision": str,
            "LONG_ANSWER": str
        }
    }
    Transform the data into a list of dictionaries to be loaded into the graph database.
    """

    def __init__(self, embedding_model: EmbeddingModel, text_splitter: TextSplitter):
        self.text_splitter = text_splitter
        self.embedding_model = embedding_model
        self.prepared_data = None

        # INPUT DATA KEYS
        self.QUESTION_KEY = "QUESTION"
        self.CONTEXTS_KEY = "CONTEXTS"
        self.LABELS_KEY = "LABELS"
        self.MESHES_KEY = "MESHES"
        self.YEAR_KEY = "YEAR"
        self.REASONING_REQUIRED_PRED_KEY = "reasoning_required_pred"
        self.REASONING_FREE_PRED_KEY = "reasoning_free_pred"
        self.FINAL_DECISION_KEY = "final_decision"
        self.LONG_ANSWER_KEY = "LONG_ANSWER"

        # NODE DATA KEYS
        self.ARTICLE_KEY = "article"
        self.CONTEXTS_KEY = "contexts"
        self.MESHES_KEY = "meshes"

        # NODE LABELS
        self.ARTICLE_LABEL = "ARTICLE"
        self.CONTEXT_LABEL = "CONTEXT"
        self.MESH_LABEL = "MESH"

        # NODE LABEL MAPPING
        self.NODE_LABEL_MAPPING = {
            "article": self.ARTICLE_LABEL,
            "contexts": self.CONTEXT_LABEL,
            "meshes": self.MESH_LABEL,
        }

        # RELATIONSHIPS
        self.HAS_CONTEXT_REL = "HAS_CONTEXT"
        self.HAS_MESH_TERM_REL = "HAS_MESH_TERM"
        self.IS_SIMILAR_TO_REL = "IS_SIMILAR_TO"

        # RELATIONSHIP MAPPING
        self.RELATIONSHIP_MAPPING = {
            "article-[]->contexts": self.HAS_CONTEXT_REL,
            "contex-[]->meshes": self.HAS_MESH_TERM_REL,
            "contexts-[]->contexts": self.IS_SIMILAR_TO_REL,
        }

        # REL SEPARATOR
        self.REL_SEPARATOR = "-[]->"

    def prepare_data_for_load(self, data: dict) -> List[dict]:
        data_for_load = []
        for pmid, info in tqdm(data.items()):
            pmid_node_data = {}
            pmid_node_data[self.ARTICLE_KEY] = self._create_article_node_properties(
                pmid=pmid, info=info
            )
            pmid_node_data[self.CONTEXTS_KEY] = self._create_context_node_properties(
                info=info
            )
            pmid_node_data[self.MESHES_KEY] = self._create_meshes_node_properties(
                info=info
            )
            data_for_load.append(pmid_node_data)
            # TODO: to be removed (for testing purposes)s
            break
        self.prepared_data = data_for_load
        return data_for_load

    def _create_article_node_properties(self, pmid: str, info: dict) -> dict:
        return {
            "properties": {
                "pmid": pmid,
                "year": info["YEAR"],
                "question": info["QUESTION"],
                "reasoning_required_pred": info["reasoning_required_pred"],
                "reasoning_free_pred": info["reasoning_free_pred"],
                "final_decision": info["final_decision"],
                "long_answer": info["LONG_ANSWER"],
            }
        }

    def _create_context_node_properties(self, info: dict) -> List[dict]:
        contexts_str = " ".join(info["CONTEXTS"])
        chunks = self.text_splitter.split_text(contexts_str)
        embeddings = self.embedding_model.embed_documents(chunks)
        labels = info["LABELS"]
        context_node_properties = []
        for chunk, embedding in zip(chunks, embeddings):
            context_node_properties.append(
                {
                    "properties": {
                        "text_content": chunk,
                        "embedding": embedding,
                        "labels": labels,
                    }
                }
            )

        return context_node_properties

    def _create_meshes_node_properties(self, info: dict) -> List[dict]:
        return [
            self._create_mesh_node_properties(mesh_term=mesh) for mesh in info["MESHES"]
        ]

    def _create_mesh_node_properties(self, mesh_term: str) -> dict:
        return {"properties": {"name": mesh_term}}

    def load_data_to_graph(self):
        for pmid_data in tqdm(self.prepared_data):
            for node_label, node_data in pmid_data.items():
                for node_properties in node_data:
                    print(json.dumps(node_properties, indent=4))
                    # graph.create_node(
                    #     label=self.NODE_LABEL_MAPPING[node_label],
                    #     properties=node_properties["properties"],
                    # )
                    # break
                # break
            # break
