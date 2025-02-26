import json
import os
from typing import List

from tqdm import tqdm

from configs.config import ConfigPath, logger
from data_preprocessing.text_splitter import TextSplitter
from knowledge_graph.crud import GraphCrud
from llms.embedding_model import EmbeddingModel
from utils.utils import read_json_file


class GraphLoader:
    """
    Class responsible for loading data into a knowledge graph.

    This class handles the preparation and loading of data, including articles,
    contexts, and mesh terms, into a graph database. It utilizes an embedding
    model for generating vector embeddings of text content.
    """

    QUESTION_KEY = "QUESTION"
    CONTEXTS_KEY = "CONTEXTS"
    LABELS_KEY = "LABELS"
    MESHES_KEY = "MESHES"
    YEAR_KEY = "YEAR"
    REASONING_REQUIRED_PRED_KEY = "reasoning_required_pred"
    REASONING_FREE_PRED_KEY = "reasoning_free_pred"
    FINAL_DECISION_KEY = "final_decision"
    LONG_ANSWER_KEY = "LONG_ANSWER"

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

    def __init__(
        self,
        data: dict,
        embedding_model: EmbeddingModel,
        text_splitter: TextSplitter,
        crud: GraphCrud,
    ):
        """
        Initializes the GraphLoader with data, embedding model, text splitter, and graph CRUD operations.

        Args:
            data (dict): The data to be loaded into the graph, structured as a dictionary.
            embedding_model (EmbeddingModel): The model used to generate text embeddings.
            text_splitter (TextSplitter): The utility used to split text into chunks.
            crud (GraphCrud): The interface for interacting with the graph database.
        """
        self.data = data
        self.text_splitter = text_splitter
        self.embedding_model = embedding_model
        self.crud = crud
        self.prepared_data = None
        self.mesh_terms = self._get_distinct_meshes()

        # sources
        self.meshes_file_name = "mesh_definitions.json"
        self.mesh_terms = read_json_file(
            file_path=os.path.join(ConfigPath.EXTERNAL_DATA_DIR, self.meshes_file_name)
        )

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

        # Properties
        self.embedding_property_name = "embedding"

    def _get_distinct_meshes(self) -> list:
        """
        Extracts a list of unique MESH terms from the input data.

        Returns:
            list: A list of distinct MESH terms.
        """
        distinct_meshes = []
        for value in self.data.values():
            meshes = value[self.MESHES_KEY]
            for mesh in meshes:
                if mesh not in distinct_meshes:
                    distinct_meshes.append(mesh)
        return distinct_meshes

    def prepare_meshes_for_load(self) -> List[dict]:
        """
        Prepares the MESH term data for loading into the graph.

        This includes creating properties dictionaries for each MESH term,
        including their definition and embedding.

        Returns:
            List[dict]: A list of dictionaries, each containing the properties
                         for a MESH node.
        """
        mesh_node_properties_list = []
        for term in tqdm(self.mesh_terms, desc="Preparing meshes..."):
            mesh_node_properties = self._create_mesh_node_properties(mesh_term=term)
            mesh_node_properties_list.append(mesh_node_properties)
        return mesh_node_properties_list

    def load_mesh_nodes(self) -> None:
        """
        Loads MESH nodes into the graph database.

        This function iterates through the prepared MESH data, creates nodes
        in the graph, and sets their vector embedding properties.
        """
        prepared_meshes = self.prepare_meshes_for_load()
        for mesh_info in tqdm(prepared_meshes, desc="Loading meshes..."):
            mesh_properties = mesh_info["properties"]
            mesh_embedding = mesh_properties.pop("embedding")
            node_id = self.crud.create_node(
                label=self.MESH_LABEL, properties=mesh_properties
            )
            self.crud.set_node_vector_property(
                node_id=node_id, embedding=mesh_embedding
            )
        logger.debug("Loading finished!")

    def prepare_data_for_load(self) -> List[dict]:
        """
        Prepares all data for loading into the graph.

        This function iterates through the raw data, creates the node
        properties for articles and contexts, and structures the data
        for efficient loading into the graph.

        Returns:
            List[dict]: A list of dictionaries, each representing a set of
                         nodes (article and its contexts) to be loaded.
        """
        data_for_load = []
        for pmid, info in tqdm(self.data.items()):
            pmid_node_data = {}
            pmid_node_data[self.ARTICLE_LABEL] = self._create_article_node_properties(
                pmid=pmid, pmid_data=info
            )
            pmid_node_data[self.CONTEXT_LABEL] = self._create_context_node_properties(
                contexts=info[self.CONTEXTS_KEY], labels=info[self.LABELS_KEY]
            )
            pmid_node_data[self.MESH_LABEL] = info[self.MESHES_KEY]
            data_for_load.append(pmid_node_data)
        self.prepared_data = data_for_load

    def _create_article_node_properties(self, pmid: str, pmid_data: dict) -> dict:
        """
        Creates the properties dictionary for an ARTICLE node.

        Args:
            pmid (str): The PMID of the article.
            pmid_data (dict): The data associated with the PMID.

        Returns:
            dict: A dictionary containing the properties for the ARTICLE node.
        """
        return {
            "properties": {
                "pmid": pmid,
                "year": pmid_data[self.YEAR_KEY],
                "question": pmid_data[self.QUESTION_KEY],
                "reasoning_required_pred": pmid_data[self.REASONING_REQUIRED_PRED_KEY],
                "reasoning_free_pred": pmid_data[self.REASONING_FREE_PRED_KEY],
                "final_decision": pmid_data[self.FINAL_DECISION_KEY],
                "long_answer": pmid_data[self.LONG_ANSWER_KEY],
            }
        }

    def _create_context_node_properties(
        self, contexts: list, labels: list
    ) -> List[dict]:
        """
        Creates the properties dictionaries for CONTEXT nodes.

        This includes splitting the context text into chunks, generating
        embeddings for each chunk, and preparing the properties dictionary.

        Args:
            contexts (list): A list of context strings.
            labels (list): A list of labels associated with the contexts.

        Returns:
            List[dict]: A list of dictionaries, each containing the properties
                         for a CONTEXT node.
        """
        contexts_str = " ".join(contexts)
        # create chunks of text
        chunks = self.text_splitter.split_text(contexts_str)
        # embed chunks
        embeddings = self.embedding_model.embed_documents(chunks)

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

    def _get_mesh_definition(self, mesh_term: str) -> str:
        """
        Retrieves the definition of a MESH term.

        Args:
            mesh_term (str): The MESH term.

        Returns:
            str: The definition of the MESH term, or an empty string if not found.
        """
        return self.mesh_terms.get(mesh_term, "")

    def _create_mesh_node_properties(self, mesh_term: str) -> dict:
        """
        Creates the properties dictionary for a MESH node.

        This includes fetching the definition and generating an embedding for the MESH term.

        Args:
            mesh_term (str): The MESH term.

        Returns:
            dict: A dictionary containing the properties for the MESH node.
        """
        mesh_definition = self._get_mesh_definition(mesh_term=mesh_term)
        embedding = self.embedding_model.embed_documents([mesh_definition])[0]
        return {
            "properties": {
                "name": mesh_term,
                "definition": mesh_definition,
                "embedding": embedding,
            }
        }

    def load_articles_and_contexts_to_graph(self):
        """
        Loads the prepared data into the graph database.

        This function iterates through the prepared data, creates article and
        context nodes, sets vector embedding properties for contexts, and creates
        relationships between the nodes. MESH nodes are assumed to be already
        loaded in advance.
        """
        for pmid_data in tqdm(self.prepared_data):
            # create nodes
            # article
            article_data = pmid_data[self.ARTICLE_LABEL]
            article_properties = article_data.get("properties")
            article_id = self.crud.create_node(
                label=self.ARTICLE_LABEL, properties=article_properties
            )

            # contexts
            meshes = pmid_data[self.MESH_LABEL]
            contexts_data_list = pmid_data[self.CONTEXT_LABEL]

            for context_data in contexts_data_list:
                context_properties = context_data.get("properties")
                embedding = context_properties.pop("embedding")
                context_id = self.crud.create_node(
                    label=self.CONTEXT_LABEL, properties=context_properties
                )
                self.crud.set_node_vector_property(
                    node_id=context_id,
                    property_name=self.embedding_property_name,
                    embedding=embedding,
                )
                # create relationship with article
                self.crud.create_relationship(
                    from_node_id=article_id,
                    to_node_id=context_id,
                    rel_type=self.HAS_CONTEXT_REL,
                )

                # create relationship with meshes
                for mesh_term in meshes:
                    self.crud.create_relationship_to_mesh_term(
                        from_node_id=context_id,
                        to_mesh_name=mesh_term,
                        rel_type=self.HAS_MESH_TERM_REL,
                    )

        def _get_context_similarities(self):
            # retrieve node embeddings with ids

            pass
