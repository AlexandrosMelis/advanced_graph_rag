import json
import os
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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

    def get_embeddings_from_graph(self) -> List[dict]:
        """
        Retrieves the embeddings and their corresponding IDs for all CONTEXT nodes in the graph.

        This method queries the graph database to fetch all CONTEXT nodes and their associated
        embeddings. It returns a list of dictionaries, where each dictionary contains the ID
        and the embedding of a CONTEXT node.

        Returns:
            List[dict]: A list of dictionaries. Each dictionary contains two keys:
                        - 'id': The element ID of a CONTEXT node (str).
                        - 'embedding': The vector embedding of the CONTEXT node (list of floats).
                        Returns an empty list if no context nodes are found.
        """
        EMBEDDINGS_RETRIEVAL_CYPHER = "MATCH (context:CONTEXT) RETURN elementId(context) AS id, context.embedding as embedding"
        with self.crud.driver.session() as session:
            results = session.run(EMBEDDINGS_RETRIEVAL_CYPHER)
            records = [dict(result) for result in results]
            return records

    def _convert_ids_to_list(self, records: list[dict]) -> list:
        return [record["id"] for record in records]

    def _convert_embeddings_to_array(self, records: list[dict]) -> np.array:
        return np.array([record["embedding"] for record in records])

    def _compute_similarities(self, embeddings):
        """
        Computes the cosine similarity matrix for the embeddings.

        Args:
        - embeddings (numpy.ndarray): An array of embeddings.

        Returns:
        - numpy.ndarray: A cosine similarity matrix for the embeddings.
        """
        return cosine_similarity(embeddings)

    def _filter_similarities(
        self, similarity_matrix, node_ids, threshold=0.7
    ) -> List[dict]:
        """
        Filters a cosine similarity matrix to find pairs of nodes with similarity above a threshold.

        Args:
            similarity_matrix (numpy.ndarray): The cosine similarity matrix.
            node_ids (list): A list of node IDs corresponding to the rows/columns of the matrix.
            threshold (float): The similarity threshold.

        Returns:
            list: A list of dictionaries, where each dictionary represents a pair of nodes with
                similarity above the threshold and contains:
                    - node1: The ID of the first node.
                    - node2: The ID of the second node.
                    - similarity: The cosine similarity between the two nodes.
        """

        num_nodes = len(node_ids)
        filtered_pairs = []

        # Check if similarity_matrix is empty
        if similarity_matrix.size == 0:
            return []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # Avoid duplicates and self-comparison
                similarity = similarity_matrix[i, j]
                if similarity >= threshold:
                    filtered_pairs.append(
                        {
                            "node1": node_ids[i],
                            "node2": node_ids[j],
                            "similarity": similarity,
                        }
                    )

        return filtered_pairs

    def load_similarities_to_graph(self) -> None:
        """
        Retrieves loaded embeddings + node ids from graph.
        Computes similarity matrix.
        Creates relationships only for nodes with similarity score > threshold.
        """
        records = self.get_embeddings_from_graph()
        node_ids = self._convert_ids_to_list(records)
        embeddings = self._convert_embeddings_to_array(records)
        similarity_matrix = self._compute_similarities(embeddings)
        similarity_relationships = self._filter_similarities(
            similarity_matrix=similarity_matrix, node_ids=node_ids, threshold=0.7
        )

        # load relationships
        for sim_relationship in similarity_relationships:
            self.crud.create_relationship(
                from_node_id=sim_relationship["node1"],
                to_node_id=sim_relationship["node2"],
                rel_type=self.IS_SIMILAR_TO_REL,
                properties={"score": sim_relationship["similarity"]},
            )
        logger.debug("Similarity relationships loaded!")
