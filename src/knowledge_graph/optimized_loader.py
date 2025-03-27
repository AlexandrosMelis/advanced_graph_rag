import json
import os
import time  # Optional: for simple timing feedback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Assumed project structure and imports (adjust as necessary)
from configs.config import ConfigPath, logger  # Assuming logger is configured
from data_preprocessing.text_splitter import TextSplitter
from knowledge_graph.crud import (
    GraphCrud,
)  # IMPORTANT: Assumes GraphCrud has batch methods
from llms.embedding_model import (
    EmbeddingModel,
)  # IMPORTANT: Assumes embed_documents handles batching
from utils.utils import read_json_file

# --- Type Aliases for Clarity ---
NodeID = str
Properties = Dict[str, Any]
NodeData = Dict[str, Any]  # e.g., {'label': str, 'properties': Properties}
RelationshipData = Dict[
    str, Any
]  # e.g., {'from_id': NodeID, 'to_id': NodeID, 'rel_type': str, 'properties': Properties}
VectorData = Dict[
    str, Any
]  # e.g., {'node_id': NodeID, 'property_name': str, 'embedding': List[float]}


# --- Configuration ---
# Best practice: Move these to a dedicated config file or system
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_MESH_DEFINITIONS_FILENAME = "mesh_term_definitions.json"


@dataclass
class PreparedNode:
    """Intermediate representation for nodes before batch creation."""

    temp_id: str  # Unique temporary ID within the batch (e.g., f"article_{pmid}")
    label: str
    properties: Properties
    embedding: List[float] | None = (
        None  # Store embedding temporarily if created during prep
    )


@dataclass
class PreparedRelationship:
    """Intermediate representation for relationships before batch creation."""

    from_temp_id: str
    to_temp_id: str
    rel_type: str
    properties: Properties = field(default_factory=dict)


class GraphLoaderOptimized:
    """
    Optimized class for preparing and loading structured data into a knowledge graph,
    utilizing batch operations for efficiency.

    Assumes the `GraphCrud` interface supports batch creation methods:
    - `create_nodes_batch(nodes_data: List[NodeData]) -> List[NodeID]`
    - `set_node_vector_properties_batch(vectors_data: List[VectorData]) -> None`
    - `create_relationships_batch(rels_data: List[RelationshipData]) -> None`
    - `get_nodes_with_property(label: str, property_name: str) -> List[Dict[str, Any]]`
        (returning list of {'id': NodeID, property_name: List[float]})

    Assumes `EmbeddingModel` has an efficient `embed_documents(texts: List[str])` method.
    """

    # --- Constants ---
    # Input Data Keys
    QUESTION_KEY = "question"
    ANSWER_KEY = "answer"
    ID_KEY = "id"
    ARTICLES_KEY = "articles"
    PMID_KEY = "pmid"
    TITLE_KEY = "title"
    ABSTRACT_KEY = "abstract"
    MESHES_KEY = "mesh_terms"

    # Node Labels
    QA_PAIR_LABEL = "QA_PAIR"
    ARTICLE_LABEL = "ARTICLE"
    CONTEXT_LABEL = "CONTEXT"
    MESH_LABEL = "MESH"

    # Relationship Types
    REFERS_TO = "REFERS_TO"  # QA -> Article (Example, adapt if needed)
    HAS_CONTEXT_REL = "HAS_CONTEXT"  # Article -> Context
    HAS_MESH_TERM_REL = "HAS_MESH_TERM"  # Context -> MESH
    IS_SIMILAR_TO_REL = "IS_SIMILAR_TO"  # Context -> Context

    # Property Names
    EMBEDDING_PROPERTY = "embedding"
    MESH_NAME_PROPERTY = "name"  # Property on MESH node holding the term name
    MESH_DEF_PROPERTY = "definition"
    QA_ID_PROPERTY = "id"  # Property on QA_PAIR node
    ARTICLE_PMID_PROPERTY = "pmid"  # Property on ARTICLE node
    CONTEXT_TEXT_PROPERTY = "text_content"
    SIMILARITY_SCORE_PROPERTY = "score"

    # Internal / Temporary Keys
    NODE_ID_KEY = "id"
    NODE_PROP_KEY = "properties"
    REL_FROM_KEY = "from_id"
    REL_TO_KEY = "to_id"
    REL_TYPE_KEY = "rel_type"
    REL_PROPS_KEY = "properties"
    VECTOR_NODE_ID_KEY = "node_id"
    VECTOR_PROP_NAME_KEY = "property_name"
    VECTOR_EMBEDDING_KEY = "embedding"

    def __init__(
        self,
        data: List[Dict],
        embedding_model: EmbeddingModel,
        text_splitter: TextSplitter,
        crud: GraphCrud,
        mesh_definitions_filename: str = DEFAULT_MESH_DEFINITIONS_FILENAME,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        external_data_dir: str = ConfigPath.EXTERNAL_DATA_DIR,  # Example: Get dir from config
    ):
        """
        Initializes the Optimized GraphLoader.

        Args:
            data: List of dictionaries structured as described in the original class docstring.
            embedding_model: Model for generating text embeddings (must support batching).
            text_splitter: Utility for splitting text into chunks.
            crud: Interface for graph database operations (must support batch methods).
            mesh_definitions_filename: Filename for MESH term definitions JSON.
            similarity_threshold: Threshold for creating IS_SIMILAR_TO relationships.
            external_data_dir: Directory containing external data like MESH definitions.
        """
        self.data = data
        self.embedding_model = embedding_model
        self.text_splitter = text_splitter
        self.crud = crud
        self.similarity_threshold = similarity_threshold

        # Load external data sources
        self.mesh_definitions_path = os.path.join(
            external_data_dir, mesh_definitions_filename
        )
        self.mesh_definitions = self._load_mesh_definitions()

        # Internal state
        self.distinct_mesh_terms: Set[str] = set()
        self.mesh_node_map: Dict[str, NodeID] = (
            {}
        )  # Maps MESH term name -> graph NodeID

        logger.info(f"GraphLoaderOptimized initialized with {len(data)} data samples.")

    def _load_mesh_definitions(self) -> Dict[str, str]:
        """Loads MESH term definitions from the specified JSON file."""
        try:
            definitions = read_json_file(file_path=self.mesh_definitions_path)
            logger.info(
                f"Loaded {len(definitions)} MESH definitions from {self.mesh_definitions_path}"
            )
            return definitions
        except FileNotFoundError:
            logger.error(
                f"MESH definitions file not found at: {self.mesh_definitions_path}"
            )
            return {}
        except json.JSONDecodeError:
            logger.error(
                f"Error decoding JSON from MESH definitions file: {self.mesh_definitions_path}"
            )
            return {}

    def _extract_distinct_meshes(self) -> None:
        """Extracts unique MESH terms from the input data."""
        logger.debug("Extracting distinct MESH terms...")
        mesh_terms = set()
        for sample in self.data:
            for article in sample.get(self.ARTICLES_KEY, []):
                mesh_terms.update(article.get(self.MESHES_KEY, []))
        self.distinct_mesh_terms = mesh_terms
        logger.info(f"Found {len(self.distinct_mesh_terms)} distinct MESH terms.")

    def _get_mesh_definition(self, mesh_term: str) -> str:
        """Retrieves the definition for a MESH term, returning empty string if not found."""
        return self.mesh_definitions.get(mesh_term, "")

    # --- MESH Node Loading (Batched) ---

    def load_mesh_nodes(self) -> None:
        """
        Prepares, embeds (in batch), and loads MESH nodes into the graph using batch operations.
        Populates `self.mesh_node_map`.
        """
        start_time = time.time()
        self._extract_distinct_meshes()  # Ensure distinct terms are known
        if not self.distinct_mesh_terms:
            logger.warning("No distinct MESH terms found to load.")
            return

        mesh_terms_list = sorted(
            list(self.distinct_mesh_terms)
        )  # Ensure consistent order
        definitions = [self._get_mesh_definition(term) for term in mesh_terms_list]

        # Batch embed definitions
        logger.info(f"Embedding {len(definitions)} MESH definitions...")
        embeddings = self.embedding_model.embed_documents(definitions)
        if len(embeddings) != len(mesh_terms_list):
            logger.error(
                f"Mismatch between number of mesh terms ({len(mesh_terms_list)}) and embeddings ({len(embeddings)}). Aborting MESH load."
            )
            return

        # Prepare batch data for node creation (without embeddings initially)
        nodes_to_create: List[NodeData] = []
        for term, definition in zip(mesh_terms_list, definitions):
            nodes_to_create.append(
                {
                    "label": self.MESH_LABEL,
                    "properties": {
                        self.MESH_NAME_PROPERTY: term,
                        self.MESH_DEF_PROPERTY: definition,
                        # Embedding will be added separately as a vector property
                    },
                }
            )

        # Batch create nodes
        logger.info(f"Batch creating {len(nodes_to_create)} MESH nodes...")
        try:
            created_node_ids = self.crud.create_nodes_batch(nodes_to_create)
            if len(created_node_ids) != len(mesh_terms_list):
                logger.error(
                    f"Mismatch creating MESH nodes. Expected {len(mesh_terms_list)}, got {len(created_node_ids)} IDs. Check CRUD implementation."
                )
                # Partial load occurred, state might be inconsistent. Decide on error handling (e.g., raise).
                return

            # Populate mesh_node_map
            self.mesh_node_map = {
                term: node_id
                for term, node_id in zip(mesh_terms_list, created_node_ids)
            }

            # Prepare batch data for vector properties
            vectors_to_set: List[VectorData] = []
            for node_id, embedding in zip(created_node_ids, embeddings):
                vectors_to_set.append(
                    {
                        self.VECTOR_NODE_ID_KEY: node_id,
                        self.VECTOR_PROP_NAME_KEY: self.EMBEDDING_PROPERTY,
                        self.VECTOR_EMBEDDING_KEY: embedding,
                    }
                )

            # Batch set vector properties
            logger.info(
                f"Batch setting vector properties for {len(vectors_to_set)} MESH nodes..."
            )
            self.crud.set_node_vector_properties_batch(vectors_to_set)

            duration = time.time() - start_time
            logger.info(
                f"MESH nodes loaded and vectors set successfully in {duration:.2f} seconds."
            )

        except Exception as e:
            logger.exception(f"Error during batch loading of MESH nodes: {e}")
            # Consider rollback or cleanup logic depending on GraphCrud capabilities

    # --- QA, Article, Context Loading (Batched - Multi-Pass) ---

    def load_qa_articles_contexts(self) -> None:
        """
        Prepares and loads QA pairs, Articles, and Context chunks (with embeddings)
        into the graph using batch operations in multiple passes.
        Requires `load_mesh_nodes` to be called first to populate `self.mesh_node_map`.
        """
        start_time = time.time()
        if not self.data:
            logger.warning("No data provided to load QA/Articles/Contexts.")
            return
        if not self.mesh_node_map:
            logger.warning(
                "MESH node map is empty. Call `load_mesh_nodes` first. Skipping Context->MESH relationships."
            )
            # Decide if this should be a hard error

        # --- Pass 1: Prepare all node/relationship data structures and collect context chunks ---
        logger.info(
            "Pass 1: Preparing node/relationship structures and collecting context chunks..."
        )
        prepared_qa_nodes: List[PreparedNode] = []
        prepared_article_nodes: List[PreparedNode] = []
        prepared_context_nodes: List[PreparedNode] = (
            []
        )  # Will be filled after embedding

        prepared_qa_article_rels: List[PreparedRelationship] = []
        prepared_article_context_rels: List[PreparedRelationship] = []
        prepared_context_mesh_rels: List[PreparedRelationship] = (
            []
        )  # Store with mesh name initially

        # Structure to hold chunk text and its associated temporary context ID
        chunks_to_embed: Dict[str, List[str]] = (
            {}
        )  # {temp_context_id: [chunk_text1, ...]} -> Incorrect, need {chunk_text: [temp_context_id1, ...]}
        chunk_to_temp_context_ids: Dict[str, Set[str]] = (
            {}
        )  # Map unique chunk text -> set of temp_context_ids using it
        temp_context_id_to_origin: Dict[str, Tuple[str, int]] = (
            {}
        )  # Map temp_context_id -> (temp_article_id, chunk_index)

        temp_id_counter = 0

        for qa_sample in tqdm(self.data, desc="Pass 1: Preparing data"):
            # QA Node
            qa_temp_id = f"qa_{qa_sample[self.ID_KEY]}"
            qa_props = {
                self.QA_ID_PROPERTY: qa_sample[self.ID_KEY],
                self.QUESTION_KEY: qa_sample[self.QUESTION_KEY],
                self.ANSWER_KEY: qa_sample[self.ANSWER_KEY],
            }
            prepared_qa_nodes.append(
                PreparedNode(
                    temp_id=qa_temp_id, label=self.QA_PAIR_LABEL, properties=qa_props
                )
            )

            # Article and Context Nodes/Relationships
            for article_data in qa_sample.get(self.ARTICLES_KEY, []):
                pmid = article_data.get(self.PMID_KEY)
                if not pmid:
                    continue  # Skip articles without PMID

                article_temp_id = f"article_{pmid}"
                # Avoid duplicate article node preparation if seen before (optional optimization)
                if not any(
                    p.temp_id == article_temp_id for p in prepared_article_nodes
                ):
                    article_props = {
                        self.ARTICLE_PMID_PROPERTY: pmid,
                        self.TITLE_KEY: article_data.get(self.TITLE_KEY, ""),
                    }
                    prepared_article_nodes.append(
                        PreparedNode(
                            temp_id=article_temp_id,
                            label=self.ARTICLE_LABEL,
                            properties=article_props,
                        )
                    )

                # QA -> Article Relationship
                prepared_qa_article_rels.append(
                    PreparedRelationship(
                        from_temp_id=qa_temp_id,
                        to_temp_id=article_temp_id,
                        rel_type=self.REFERS_TO,
                    )
                )

                # Context Nodes (Chunks)
                abstract = article_data.get(self.ABSTRACT_KEY)
                if abstract:
                    chunks = self.text_splitter.split_text(abstract)
                    for i, chunk_text in enumerate(chunks):
                        context_temp_id = (
                            f"context_{pmid}_{i}"  # Unique temp ID for the chunk
                        )

                        # Store chunk for batch embedding
                        if chunk_text not in chunk_to_temp_context_ids:
                            chunk_to_temp_context_ids[chunk_text] = set()
                        chunk_to_temp_context_ids[chunk_text].add(context_temp_id)
                        temp_context_id_to_origin[context_temp_id] = (
                            article_temp_id,
                            i,
                        )

                        # Article -> Context Relationship
                        prepared_article_context_rels.append(
                            PreparedRelationship(
                                from_temp_id=article_temp_id,
                                to_temp_id=context_temp_id,
                                rel_type=self.HAS_CONTEXT_REL,
                            )
                        )

                        # Context -> MESH Relationships (using mesh name for now)
                        for mesh_term in article_data.get(self.MESHES_KEY, []):
                            if (
                                mesh_term in self.mesh_node_map
                            ):  # Only add rel if MESH node exists
                                prepared_context_mesh_rels.append(
                                    PreparedRelationship(
                                        from_temp_id=context_temp_id,
                                        to_temp_id=mesh_term,  # Store mesh name as temporary ID target
                                        rel_type=self.HAS_MESH_TERM_REL,
                                    )
                                )
                            else:
                                logger.warning(
                                    f"MESH term '{mesh_term}' found in data but not in loaded MESH nodes. Skipping relationship from context {context_temp_id}."
                                )

        # --- Pass 2: Batch Embed Unique Chunks ---
        logger.info("Pass 2: Batch embedding context chunks...")
        unique_chunks = list(chunk_to_temp_context_ids.keys())
        chunk_embeddings: List[List[float]] = []
        if unique_chunks:
            chunk_embeddings = self.embedding_model.embed_documents(unique_chunks)
            if len(chunk_embeddings) != len(unique_chunks):
                logger.error(
                    "Mismatch between number of unique chunks and embeddings. Aborting context load."
                )
                return
        embedding_map = {
            chunk: emb for chunk, emb in zip(unique_chunks, chunk_embeddings)
        }

        # --- Pass 3: Prepare Context Nodes with Embeddings ---
        logger.info("Pass 3: Finalizing context node preparation...")
        temp_context_id_to_embedding: Dict[str, List[float]] = {}
        for chunk_text, temp_ids in chunk_to_temp_context_ids.items():
            embedding = embedding_map.get(chunk_text)
            if embedding:
                for temp_id in temp_ids:
                    context_props = {self.CONTEXT_TEXT_PROPERTY: chunk_text}
                    # Store embedding temporarily; will be set via batch vector property call
                    prepared_context_nodes.append(
                        PreparedNode(
                            temp_id=temp_id,
                            label=self.CONTEXT_LABEL,
                            properties=context_props,
                            embedding=embedding,
                        )
                    )
                    temp_context_id_to_embedding[temp_id] = embedding
            else:
                logger.warning(
                    f"Could not find embedding for chunk: '{chunk_text[:50]}...'. Skipping context nodes: {temp_ids}"
                )

        # --- Pass 4: Batch Create Nodes ---
        logger.info("Pass 4: Batch creating QA, Article, and Context nodes...")
        all_nodes_to_create_prepared = (
            prepared_qa_nodes + prepared_article_nodes + prepared_context_nodes
        )
        node_creation_data: List[NodeData] = [
            {"label": p.label, "properties": p.properties}
            for p in all_nodes_to_create_prepared
        ]
        temp_id_map: Dict[str, NodeID] = {}  # Maps temp_id -> real graph NodeID

        try:
            if node_creation_data:
                created_node_ids = self.crud.create_nodes_batch(node_creation_data)
                if len(created_node_ids) == len(all_nodes_to_create_prepared):
                    for prepared_node, real_id in zip(
                        all_nodes_to_create_prepared, created_node_ids
                    ):
                        temp_id_map[prepared_node.temp_id] = real_id
                    logger.info(f"Successfully created {len(created_node_ids)} nodes.")
                else:
                    logger.error(
                        f"Node creation mismatch. Expected {len(all_nodes_to_create_prepared)}, got {len(created_node_ids)} IDs."
                    )
                    # Decide on error handling - Cannot proceed without correct IDs
                    return
            else:
                logger.info("No new QA, Article, or Context nodes to create.")

        except Exception as e:
            logger.exception(f"Error during batch node creation: {e}")
            return  # Stop processing

        # --- Pass 5: Batch Set Context Vector Properties ---
        logger.info("Pass 5: Batch setting context vector properties...")
        context_vectors_to_set: List[VectorData] = []
        for prepared_node in prepared_context_nodes:
            real_id = temp_id_map.get(prepared_node.temp_id)
            embedding = prepared_node.embedding  # Get stored embedding
            if real_id and embedding:
                context_vectors_to_set.append(
                    {
                        self.VECTOR_NODE_ID_KEY: real_id,
                        self.VECTOR_PROP_NAME_KEY: self.EMBEDDING_PROPERTY,
                        self.VECTOR_EMBEDDING_KEY: embedding,
                    }
                )

        try:
            if context_vectors_to_set:
                self.crud.set_node_vector_properties_batch(context_vectors_to_set)
                logger.info(
                    f"Successfully set {len(context_vectors_to_set)} context vector properties."
                )
            else:
                logger.info("No context vectors to set.")
        except Exception as e:
            logger.exception(f"Error during batch setting of context vectors: {e}")
            # Non-critical? Nodes exist, but are not queryable by vector similarity.

        # --- Pass 6: Batch Create Relationships ---
        logger.info("Pass 6: Batch creating relationships...")
        all_rels_to_create: List[RelationshipData] = []

        # Resolve QA -> Article rels
        for prep_rel in prepared_qa_article_rels:
            from_id = temp_id_map.get(prep_rel.from_temp_id)
            to_id = temp_id_map.get(prep_rel.to_temp_id)
            if from_id and to_id:
                all_rels_to_create.append(
                    {
                        self.REL_FROM_KEY: from_id,
                        self.REL_TO_KEY: to_id,
                        self.REL_TYPE_KEY: prep_rel.rel_type,
                        self.REL_PROPS_KEY: prep_rel.properties,
                    }
                )

        # Resolve Article -> Context rels
        for prep_rel in prepared_article_context_rels:
            from_id = temp_id_map.get(prep_rel.from_temp_id)
            to_id = temp_id_map.get(prep_rel.to_temp_id)
            if from_id and to_id:
                all_rels_to_create.append(
                    {
                        self.REL_FROM_KEY: from_id,
                        self.REL_TO_KEY: to_id,
                        self.REL_TYPE_KEY: prep_rel.rel_type,
                        self.REL_PROPS_KEY: prep_rel.properties,
                    }
                )

        # Resolve Context -> MESH rels (using mesh_node_map)
        for prep_rel in prepared_context_mesh_rels:
            from_id = temp_id_map.get(prep_rel.from_temp_id)
            mesh_name = prep_rel.to_temp_id  # Stored mesh name here
            to_id = self.mesh_node_map.get(mesh_name)  # Resolve using the map
            if from_id and to_id:
                all_rels_to_create.append(
                    {
                        self.REL_FROM_KEY: from_id,
                        self.REL_TO_KEY: to_id,
                        self.REL_TYPE_KEY: prep_rel.rel_type,
                        self.REL_PROPS_KEY: prep_rel.properties,
                    }
                )

        try:
            if all_rels_to_create:
                self.crud.create_relationships_batch(all_rels_to_create)
                logger.info(
                    f"Successfully created {len(all_rels_to_create)} relationships."
                )
            else:
                logger.info("No relationships to create.")

            duration = time.time() - start_time
            logger.info(
                f"QA, Articles, Contexts, and their relationships loaded successfully in {duration:.2f} seconds."
            )

        except Exception as e:
            logger.exception(f"Error during batch relationship creation: {e}")

    # --- Similarity Loading (Batched) ---

    def _get_context_embeddings_from_graph(self) -> List[Dict[str, Any]]:
        """
        Retrieves context node IDs and their embeddings from the graph
        using the GraphCrud interface.
        """
        logger.debug(
            f"Retrieving embeddings for label '{self.CONTEXT_LABEL}' with property '{self.EMBEDDING_PROPERTY}'..."
        )
        try:
            # Assumes crud method returns list of {'id': NodeID, self.EMBEDDING_PROPERTY: List[float]}
            records = self.crud.get_nodes_with_property(
                label=self.CONTEXT_LABEL, property_name=self.EMBEDDING_PROPERTY
            )
            # Ensure the embedding key matches what subsequent methods expect
            return [
                {
                    self.NODE_ID_KEY: r[self.NODE_ID_KEY],
                    self.EMBEDDING_PROPERTY: r[self.EMBEDDING_PROPERTY],
                }
                for r in records
            ]
        except Exception as e:
            logger.exception(f"Failed to retrieve context embeddings from graph: {e}")
            return []

    def _convert_records_to_ids_and_embeddings(
        self, records: List[Dict]
    ) -> Tuple[List[NodeID], np.ndarray]:
        """Separates records into lists of IDs and a NumPy array of embeddings."""
        if not records:
            return [], np.array([])
        node_ids = [record[self.NODE_ID_KEY] for record in records]
        embeddings = np.array([record[self.EMBEDDING_PROPERTY] for record in records])
        return node_ids, embeddings

    def _compute_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """Computes the cosine similarity matrix for the embeddings."""
        if embeddings.size == 0 or embeddings.shape[0] < 2:
            return np.array([])
        logger.debug(
            f"Computing similarity matrix for {embeddings.shape[0]} embeddings..."
        )
        return cosine_similarity(embeddings)

    def _filter_similarities(
        self, similarity_matrix: np.ndarray, node_ids: List[NodeID]
    ) -> List[Dict]:
        """
        Filters the similarity matrix to find pairs above the threshold.
        Returns data suitable for batch relationship creation.
        """
        if similarity_matrix.size == 0 or not node_ids or len(node_ids) < 2:
            return []

        num_nodes = len(node_ids)
        filtered_rels_data = []

        # Use numpy's triu_indices to avoid duplicates and self-comparison efficiently
        rows, cols = np.triu_indices(num_nodes, k=1)
        similarities = similarity_matrix[rows, cols]

        above_threshold_indices = np.where(similarities >= self.similarity_threshold)[0]

        for index in tqdm(above_threshold_indices, desc="Filtering similarities"):
            i, j = rows[index], cols[index]
            similarity_score = similarities[index]
            filtered_rels_data.append(
                {
                    self.REL_FROM_KEY: node_ids[i],
                    self.REL_TO_KEY: node_ids[j],
                    self.REL_TYPE_KEY: self.IS_SIMILAR_TO_REL,
                    self.REL_PROPS_KEY: {
                        self.SIMILARITY_SCORE_PROPERTY: float(similarity_score)
                    },  # Ensure float
                }
            )

        logger.info(
            f"Found {len(filtered_rels_data)} similarity relationships above threshold {self.similarity_threshold}."
        )
        return filtered_rels_data

    def load_similarities_to_graph(self) -> None:
        """
        Retrieves context embeddings, computes similarities, filters based on
        threshold, and loads IS_SIMILAR_TO relationships using batch operations.
        """
        start_time = time.time()
        logger.info("Starting similarity relationship loading...")

        # 1. Get embeddings
        records = self._get_context_embeddings_from_graph()
        if not records:
            logger.warning(
                "No context embeddings found in graph. Cannot compute similarities."
            )
            return

        # 2. Convert
        node_ids, embeddings = self._convert_records_to_ids_and_embeddings(records)

        # 3. Compute similarity matrix
        similarity_matrix = self._compute_similarities(embeddings)
        if similarity_matrix.size == 0:
            logger.warning(
                "Could not compute similarity matrix (likely <= 1 embedding)."
            )
            return

        # 4. Filter pairs above threshold
        similarity_rels_data = self._filter_similarities(similarity_matrix, node_ids)

        # 5. Batch create relationships
        if similarity_rels_data:
            logger.info(
                f"Batch creating {len(similarity_rels_data)} similarity relationships..."
            )
            try:
                self.crud.create_relationships_batch(similarity_rels_data)
                duration = time.time() - start_time
                logger.info(
                    f"Similarity relationships loaded successfully in {duration:.2f} seconds."
                )
            except Exception as e:
                logger.exception(
                    f"Error during batch creation of similarity relationships: {e}"
                )
        else:
            logger.info("No similarity relationships to load.")

    # --- Main Execution Method ---

    def load_all(self, load_similarities: bool = True) -> None:
        """
        Executes the full graph loading pipeline: MESH nodes, QA/Articles/Contexts,
        and optionally similarity relationships.

        Args:
            load_similarities: If True, computes and loads context similarity relationships.
        """
        logger.info("Starting full graph loading process...")

        self.load_mesh_nodes()
        self.load_qa_articles_contexts()

        if load_similarities:
            self.load_similarities_to_graph()
        else:
            logger.info("Skipping similarity relationship loading as requested.")

        logger.info("Full graph loading process completed.")


# --- Example Usage (Illustrative) ---
if __name__ == "__main__":
    # This is a placeholder for how you might use the class
    # Replace with your actual data loading, model/crud instantiation

    logger.info("Setting up dependencies (Mocks/Placeholders)...")

    # Mock/Placeholder implementations (Replace with real ones)
    class MockEmbeddingModel(EmbeddingModel):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            logger.debug(f"Mock embed_documents called for {len(texts)} texts.")
            # Return embeddings of consistent dimension, e.g., 3 dimensions
            return [list(np.random.rand(3).astype(float)) for _ in texts]

    class MockTextSplitter(TextSplitter):
        def split_text(self, text: str) -> List[str]:
            # Simple splitting for example
            return [text[i : i + 100] for i in range(0, len(text), 100)]

    class MockGraphCrud(GraphCrud):
        _node_counter = 0
        _nodes = {}
        _rels = []

        def __init__(self):
            # Add necessary driver/connection setup if needed
            pass

        def create_nodes_batch(self, nodes_data: List[NodeData]) -> List[NodeID]:
            ids = []
            for node_d in nodes_data:
                self._node_counter += 1
                new_id = f"mock_node_{self._node_counter}"
                self._nodes[new_id] = {
                    "label": node_d["label"],
                    "properties": node_d["properties"],
                }
                ids.append(new_id)
            logger.debug(
                f"Mock created {len(ids)} nodes. Total nodes: {len(self._nodes)}"
            )
            return ids

        def set_node_vector_properties_batch(
            self, vectors_data: List[VectorData]
        ) -> None:
            for vec_d in vectors_data:
                node_id = vec_d[self.VECTOR_NODE_ID_KEY]
                prop_name = vec_d[self.VECTOR_PROP_NAME_KEY]
                if node_id in self._nodes:
                    if "vectors" not in self._nodes[node_id]:
                        self._nodes[node_id]["vectors"] = {}
                    self._nodes[node_id]["vectors"][prop_name] = vec_d[
                        self.VECTOR_EMBEDDING_KEY
                    ]
                else:
                    logger.warning(f"Mock set_vector: Node {node_id} not found.")
            logger.debug(f"Mock set vectors for {len(vectors_data)} nodes.")

        def create_relationships_batch(self, rels_data: List[RelationshipData]) -> None:
            valid_rels = 0
            for rel_d in rels_data:
                if (
                    rel_d[self.REL_FROM_KEY] in self._nodes
                    and rel_d[self.REL_TO_KEY] in self._nodes
                ):
                    self._rels.append(rel_d)
                    valid_rels += 1
                else:
                    logger.warning(
                        f"Mock create_rel: Node {rel_d[self.REL_FROM_KEY]} or {rel_d[self.REL_TO_KEY]} not found."
                    )
            logger.debug(
                f"Mock created {valid_rels}/{len(rels_data)} relationships. Total rels: {len(self._rels)}"
            )

        def get_nodes_with_property(
            self, label: str, property_name: str
        ) -> List[Dict[str, Any]]:
            results = []
            for node_id, node_data in self._nodes.items():
                if node_data["label"] == label:
                    # Check if vector property exists
                    if "vectors" in node_data and property_name in node_data["vectors"]:
                        results.append(
                            {
                                GraphLoaderOptimized.NODE_ID_KEY: node_id,
                                property_name: node_data["vectors"][property_name],
                            }
                        )
                    # Also check regular properties if needed, adjust logic based on 'property_name'
            logger.debug(
                f"Mock get_nodes_with_property found {len(results)} nodes for label '{label}' property '{property_name}'."
            )
            return results

        def close(self):
            logger.debug("Mock CRUD closed.")
            pass

    # Sample Data (simplified)
    sample_input_data = [
        {
            "question": "Q1?",
            "answer": "A1.",
            "id": "q1",
            "articles": [
                {
                    "pmid": "p1",
                    "title": "T1",
                    "abstract": "Abstract content one chunk. Another part.",
                    "mesh_terms": ["MeshA", "MeshB"],
                },
                {
                    "pmid": "p2",
                    "title": "T2",
                    "abstract": "Different abstract text here.",
                    "mesh_terms": ["MeshB", "MeshC"],
                },
            ],
        },
        {
            "question": "Q2?",
            "answer": "A2.",
            "id": "q2",
            "articles": [
                {
                    "pmid": "p3",
                    "title": "T3",
                    "abstract": "Yet another abstract, similar to the first maybe.",
                    "mesh_terms": ["MeshA", "MeshD"],
                },
                {
                    "pmid": "p2",
                    "title": "T2",
                    "abstract": "Different abstract text here.",
                    "mesh_terms": ["MeshB", "MeshC"],
                },  # Duplicate article reference
            ],
        },
    ]

    # Create dummy mesh definitions file
    mesh_def_file = "mock_mesh_definitions.json"
    external_dir = "."  # Use current dir for example
    mock_defs = {
        "MeshA": "Definition A",
        "MeshB": "Definition B",
        "MeshC": "Definition C",
        "MeshD": "Definition D",
    }
    with open(os.path.join(external_dir, mesh_def_file), "w") as f:
        json.dump(mock_defs, f)

    # Instantiate
    mock_embedder = MockEmbeddingModel()
    mock_splitter = MockTextSplitter()
    mock_crud = MockGraphCrud()

    loader = GraphLoaderOptimized(
        data=sample_input_data,
        embedding_model=mock_embedder,
        text_splitter=mock_splitter,
        crud=mock_crud,
        mesh_definitions_filename=mesh_def_file,
        external_data_dir=external_dir,
        similarity_threshold=0.5,  # Example threshold
    )

    # Run the loading process
    loader.load_all(load_similarities=True)

    # Cleanup dummy file
    os.remove(os.path.join(external_dir, mesh_def_file))

    # Inspect mock_crud._nodes and mock_crud._rels to verify results (optional)
    # print("\n--- Mock Graph State ---")
    # print("Nodes:", json.dumps(mock_crud._nodes, indent=2))
    # print("\nRels:", json.dumps(mock_crud._rels, indent=2))

    mock_crud.close()
    logger.info("Example finished.")
