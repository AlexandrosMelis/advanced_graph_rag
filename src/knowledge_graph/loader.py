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
from data_collection.text_splitter import TextSplitter
from knowledge_graph.crud import GraphCrud
from llms.embedding_model import (
    EmbeddingModel,
)  # IMPORTANT: Assumes embed_documents handles batching
from utils.utils import read_json_file

# --- Configuration ---
# Best practice: Move these to a dedicated config file or system
DEFAULT_SIMILARITY_THRESHOLD = 0.8
DEFAULT_MESH_DEFINITIONS_FILENAME = "mesh_term_definitions.json"


@dataclass
class PreparedNode:
    """Intermediate representation for nodes before batch creation."""

    temp_id: str  # Unique temporary ID within the batch (e.g., f"article_{pmid}")
    label: str
    properties: Dict[str, Any]
    embedding: List[float] | None = (
        None  # Store embedding temporarily if created during prep
    )


@dataclass
class PreparedRelationship:
    """Intermediate representation for relationships before batch creation."""

    from_temp_id: str
    to_temp_id: str
    rel_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


class GraphLoader:
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
        self.mesh_node_map: Dict[str, str] = {}  # Maps MESH term name -> graph NodeID

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
                try:
                    mesh_terms.update(article.get(self.MESHES_KEY, []))
                except Exception as e:
                    logger.error(
                        f"Error extracting MESH terms in sample with id: {sample['id']}, error: {e}"
                    )
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
        nodes_to_create: List[Dict] = []
        for term, definition in zip(mesh_terms_list, definitions):
            nodes_to_create.append(
                {
                    self.MESH_NAME_PROPERTY: term,
                    self.MESH_DEF_PROPERTY: definition,
                }
            )

        # Batch create nodes
        logger.info(f"Batch creating {len(nodes_to_create)} MESH nodes...")
        try:
            created_node_ids = self.crud.create_nodes_batch(
                label=self.MESH_LABEL, properties_list=nodes_to_create
            )
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
            vectors_to_set: List[Dict[str, Any]] = []
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
    def _prepare_nodes_and_chunks(
        self,
    ) -> Tuple[
        List[PreparedNode],
        List[PreparedNode],
        List[PreparedRelationship],
        List[PreparedRelationship],
        List[PreparedRelationship],
        Dict[str, Set[str]],
    ]:
        """
        Pass 1: Parses input data, prepares initial node/relationship structures
        using temporary IDs, and collects unique context chunks.
        """
        logger.info(
            "Pass 1: Preparing node/relationship structures and collecting context chunks..."
        )
        prepared_qa_nodes: List[PreparedNode] = []
        prepared_article_nodes: List[PreparedNode] = []
        prepared_qa_article_rels: List[PreparedRelationship] = []
        prepared_article_context_rels: List[PreparedRelationship] = []
        prepared_context_mesh_rels: List[PreparedRelationship] = []
        chunk_to_temp_context_ids: Dict[str, Set[str]] = {}
        # Keep track of prepared articles to avoid duplicates efficiently
        seen_article_temp_ids: Set[str] = set()

        for qa_sample in tqdm(self.data, desc="Pass 1: Preparing data"):
            # QA node
            qa_temp_id = f"qa_{qa_sample[self.ID_KEY]}"
            # QA node properties
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
                    continue

                article_temp_id = f"article_{pmid}"
                # Prepare article node only if not seen before in this batch
                if article_temp_id not in seen_article_temp_ids:
                    # Article node properties
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
                    seen_article_temp_ids.add(article_temp_id)

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
                        # Ensure unique temp ID even if PMIDs repeat across QA samples but refer to same article chunk conceptually
                        context_temp_id = f"context_{pmid}_{i}"

                        # Store chunk for batch embedding
                        if chunk_text not in chunk_to_temp_context_ids:
                            chunk_to_temp_context_ids[chunk_text] = set()
                        chunk_to_temp_context_ids[chunk_text].add(context_temp_id)
                        # temp_context_id_to_origin[context_temp_id] = (article_temp_id, i) # Not strictly needed anymore

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
                            if mesh_term in self.mesh_node_map:
                                prepared_context_mesh_rels.append(
                                    PreparedRelationship(
                                        from_temp_id=context_temp_id,
                                        to_temp_id=mesh_term,  # Store mesh name as target temp ID
                                        rel_type=self.HAS_MESH_TERM_REL,
                                    )
                                )
                            else:
                                logger.warning(
                                    f"MESH term '{mesh_term}' found in data but not in mesh_node_map. Skipping relationship from context {context_temp_id}."
                                )

        logger.info(
            f"Pass 1 completed. Prepared {len(prepared_qa_nodes)} QA, {len(prepared_article_nodes)} Article nodes. Found {len(chunk_to_temp_context_ids)} unique chunks."
        )
        return (
            prepared_qa_nodes,
            prepared_article_nodes,
            prepared_qa_article_rels,
            prepared_article_context_rels,
            prepared_context_mesh_rels,
            chunk_to_temp_context_ids,
        )

    def _embed_context_chunks(
        self, chunk_to_temp_context_ids: Dict[str, Set[str]]
    ) -> Dict[str, List[float]]:
        """
        Pass 2: Performs batch embedding of unique context chunks.
        """
        logger.info("Pass 2: Batch embedding context chunks...")
        unique_chunks = list(chunk_to_temp_context_ids.keys())
        embedding_map: Dict[str, List[float]] = {}

        if not unique_chunks:
            logger.info("Pass 2: No unique context chunks to embed.")
            return embedding_map

        try:
            chunk_embeddings = self.embedding_model.embed_documents(unique_chunks)
            if len(chunk_embeddings) != len(unique_chunks):
                logger.error(
                    f"Pass 2 Error: Mismatch between number of unique chunks ({len(unique_chunks)}) and embeddings ({len(chunk_embeddings)}). Unique chunks: {unique_chunks}"
                )
                # Decide how critical this is - returning empty map signifies failure to proceed.
                return {}  # Return empty map to signal failure

            embedding_map = {
                chunk: emb for chunk, emb in zip(unique_chunks, chunk_embeddings)
            }
            logger.info(
                f"Pass 2 completed. Embedded {len(unique_chunks)} unique chunks."
            )

        except Exception as e:
            logger.exception(f"Pass 2 Error: Failed to embed context chunks: {e}")
            return {}  # Return empty map to signal failure

        return embedding_map

    def _finalize_context_nodes(
        self,
        chunk_to_temp_context_ids: Dict[str, Set[str]],
        embedding_map: Dict[str, List[float]],
    ) -> List[PreparedNode]:
        """
        Pass 3: Creates PreparedNode objects for context nodes, associating them with their embeddings.
        """
        logger.info("Pass 3: Finalizing context node preparation...")
        prepared_context_nodes: List[PreparedNode] = []

        if not embedding_map:  # Check if embedding failed in previous step
            logger.warning(
                "Pass 3: Embedding map is empty. Skipping context node finalization."
            )
            return prepared_context_nodes

        for chunk_text, temp_ids in chunk_to_temp_context_ids.items():
            embedding = embedding_map.get(chunk_text)
            if embedding:
                for temp_id in temp_ids:
                    context_props = {self.CONTEXT_TEXT_PROPERTY: chunk_text}
                    prepared_context_nodes.append(
                        PreparedNode(
                            temp_id=temp_id,
                            label=self.CONTEXT_LABEL,
                            properties=context_props,
                            embedding=embedding,  # Store embedding temporarily
                        )
                    )
            else:
                # This case should ideally not happen if embedding_map check passed, but good for safety
                logger.warning(
                    f"Pass 3 Warning: Could not find embedding for chunk '{chunk_text[:50]}...'. Skipping context nodes: {temp_ids}"
                )

        logger.info(
            f"Pass 3 completed. Finalized {len(prepared_context_nodes)} context node preparations."
        )
        return prepared_context_nodes

    def _batch_create_nodes(
        self,
        prepared_qa_nodes: List[PreparedNode],
        prepared_article_nodes: List[PreparedNode],
        prepared_context_nodes: List[PreparedNode],
    ) -> Dict[str, str]:
        """
        Pass 4: Performs batch creation of QA, Article, and Context nodes using
        separate calls for each label type, as required by the updated CRUD method.
        Returns a map from temporary IDs to real graph NodeIDs.
        """
        logger.info("Pass 4: Starting batch node creation (per label type)...")
        temp_id_map: Dict[str, str] = {}
        total_nodes_created = 0

        # Helper function to process one batch for a specific label
        def process_batch(label: str, prepared_nodes: List[PreparedNode]):
            nonlocal total_nodes_created  # Allow modification of outer scope variable
            if not prepared_nodes:
                logger.debug(f"Pass 4: No nodes to create for label '{label}'.")
                return True  # Indicate success (nothing to do)

            logger.debug(
                f"Pass 4: Preparing {len(prepared_nodes)} nodes for label '{label}'..."
            )
            properties_list = [p.properties for p in prepared_nodes]

            try:
                created_node_ids = self.crud.create_nodes_batch(
                    label=label, properties_list=properties_list
                )

                if len(created_node_ids) == len(prepared_nodes):
                    for prepared_node, real_id in zip(prepared_nodes, created_node_ids):
                        temp_id_map[prepared_node.temp_id] = real_id
                    logger.debug(
                        f"Pass 4: Successfully created {len(created_node_ids)} nodes for label '{label}'."
                    )
                    total_nodes_created += len(created_node_ids)
                    return True  # Indicate success
                else:
                    # Error case: Mismatch in expected vs created IDs from CRUD method
                    logger.error(
                        f"Pass 4 Error: Node creation mismatch for label '{label}'. "
                        f"Expected {len(prepared_nodes)}, got {len(created_node_ids)} IDs. Cannot proceed reliably."
                    )
                    return False  # Indicate failure

            except Exception as e:
                # Catch potential exceptions from the CRUD method (Neo4jError, RuntimeError, etc.)
                logger.exception(
                    f"Pass 4 Error: Exception during batch node creation for label '{label}': {e}"
                )
                return False  # Indicate failure

        # --- Process each node type ---
        success = True
        if success:
            success = process_batch(self.QA_PAIR_LABEL, prepared_qa_nodes)

        if success:
            success = process_batch(self.ARTICLE_LABEL, prepared_article_nodes)

        if success:
            success = process_batch(self.CONTEXT_LABEL, prepared_context_nodes)

        # --- Final Check and Return ---
        if success:
            logger.info(
                f"Pass 4 completed. Successfully created a total of {total_nodes_created} nodes across all types."
            )
            return temp_id_map
        else:
            logger.error(
                "Pass 4 failed due to errors during batch node creation. Returning empty ID map."
            )
            return {}  # Return empty map to signal failure to the orchestrator

    def _batch_set_context_vectors(
        self, prepared_context_nodes: List[PreparedNode], temp_id_map: Dict[str, str]
    ) -> None:
        """
        Pass 5: Performs batch setting of vector properties for context nodes.
        """
        logger.info("Pass 5: Batch setting context vector properties...")
        # Original VectorData was: {'node_id': NodeID, 'property_name': str, 'embedding': List[float]}
        # Assumed VectorDataForBatch is: {'node_id': NodeID, 'embedding': List[float]}
        # Need to adapt if GraphCrud expects the former structure. Let's use VectorDataForBatch.

        context_vectors_to_set: List[Dict] = []  # Use VectorDataForBatch type
        for prepared_node in prepared_context_nodes:
            real_id = temp_id_map.get(prepared_node.temp_id)
            embedding = prepared_node.embedding  # Get stored embedding
            if real_id and embedding:
                context_vectors_to_set.append(
                    {
                        # self.VECTOR_NODE_ID_KEY: real_id, # Assuming keys match VectorDataForBatch
                        # self.VECTOR_EMBEDDING_KEY: embedding
                        "node_id": real_id,  # Adjust keys based on GraphCrud input expectations
                        "embedding": embedding,
                    }
                )

        if not context_vectors_to_set:
            logger.info("Pass 5: No context vectors to set.")
            return

        try:
            # Pass the correct property name if GraphCrud requires it
            self.crud.set_node_vector_properties_batch(
                vectors_data=context_vectors_to_set,
                property_name=self.EMBEDDING_PROPERTY,  # Pass the target property name
            )
            logger.info(
                f"Pass 5 completed. Attempted to set {len(context_vectors_to_set)} context vector properties."
            )
        except Exception as e:
            logger.exception(
                f"Pass 5 Error: Exception during batch setting of context vectors: {e}"
            )
            # Log error but potentially continue, as nodes/rels might still be useful

    def _batch_create_relationships(
        self,
        prepared_qa_article_rels: List[PreparedRelationship],
        prepared_article_context_rels: List[PreparedRelationship],
        prepared_context_mesh_rels: List[PreparedRelationship],
        temp_id_map: Dict[str, str],
        mesh_node_map: Dict[str, str],  # Pass mesh_node_map explicitly
    ) -> None:
        """
        Pass 6: Performs batch creation of all relationships, resolving temporary IDs.
        """
        logger.info("Pass 6: Batch creating relationships...")
        all_rels_to_create: List[Dict[str, Any]] = []

        # Helper to resolve and append relationship data
        def resolve_and_append(
            prep_rel: PreparedRelationship, to_id_map: Dict[str, str] = temp_id_map
        ):
            from_id = temp_id_map.get(prep_rel.from_temp_id)
            # Use the provided map (temp_id_map or mesh_node_map) for the target ID
            to_id = to_id_map.get(prep_rel.to_temp_id)
            if from_id and to_id:
                all_rels_to_create.append(
                    {
                        self.REL_FROM_KEY: from_id,
                        self.REL_TO_KEY: to_id,
                        self.REL_TYPE_KEY: prep_rel.rel_type,
                        self.REL_PROPS_KEY: prep_rel.properties,
                    }
                )
            # else: # Optional: Log if a relationship cannot be resolved
            #     logger.warning(f"Could not resolve relationship: {prep_rel.from_temp_id} -> {prep_rel.to_temp_id}")

        # Resolve QA -> Article rels
        for prep_rel in prepared_qa_article_rels:
            resolve_and_append(prep_rel)

        # Resolve Article -> Context rels
        for prep_rel in prepared_article_context_rels:
            resolve_and_append(prep_rel)

        # Resolve Context -> MESH rels (use mesh_node_map for target)
        for prep_rel in prepared_context_mesh_rels:
            resolve_and_append(prep_rel, mesh_node_map)

        if not all_rels_to_create:
            logger.info("Pass 6: No relationships to create.")
            return

        try:
            self.crud.create_relationships_batch(all_rels_to_create)
            logger.info(
                f"Pass 6 completed. Attempted to create {len(all_rels_to_create)} relationships."
            )
        except Exception as e:
            logger.exception(
                f"Pass 6 Error: Exception during batch relationship creation: {e}"
            )

    # --- Orchestrator Method ---

    def load_qa_articles_contexts(self) -> None:
        """
        Orchestrates the preparation and loading of QA pairs, Articles, and
        Context chunks into the graph using modularized, multi-pass batch operations.
        Requires `load_mesh_nodes` to be called first.
        """
        start_time = time.time()
        logger.info("Starting QA/Article/Context loading process...")

        if not self.data:
            logger.warning("No data provided. Aborting QA/Article/Context load.")
            return
        if not self.mesh_node_map:
            logger.error(
                "MESH node map is empty. Call `load_mesh_nodes` first. Aborting QA/Article/Context load as Context->MESH relationships cannot be resolved."
            )
            # Make this a hard stop
            return

        # Pass 1: Prepare nodes/rels and collect chunks
        (
            prepared_qa_nodes,
            prepared_article_nodes,
            prepared_qa_article_rels,
            prepared_article_context_rels,
            prepared_context_mesh_rels,
            chunk_to_temp_context_ids,
        ) = self._prepare_nodes_and_chunks()

        # Pass 2: Embed unique chunks
        embedding_map = self._embed_context_chunks(chunk_to_temp_context_ids)
        if (
            not embedding_map and chunk_to_temp_context_ids
        ):  # Check if embedding failed but there were chunks
            logger.error("Aborting load due to context chunk embedding failure.")
            return

        # Pass 3: Finalize context node data with embeddings
        prepared_context_nodes = self._finalize_context_nodes(
            chunk_to_temp_context_ids, embedding_map
        )

        # Pass 4: Batch create all nodes and get ID map
        temp_id_map = self._batch_create_nodes(
            prepared_qa_nodes, prepared_article_nodes, prepared_context_nodes
        )
        if not temp_id_map and (
            prepared_qa_nodes or prepared_article_nodes or prepared_context_nodes
        ):  # Check if creation failed but there were nodes
            logger.error("Aborting load due to batch node creation failure.")
            return

        # Pass 5: Batch set context vectors (proceed even if this fails, but log it)
        self._batch_set_context_vectors(prepared_context_nodes, temp_id_map)

        # Pass 6: Batch create all relationships
        self._batch_create_relationships(
            prepared_qa_article_rels,
            prepared_article_context_rels,
            prepared_context_mesh_rels,
            temp_id_map,
            self.mesh_node_map,
        )

        duration = time.time() - start_time
        logger.info(
            f"QA, Articles, Contexts loading process finished in {duration:.2f} seconds."
        )

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
    ) -> Tuple[List[str], np.ndarray]:
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
        self, similarity_matrix: np.ndarray, node_ids: List[str]
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
