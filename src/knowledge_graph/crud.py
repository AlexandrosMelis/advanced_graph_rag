import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple  # Added Optional

from neo4j.exceptions import Neo4jError

from configs.config import logger
from knowledge_graph.connection import Neo4jConnection


class GraphCrud:
    def __init__(self, neo4j_connection: Neo4jConnection):
        self.driver = neo4j_connection.get_driver()

    def close(self):
        """
        Close the connection to the Neo4j db
        """
        self.drive.close()

    def create_node(self, label: str, properties: dict) -> str:
        """
        Creates a node with given label and entities.
        Returns the Neo4j internal node elementId.
        """
        if properties is None:
            properties = {}
            raise ValueError("Properties can not be empty!")

        query = f"CREATE (n:{label} $properties) RETURN elementId(n) as id"
        with self.driver.session() as session:
            result = session.run(query, properties=properties)
            record = result.single()
            if record is None:
                raise RuntimeError(f"Failed to create node for label {label}")
            logger.debug(
                f"Created node with label '{label}' and elementId: {record['id']}"
            )
            return record["id"]

    def update_node(self, node_id: str, properties: dict) -> None:
        """
        Updates a node's properties - merging new properties in.
        """
        query = "MATCH (n) WHERE elementId(n) = $node_id SET n += $properties"
        with self.driver.session() as session:
            session.run(query, node_id=node_id, properties=properties)
            logger.debug(f"Updated node {node_id} with properties {properties}")

    def delete_node(self, node_id: str) -> None:
        """
        Deletes a node - detaches all the relationships.
        """
        query = "MATCH (n) WHERE elementId(n) = $node_id DETACH DELETE n"
        with self.driver.session() as session:
            session.run(query, node_id=node_id)
        logger.debug(f"Deleted node {node_id}")

    def create_relationship(
        self, from_node_id: str, to_node_id: str, rel_type: str, properties: dict = None
    ) -> str:
        """
        Creates a relationship of the given type between two nodes identified by their Neo4j ids.
        Optionally accepts a dictionary of relationship properties.
        Returns the internal Neo4j relationship id.
        """
        if properties is None:
            properties = {}

        query = (
            f"MATCH (a), (b) "
            f"WHERE elementId(a) = $from_node_id AND elementId(b) = $to_node_id "
            f"CREATE (a)-[r:{rel_type} $properties]->(b) "
            f"RETURN elementId(r) as id"
        )
        with self.driver.session() as session:
            result = session.run(
                query,
                from_node_id=from_node_id,
                to_node_id=to_node_id,
                properties=properties,
            )
            record = result.single()
            if record is None:
                raise RuntimeError(
                    f"Failed to create relationship {rel_type} for elementIds: ({from_node_id}, {to_node_id})"
                )
            logger.debug(
                f"Created relationship `{rel_type}` from node {from_node_id} to {to_node_id}"
            )
        return record["id"]

    def create_relationship_to_mesh_term(
        self,
        from_node_id: str,
        to_mesh_name: str,
        rel_type: str,
        properties: dict = None,
    ) -> str:
        if properties is None:
            properties = {}

        query = (
            f"MATCH (a), (mesh: MESH) "
            f"WHERE elementId(a) = $from_node_id AND mesh.name = $to_mesh_name "
            f"CREATE (a)-[r:{rel_type} $properties]->(mesh) "
            f"RETURN elementId(r) as id"
        )

        with self.driver.session() as session:
            result = session.run(
                query,
                from_node_id=from_node_id,
                to_mesh_name=to_mesh_name,
                properties=properties,
            )
            record = result.single()
            if record is None:
                raise RuntimeError(
                    f"Failed to create relationship {rel_type} for elementIds: ({from_node_id}, {to_node_id})"
                )
            logger.debug(
                f"Created relationship `{rel_type}` from node {from_node_id} to {to_mesh_name}"
            )
        return record["id"]

    def set_node_vector_property(
        self, node_id: str, embedding: list, property_name: str = "embedding"
    ):
        """
        Sets a vector property on a node using Neo4j's vector procedure.
        Note: `property_name` is interpolated directly - ensure it is trusted
        """
        query = (
            f"MATCH (n) WHERE elementId(n) = $node_id "
            f"CALL db.create.setNodeVectorProperty(n, '{property_name}', $embedding) "
            f"RETURN n"
        )
        with self.driver.session() as session:
            session.run(query, node_id=node_id, embedding=embedding)

    def create_vector_index(
        self,
        index_name: str,
        label: str,
        property_name: str,
        dimensions: int = 768,
        similarity_function: str = "cosine",
    ):
        """
        Creates a vector index if it doesn't exist.
        The method assumes you want to index nodes with a given label and vector property
        """
        query = (
            f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
            f"FOR (n: {label}) "
            f"ON n.{property_name} "
            f"OPTIONS {{indexConfig {{`vector.dimensions`: {dimensions}, `vector.similarity_function`: '{similarity_function}'}}}} "
        )
        with self.driver.session() as session:
            session.run(query)
            logger.debug(
                f"Vector index: {index_name} created, for label: {label} property_name: {property_name}, dimensions: {dimensions}"
            )

    def get_embeddings_from_label(self, label: str) -> list:
        node_variable_name = label.lower()
        query = f"MATCH ({node_variable_name}:{label}) RETURN elementId({node_variable_name}) as id, {node_variable_name}.embedding as embedding"
        result = self.driver.run(query)
        record = result.single()
        return record

    # --- NEW BATCH METHODS ---

    def create_nodes_batch(
        self, label: str, properties_list: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Creates multiple nodes of the *same label* in a single transaction using
        standard Cypher `UNWIND...CREATE...SET`.

        Args:
            label: The common label for all nodes in this batch.
            properties_list: A list of dictionaries, where each dict contains
                             the properties for one node.

        Returns:
            A list of the elementIds of the created nodes, in the same order as input properties_list.

        Raises:
            ValueError: If label is empty or properties_list is empty.
            Neo4jError: For database-related issues.
        """
        if not label:
            raise ValueError("Label cannot be empty for batch node creation.")
        if not properties_list:
            logger.info(
                f"create_nodes_batch called for label '{label}' with empty list. No nodes created."
            )
            return []

        # Standard Cypher approach for batch creation with a fixed label
        # Use backticks for label safety
        query = f"""
        UNWIND $props_list AS props
        CREATE (n:`{label}`)
        SET n = props
        RETURN elementId(n) AS id
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, props_list=properties_list)
                # Collect all IDs - data() fetches all results
                created_ids = [record["id"] for record in result.data()]

                summary = result.consume()
                expected_count = len(properties_list)
                if len(created_ids) == expected_count:
                    logger.info(
                        f"Batch created {len(created_ids)} nodes with label '{label}' successfully."
                    )  # Removed Summary counters as they might not be accurate for SET
                    return created_ids
                else:
                    # This case might indicate partial failure
                    logger.error(
                        f"Batch node creation mismatch for label '{label}'. Expected {expected_count}, got {len(created_ids)} IDs. Summary: {summary}"
                    )
                    raise RuntimeError(
                        f"Batch node creation failed for label '{label}': Expected {expected_count} IDs, got {len(created_ids)}"
                    )

        except Neo4jError as e:
            logger.error(
                f"Database error during batch node creation (Label: '{label}'): {e}",
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error during batch node creation (Label: '{label}'): {e}",
                exc_info=True,
            )
            raise

    def set_node_vector_properties_batch(
        self, vectors_data: List[Dict[str, Any]], property_name: str = "embedding"
    ) -> None:
        """
        Sets a vector property on multiple nodes in a single transaction using UNWIND.
        Assumes all items in the batch target the *same* vector property name.
        Uses Neo4j procedure `db.create.setNodeVectorProperty`.

        Args:
            vectors_data: A list of dictionaries, where each dict has 'node_id' and 'embedding'.
            property_name: The name of the vector property to set.

        Raises:
            RuntimeError: If the vector procedure call fails.
            Neo4jError: For database-related issues.
        """
        # This method remains the same as it relies on the specific procedure, not standard CREATE/SET
        if not vectors_data:
            logger.info(
                "set_node_vector_properties_batch called with empty list. No vectors set."
            )
            return
        query = """
        UNWIND $vectors_list AS vector_data
        MATCH (n) WHERE elementId(n) = vector_data.node_id
        CALL db.create.setNodeVectorProperty(n, $property_name, vector_data.embedding)
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    query, vectors_list=vectors_data, property_name=property_name
                )
                summary = result.consume()
                logger.info(
                    f"Batch set vector property '{property_name}' attempt completed. Summary: {summary.counters}"
                )
        # ... (keep the rest of the error handling from the previous version) ...
        except Neo4jError as e:
            if (
                "There is no procedure with the name `db.create.setNodeVectorProperty`"
                in str(e)
            ):
                logger.error(
                    "Neo4j vector procedure 'db.create.setNodeVectorProperty' not found. Batch vector setting failed.",
                    exc_info=True,
                )
                raise RuntimeError(
                    "Vector procedure 'db.create.setNodeVectorProperty' not found."
                ) from e
            elif "vector.dimensions" in str(e):
                logger.error(
                    f"Vector dimension mismatch error setting property '{property_name}'. Ensure index and embedding dimensions match. Error: {e}",
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Vector dimension mismatch for property '{property_name}'."
                ) from e
            else:
                logger.error(
                    f"Database error during batch vector property setting ('{property_name}'): {e}",
                    exc_info=True,
                )
                raise
        except Exception as e:
            logger.error(
                f"Unexpected error during batch vector property setting ('{property_name}'): {e}",
                exc_info=True,
            )
            raise

    def create_relationships_batch(self, rels_data: List[Dict[str, Any]]) -> None:
        """
        Creates multiple relationships in potentially multiple transactions (one per
        relationship type) using standard Cypher `UNWIND...CREATE...SET`.

        Args:
            rels_data: A list of dictionaries, each containing 'from_id', 'to_id',
                       'rel_type', and 'properties'.

        Raises:
            ValueError: If rel_type is missing in any item.
            Neo4jError: For database-related issues.
        """
        if not rels_data:
            logger.info(
                "create_relationships_batch called with empty list. No relationships created."
            )
            return

        # Group relationships by type
        rels_by_type: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rel_data in rels_data:
            rel_type = rel_data.get("rel_type")
            if not rel_type:
                raise ValueError(
                    f"Missing 'rel_type' in relationship data item: {rel_data}"
                )
            # Store only necessary info for the specific type's query
            rels_by_type[rel_type].append(
                {
                    "from_id": rel_data["from_id"],
                    "to_id": rel_data["to_id"],
                    "properties": rel_data.get(
                        "properties", {}
                    ),  # Ensure properties exist
                }
            )

        total_created_count = 0
        # Execute one batch query per relationship type
        with self.driver.session() as session:
            for rel_type, type_specific_rels in rels_by_type.items():
                if not type_specific_rels:
                    continue

                # Use backticks for relationship type safety
                query = f"""
                UNWIND $rels_list AS rel_data
                MATCH (a), (b) WHERE elementId(a) = rel_data.from_id AND elementId(b) = rel_data.to_id
                CREATE (a)-[r:`{rel_type}`]->(b)
                SET r = rel_data.properties
                """
                try:
                    result = session.run(query, rels_list=type_specific_rels)
                    summary = result.consume()
                    created_count = summary.counters.relationships_created
                    total_created_count += created_count
                    if created_count == len(type_specific_rels):
                        logger.debug(
                            f"Batch created {created_count} relationships of type '{rel_type}'."
                        )
                    else:
                        # Log discrepancy, might indicate missing nodes
                        logger.warning(
                            f"Batch relationship creation for type '{rel_type}': Attempted {len(type_specific_rels)}, created {created_count}. Some nodes might be missing."
                        )
                except Neo4jError as e:
                    logger.error(
                        f"Database error during batch relationship creation (Type: '{rel_type}'): {e}",
                        exc_info=True,
                    )
                    raise  # Re-raise after logging
                except Exception as e:
                    logger.error(
                        f"Unexpected error during batch relationship creation (Type: '{rel_type}'): {e}",
                        exc_info=True,
                    )
                    raise  # Re-raise after logging

        logger.info(
            f"Batch relationship creation process completed. Total relationships created across all types: {total_created_count}."
        )

    def get_nodes_with_property(
        self, label: str, property_name: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the elementId and a specific property for all nodes with a given label
        where the property exists and is not null.

        Args:
            label: The node label to match.
            property_name: The name of the property to retrieve.

        Returns:
            A list of dictionaries, each containing 'id' (the elementId) and
            the specified 'property_name' with its value.

        Raises:
            Neo4jError: For database-related issues.
        """
        # Use backticks for safety around label and property name interpolation
        # Note: This is generally safe if label/property_name are from trusted sources,
        # but parameterization is preferred if possible (not standard for labels/keys).
        query = f"""
        MATCH (n:`{label}`)
        WHERE n.`{property_name}` IS NOT NULL
        RETURN elementId(n) AS id, n.`{property_name}` AS `{property_name}`
        """
        try:
            with self.driver.session() as session:
                result = session.run(query)
                # Use result.data() to fetch all records directly as dicts
                records = result.data()
                logger.debug(
                    f"Retrieved {len(records)} nodes with label '{label}' and property '{property_name}'."
                )
                return records
        except Neo4jError as e:
            logger.error(
                f"Database error retrieving nodes (Label: {label}, Property: {property_name}): {e}",
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error retrieving nodes (Label: {label}, Property: {property_name}): {e}",
                exc_info=True,
            )
            raise

    # --- Existing Vector/Other Methods (Keep or Refactor as needed) ---

    def set_node_vector_property(
        self, node_id: str, embedding: list, property_name: str = "embedding"
    ):
        """
        Sets a vector property on a *single* node using Neo4j's vector procedure.
        """
        # Note property_name interpolation - ensure it's trusted.
        # Backticks added for safety.
        query = (
            f"MATCH (n) WHERE elementId(n) = $node_id "
            # The procedure expects the property name as a literal string argument.
            f"CALL db.create.setNodeVectorProperty(n, '{property_name}', $embedding) "
            f"RETURN count(n) as updated_count"  # Return something to check success
        )
        try:
            with self.driver.session() as session:
                result = session.run(query, node_id=node_id, embedding=embedding)
                record = result.single()
                if record and record["updated_count"] > 0:
                    logger.debug(
                        f"Set vector property '{property_name}' for node {node_id}"
                    )
                else:
                    logger.warning(
                        f"Node {node_id} not found or vector property '{property_name}' not set."
                    )

        except Neo4jError as e:
            if (
                "There is no procedure with the name `db.create.setNodeVectorProperty`"
                in str(e)
            ):
                logger.error(
                    "Neo4j vector procedure 'db.create.setNodeVectorProperty' not found.",
                    exc_info=True,
                )
                raise RuntimeError(
                    "Vector procedure 'db.create.setNodeVectorProperty' not found."
                ) from e
            else:
                logger.error(
                    f"Database error setting single vector property '{property_name}' for node {node_id}: {e}",
                    exc_info=True,
                )
                raise
        except Exception as e:
            logger.error(
                f"Unexpected error setting single vector property '{property_name}' for node {node_id}: {e}",
                exc_info=True,
            )
            raise

    def create_vector_index(
        self,
        index_name: str,
        label: str,
        property_name: str,
        dimensions: int = 768,  # Default dimension, make sure it matches your model
        similarity_function: str = "cosine",
    ):
        """
        Creates a vector index if it doesn't exist for nodes with a given label
        and vector property. Uses safe interpolation for identifiers.
        """
        # Use backticks for safety around identifiers
        query = (
            f"CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS "
            f"FOR (n:`{label}`) "
            f"ON (n.`{property_name}`) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimensions}, `vector.similarity_function`: '{similarity_function}'}}}}"
        )
        try:
            with self.driver.session() as session:
                session.run(query)
                logger.info(
                    f"Ensured vector index `{index_name}` exists for label `{label}`, property `{property_name}` "
                    f"(Dims: {dimensions}, Similarity: {similarity_function})."
                )
        except Neo4jError as e:
            logger.error(
                f"Database error creating vector index `{index_name}`: {e}",
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error creating vector index `{index_name}`: {e}",
                exc_info=True,
            )
            raise
