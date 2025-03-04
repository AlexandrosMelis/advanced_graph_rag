import os

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
