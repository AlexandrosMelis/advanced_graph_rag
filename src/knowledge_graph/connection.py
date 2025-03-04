from neo4j import GraphDatabase

from configs.config import logger


class Neo4jConnection:
    def __init__(self, uri: str, user: str, password: str, database: str):
        self.driver = GraphDatabase.driver(
            uri, auth=(user, password), database=database
        )
        try:
            self.driver.verify_connectivity()
            logger.debug("Connection successful!")
        except Exception as e:
            logger.debug(f"Failed to connect to Neo4j: {e}")

    def get_driver(self) -> GraphDatabase:
        return self.driver
