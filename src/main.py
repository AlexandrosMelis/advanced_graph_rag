import json
import os

from tqdm import tqdm

from configs import ConfigEnv, ConfigPath
from configs.config import logger
from data_collection.dataset_constructor import DatasetConstructor
from data_collection.fetcher import MeshTermFetcher, PubMedArticleFetcher
from data_collection.reader import BioASQDataReader
from data_preprocessing.text_splitter import TextSplitter
from knowledge_graph.connection import Neo4jConnection
from knowledge_graph.crud import GraphCrud
from knowledge_graph.loader import GraphLoader
from llms.embedding_model import EmbeddingModel
from utils.utils import read_json_file


def construct_graph_dataset():
    """
    The function aims to construct the dataset for the graph database.
    The following steps are performed:
    1. Read the BIOASQ data from the parquet file.
    2. Fetch the articles from PubMed for the distinct PMIDs mentioned in the BIOASQ data.
    3. Fetch the Mesh Term Definitions for the Mesh Terms mentioned in the PubMed articles.
    4. Combine the BIOASQ, PubMed, and Mesh Term Definitions to create the graph data for loading into Neo4j.
    """
    asq_reader = BioASQDataReader(rows_limit=3)
    article_fetcher = PubMedArticleFetcher()

    # 1. Read the BIOASQ parquet data file
    asq_data_file_path = os.path.join(ConfigPath.RAW_DATA_DIR, "bioasq_train.parquet")
    asq_data = asq_reader.read_parquet_file(file_path=asq_data_file_path)
    pmids_for_fetch = asq_reader.get_distinct_pmids()

    # 2. Fetch articles from PubMed
    pubmed_data = article_fetcher.fetch_articles(pmids=pmids_for_fetch)
    mesh_terms = article_fetcher.get_mesh_terms()

    # 3. Fetch mesh term definitions
    mesh_fetcher = MeshTermFetcher()
    mesh_term_definitions = mesh_fetcher.fetch_definitions(mesh_terms=mesh_terms)
    logger.info(f"Total Mesh Term Definitions: {len(mesh_term_definitions)}")

    # 4. Combine BIOASQ and PubMed to create the graph data for loading into Neo4j
    dataset_constructor = DatasetConstructor(
        bioasq_data=asq_data, pubmed_data=pubmed_data
    )
    dataset_constructor.create_graph_data()


def load_graph_data():
    """
    The function aims to load the graph data into Neo4j.
    """
    # required initializations
    embedding_model = EmbeddingModel()
    neo4j_connection = Neo4jConnection(
        uri=ConfigEnv.NEO4J_URI,
        user=ConfigEnv.NEO4J_USER,
        password=ConfigEnv.NEO4J_PASSWORD,
        database=ConfigEnv.NEO4J_DB,
    )
    graph_crud = GraphCrud(neo4j_connection=neo4j_connection)
    text_splitter = TextSplitter()
    graph_data = read_json_file(
        file_path=os.path.join(ConfigPath.RAW_DATA_DIR, "bioasq_graph_data.json")
    )
    logger.info(f"Total Graph Data: {len(graph_data)}")

    graph_loader = GraphLoader(
        data=graph_data,
        embedding_model=embedding_model,
        text_splitter=text_splitter,
        crud=graph_crud,
    )
    graph_loader.load_mesh_nodes()
    graph_loader.load_qa_articles_contexts()
    graph_loader.load_similarities_to_graph()


if __name__ == "__main__":
    # 1 step: construct the graph dataset
    construct_graph_dataset()
    # 2 step: load the dataset to Neo4j db
    load_graph_data()
