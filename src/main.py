import os
from datetime import datetime
from typing import Any

from configs import ConfigEnv, ConfigPath
from configs.config import logger
from data_collection.dataset_constructor import DatasetConstructor
from data_collection.fetcher import MeshTermFetcher, PubMedArticleFetcher
from data_collection.reader import BioASQDataReader
from data_preprocessing.text_splitter import TextSplitter
from evaluation.retrieval_evaluation import run_evaluation_on_retrieved_chunks
from evaluation.retriever_executor import collect_retrieved_chunks
from knowledge_graph.connection import Neo4jConnection
from knowledge_graph.crud import GraphCrud
from knowledge_graph.loader import GraphLoader
from llms.embedding_model import EmbeddingModel
from llms.llm import ChatModel
from retrieval.tools.vector_search_tool import VectorSimilaritySearchTool
from utils.utils import read_json_file


def construct_graph_dataset(asq_reader: BioASQDataReader):
    """
    The function aims to construct the dataset for the graph database.
    The following steps are performed:
    1. Read the BIOASQ data from the parquet file.
    2. Fetch the articles from PubMed for the distinct PMIDs mentioned in the BIOASQ data.
    3. Fetch the Mesh Term Definitions for the Mesh Terms mentioned in the PubMed articles.
    4. Combine the BIOASQ, PubMed, and Mesh Term Definitions to create the graph data for loading into Neo4j.
    """

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


def load_graph_data(embedding_model, graph_crud):
    """
    The function aims to load the graph data into Neo4j.
    1. Initialize the EmbeddingModel, Neo4jConnection, GraphCrud, TextSplitter, and GraphLoader.
    2. Load the Mesh Nodes into the Neo4j graph.
    3. Load the QA Pairs, Articles, and Context Nodes into the Neo4j graph.
    4. Load the Similarity Relationships between Context Nodes into the Neo4j graph.
    """

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
    logger.info("Loading in Neo4j Database completed successfully!")


def evaluate_retriever(source_data: list, retriever: Any):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # create a folder with the datetime name
    output_dir_path = os.path.join(ConfigPath.RESULTS_DIR, timestamp)
    os.makedirs(output_dir_path, exist_ok=True)
    retrieved_chunks = collect_retrieved_chunks(
        source_data=source_data,
        retriever=retriever,
        output_dir=output_dir_path,
    )
    run_evaluation_on_retrieved_chunks(
        benchmark_data=source_data,
        retrieval_results=retrieved_chunks,
        output_dir=output_dir_path,
    )
    print("\n\nEvaluation completed successfully!")


if __name__ == "__main__":
    # required initializations
    samples_limit = 500
    asq_reader = BioASQDataReader(samples_limit=samples_limit)
    asq_data_file_path = os.path.join(ConfigPath.RAW_DATA_DIR, "bioasq_train.parquet")
    data = asq_reader.read_parquet_file(file_path=asq_data_file_path)
    embedding_model = EmbeddingModel()
    llm = ChatModel(
        provider="google", model_name="gemini-2.0-flash-lite"
    ).initialize_model()
    neo4j_connection = Neo4jConnection(
        uri=ConfigEnv.NEO4J_URI,
        user=ConfigEnv.NEO4J_USER,
        password=ConfigEnv.NEO4J_PASSWORD,
        database=ConfigEnv.NEO4J_DB,
    )
    graph_crud = GraphCrud(neo4j_connection=neo4j_connection)

    # 1 step: construct the graph dataset
    # construct_graph_dataset(asq_reader=asq_reader)
    # 2 step: load the dataset to Neo4j db
    # load_graph_data(embedding_model=embedding_model, graph_crud=graph_crud)

    # ******** Evaluation ********
    # Evaluate similarity search retriever
    vector_search_tool = VectorSimilaritySearchTool(
        llm=llm,
        embedding_model=embedding_model,
        neo4j_connection=neo4j_connection,
        return_direct=True,
    )
    evaluate_retriever(source_data=data, retriever=vector_search_tool)
