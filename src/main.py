import os

from configs import ConfigPath
from data_collection.fetcher import PubMedArticleFetcher
from data_collection.reader import MetadataReader
from knowledge_graph.loader import GraphLoader


def download_articles():
    external_metadata_file_name = "pqa_labeled.json"

    # read pmids
    external_metadata_file_path = os.path.join(
        ConfigPath.EXTERNAL_DATA_DIR, external_metadata_file_name
    )
    metadata_reader = MetadataReader(file_path=external_metadata_file_path)

    # download articles
    fetcher = PubMedArticleFetcher(data_path=ConfigPath.RAW_DATA_DIR)
    fetcher.fetch_articles(pmids=metadata_reader.pmids)


def initialized_graph_loader():
    file_path = os.path.join(ConfigPath.KG_CONFIG_DIR, "schema_config.json")
    graph_loader = GraphLoader.from_json_file(path=file_path)
    print("graph loader initiated successfully!")


if __name__ == "__main__":
    initialized_graph_loader()
