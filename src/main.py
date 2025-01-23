import os

from configs.config import ConfigPath
from data_collection.article_fetcher import PubMedArticleFetcher
from data_collection.reader import MetadataReader


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


if __name__ == "__main__":
    download_articles()
