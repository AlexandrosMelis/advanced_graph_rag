import os
import re

from Bio import Entrez
from tqdm import tqdm
from configs.config import logging, ConfigEnv


class PubMedArticleFetcher:

    def __init__(self, data_path: str):

        Entrez.email = ConfigEnv.ENTREZ_EMAIL
        self._db = "pubmed"
        self._rettype = "medline"
        self._retmode = "text"
        self._pmid_regex_pattern = r"PMID-\s*(\d+)"
        self._RAW_DATA_PATH = data_path

    def fetch_articles(self, pmids: list):

        if not pmids:
            raise ValueError(
                "No PMIDs provided. Please provide at least one PMID to fetch articles."
            )

        pmids_str = ", ".join(pmids)
        logging.info(f"Fetching articles for PMIDs: {pmids_str}")

        try:
            handle = Entrez.efetch(
                db=self._db, id=pmids_str, rettype=self._rettype, retmode=self._retmode
            )
        except Exception as e:
            logging.error(
                f"Failed to fetch articles from Entrez for PMIDs: {pmids_str}. Error: {e}"
            )
            raise

        try:
            concatenated_articles = handle.read()
            handle.close()
        except Exception as e:
            logging.error(
                f"Failed to read Entrez handle for PMIDs: {pmids_str}. Error: {e}"
            )
            raise

        try:
            data_mapping = self._get_article_mappings(
                concatenated_articles=concatenated_articles
            )
            self._save_articles(mapping=data_mapping)
        except Exception as e:
            logging.error(f"Failed to save articles for PMIDs: {pmids_str}. Error: {e}")
            raise

        logging.info("Articles saved successfully.")

    def _get_article_mappings(self, concatenated_articles: str):
        articles = concatenated_articles.strip().split("\n\n")
        logging.debug(f"Number of potential article segments: {len(articles)}")

        mapping = {}
        for article in articles:
            pmid = self._extract_pmid(article)
            if pmid:
                mapping[pmid] = article
            else:
                # Could be malformed text or parser not picking up PMID
                logging.warning(
                    "Article segment with no valid PMID encountered. Skipping."
                )
        return mapping

    def _extract_pmid(self, text):
        match = re.search(self._pmid_regex_pattern, text)
        if match:
            return match.group(1)
        else:
            return None

    def extract_pmids_from_articles(self, articles):
        pmid_list = []
        for article in articles.split("\n\n"):
            pmid = self.extract_pmid_from_article(article)
            if pmid:
                pmid_list.append(pmid)
        return pmid_list

    def _save_article(self, file_path: str, text: str):
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text)
        except OSError as e:
            logging.error(f"Failed to write article to {file_path}. Error: {e}")
            raise

    def _save_articles(self, mapping: dict):
        logging.info(f"Saving {len(mapping)} articles to {self._RAW_DATA_PATH}")
        for pmid, content in tqdm(mapping.items(), total=len(mapping)):
            file_path = os.path.join(self._RAW_DATA_PATH, f"{pmid}.txt")
            self._save_article(file_path=file_path, text=content)
