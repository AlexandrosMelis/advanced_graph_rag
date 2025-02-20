import json
import os
import re

from Bio import Entrez
from tqdm import tqdm

from configs.config import ConfigEnv, ConfigPath, logger


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
        logger.info(f"Fetching articles for total PMIDs: {len(pmids)}")

        try:
            handle = Entrez.efetch(
                db=self._db, id=pmids_str, rettype=self._rettype, retmode=self._retmode
            )
        except Exception as e:
            logger.error(
                f"Failed to fetch articles from Entrez for PMIDs: {pmids_str}. Error: {e}"
            )
            raise

        try:
            concatenated_articles = handle.read()
            handle.close()
        except Exception as e:
            logger.error(
                f"Failed to read Entrez handle for PMIDs: {pmids_str}. Error: {e}"
            )
            raise

        try:
            data_mapping = self._get_article_mappings(
                concatenated_articles=concatenated_articles
            )
            self._save_articles(mapping=data_mapping)
        except Exception as e:
            logger.error(f"Failed to save articles for PMIDs: {pmids_str}. Error: {e}")
            raise

        logger.info("Articles saved successfully.")

    def _get_article_mappings(self, concatenated_articles: str):
        articles = concatenated_articles.strip().split("\n\n")
        logger.debug(f"Number of potential article segments: {len(articles)}")

        mapping = {}
        for article in articles:
            pmid = self._extract_pmid(article)
            if pmid:
                mapping[pmid] = article
            else:
                # Could be malformed text or parser not picking up PMID
                logger.warning(
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
            logger.error(f"Failed to write article to {file_path}. Error: {e}")
            raise

    def _save_articles(self, mapping: dict):
        logger.info(f"Saving {len(mapping)} articles to {self._RAW_DATA_PATH}")
        for pmid, content in tqdm(mapping.items(), total=len(mapping)):
            file_path = os.path.join(self._RAW_DATA_PATH, f"{pmid}.txt")
            self._save_article(file_path=file_path, text=content)


class MeshTermFetcher:
    """
    Fetches MeSH term definitions from the MeSH database using the Entrez API.
    """

    def __init__(self) -> None:
        Entrez.email = ConfigEnv.ENTREZ_EMAIL
        self._db = "mesh"
        self._field = "MH"
        self._retmax = 1
        self._stop_terms = [
            "Year introduced:",
            "Subheadings:",
            "Tree Number(s)",
            "Previous Indexing:",
            "See Also:",
            "All MeSH Categories",
        ]
        self._file_name = "definitions_test.json"

    def _read_json_file(self, file_path: str) -> dict:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                return data
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            raise e
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from file: {file_path}")
            raise e

    def _save_json_file(self, file_path: str, data: dict) -> None:
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data, file)
        except Exception as e:
            logger.error(f"Failed to write JSON to {file_path}. Error: {e}")
            raise

    def get_mesh_ui(self, term: str) -> str:
        """
        Fetches the MeSH UI for a given MeSH term.
        """
        handle = Entrez.esearch(
            db=self._db,
            term=f'"{term}"[MeSH Terms]',
            field=self._field,
            retmax=self._retmax,
        )
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"][0] if record["IdList"] else None

    def extract_definition(self, term: str, text: str) -> str:
        text_lower = text.lower()
        term_lower = term.lower()
        start_index = text_lower.find(term_lower)
        if start_index == -1:
            start_index = 0
        # Determine the earliest occurrence of any stop term after the term
        stop_indexes = []
        for stop_term in self._stop_terms:
            idx = text.find(stop_term, start_index)
            if idx != -1:
                stop_indexes.append(idx)
        if stop_indexes:
            end_index = min(stop_indexes)
            definition = text[start_index:end_index].strip()
        else:
            definition = text[start_index:].strip()
        definition = definition.replace("[Subheading]", "")
        return definition

    def get_definition(self, ui: str, term: str):
        handle = Entrez.efetch(db=self._db, id=ui)
        text_data = handle.read()
        handle.close()
        return self.extract_definition(term, text_data)

    def batch_retrieve(self, mesh_terms: list):

        logger.debug(f"Working on file: {self._file_name}")
        file_path = os.path.join(ConfigPath.DATA_DIR, self._file_name)
        if os.path.exists(file_path):
            definitions = self._read_json_file(file_path)
            if not definitions:
                definitions = {}
        else:
            definitions = {}

        for term in tqdm(mesh_terms):
            if term in definitions:
                continue
            try:
                ui = self.get_mesh_ui(term)
                if ui:
                    definition = self.get_definition(ui, term)
                    definitions[term] = definition
                else:
                    definitions[term] = "No MeSH definition found"
            except Exception as e:
                logger.error(
                    f"Failed to retrieve MeSH definition for term: {term}. Error: {e}"
                )
                definitions[term] = "No MeSH definition found"

            self._save_json_file(file_path=file_path, data=definitions)

        return definitions
