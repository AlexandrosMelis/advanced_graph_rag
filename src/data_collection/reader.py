import json

from configs.config import logger


class MetadataReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data: dict = {}
        self.pmids: list = []
        self._read_file()

    def _read_file(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                self.data = json.load(file)
                if not self.data:
                    raise ValueError("Metadata json file cannot be empty!")
                self.pmids = list(self.data.keys())
                if not self.pmids:
                    raise ValueError("PMIDs not found!")
                logger.debug("External metadata file loaded successfully!")
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from file: {self.file_path}")
        except ValueError as ve:
            logger.error(f"Value error: {ve}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

    def get_pmids(self) -> list:
        return self.pmids
