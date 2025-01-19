# config.py

import os
import logging
from pathlib import Path
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)


class ConfigEnv:
    """
    Central configuration class for environment variables.
    The variables are loaded at import time using `python-dotenv`
    (plus any already-existing OS environment variables).
    """

    # Load variables from .env file (if present) and system environment
    load_dotenv(override=True)

    # Environment variables
    # Provide defaults or raise an error if a critical env variable is missing.
    ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")

    # Immediately validate after loading
    _REQUIRED_VARS = [
        "ENTREZ_EMAIL",  # Example: required for Entrez (NCBI) API
    ]

    @classmethod
    def _validate_required_vars(cls) -> None:
        """
        Ensures that required environment variables are present.
        Logs or raises an error if any are missing.
        """
        for var_name in cls._REQUIRED_VARS:
            if getattr(cls, var_name) is None:
                message = f"Missing required environment variable: {var_name}"
                logging.error(message)
                raise EnvironmentError(message)


class ConfigPath:
    """
    Central configuration class for all project directory paths.
    Responsible for creating each directory if it does not already exist.
    """

    # Base directory is the location of this config file.
    # You may adjust the parent reference as needed.
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # Data directories
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"

    @classmethod
    def create_directories(cls):
        """Create each directory if it doesn't already exist."""
        dirs_to_create = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.INTERMEDIATE_DATA_DIR,
            cls.EXTERNAL_DATA_DIR,
        ]

        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)


# Create all directories when this file is imported
ConfigPath.create_directories()
ConfigEnv._validate_required_vars()
