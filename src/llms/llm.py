import os

from langchain.chat_models import init_chat_model

from configs.config import logger


class LLM:
    """
    LLM class for handling LLM models.
    Groq is used as the provider.
    The model is initialized from the name of the model.
    Available models:
    - Production models:
        - llama3-70b-8192
        - mixtral-8x7b-32768
    - Preview models:
    - qwen-2.5-32b
    - deepseek-r1-distill-qwen-32b
    - deepseek-r1-distill-llama-70b
    - llama-3.3-70b-specdec
    """

    @classmethod
    def from_name(cls, model_name: str):
        if not os.environ.get("GROQ_API_KEY"):
            logger.debug("GROQ_API_KEY is not set")
            raise ValueError("GROQ_API_KEY is not set")

        cls.model_name = model_name
        cls.provider = "groq"
        cls.model = init_chat_model(model_name, model_provider=cls.provider)
        return cls.model
