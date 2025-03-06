import os

from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI

from configs.config import logger


class LLM:
    """
    LLM class for handling LLM models.
    Groq is used as the provider.
    The model is initialized from the name of the model.
    Available models:
    **Production models**
        - llama-3.3-70b-versatile
        - llama3-70b-8192
        - mixtral-8x7b-32768
    - Preview models:
    - qwen-2.5-32b
    - deepseek-r1-distill-qwen-32b
    - deepseek-r1-distill-llama-70b
    - deepseek-r1-distill-llama-70b
    - llama-3.3-70b-specdec

    Google provider, models:
    - gemini-2.0-flash //  limit: 15 requests/minute
    """

    @classmethod
    def initialize_model(cls, provider: str, model_name: str):
        if provider == "groq":
            if not os.environ.get("GROQ_API_KEY"):
                logger.debug("GROQ_API_KEY is not set")
                raise ValueError("GROQ_API_KEY is not set")
            cls.model = init_chat_model(
                model_name, model_provider=provider, temperature=0
            )
        elif provider == "google":
            if not os.environ.get("GOOGLE_API_KEY"):
                logger.debug("GOOGLE_API_KEY is not set")
                raise ValueError("GOOGLE_API_KEY is not set")
            cls.model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=3,
            )
        else:
            raise ValueError(f"Provider {provider} not supported")
        return cls.model
