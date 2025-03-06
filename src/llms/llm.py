import os

from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI

from configs.config import logger


class ChatModel:
    """
    **1. Groq provider**
    Available models:
        - Production models:
            - llama-3.3-70b-versatile // 1,000 requests/day | 6,000 tokens/minute
            - llama3-70b-8192 // 1,000 requests/day | 6,000 tokens/minute
            - mixtral-8x7b-32768 // 14,400 requests/day | 5,000 tokens/minute
        - Preview models:
            - qwen-2.5-32b // 1,000 requests/day | 6,000 tokens/minute
            - deepseek-r1-distill-qwen-32b // 1,000 requests/day | 6,000 tokens/minute
            - deepseek-r1-distill-llama-70b // 1,000 requests/day | 6,000 tokens/minute
    ------------------------------------------
    **2. Google provider**
    Available models:
        - gemini-2.0-flash // 15 requests/minute
    """

    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        self.llm = None
        self.initialize_model()

    def initialize_model(self):
        if self.provider == "groq":
            if not os.environ.get("GROQ_API_KEY"):
                logger.debug("GROQ_API_KEY is not set")
                raise ValueError("GROQ_API_KEY is not set")
            self.llm = init_chat_model(
                self.model_name, model_provider=self.provider, temperature=0
            )
        elif self.provider == "google":
            if not os.environ.get("GOOGLE_API_KEY"):
                logger.debug("GOOGLE_API_KEY is not set")
                raise ValueError("GOOGLE_API_KEY is not set")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=3,
            )
        else:
            raise ValueError(f"Provider {self.provider} not supported")
