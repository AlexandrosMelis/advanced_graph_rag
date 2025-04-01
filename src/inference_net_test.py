import os

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv(), override=True)
print(f"inference key: {os.environ.get('INFERENCE_API_KEY')}")

client = OpenAI(
    base_url="https://api.inference.net/v1",
    api_key=os.environ.get("INFERENCE_API_KEY"),
)


response = client.chat.completions.create(
    model="mistralai/mistral-nemo-12b-instruct/fp-8",
    messages=[
        {
            "role": "user",
            "content": "What do you know about Graph Retrieval Augmented Generation (GraphRAG)?",
        }
    ],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
