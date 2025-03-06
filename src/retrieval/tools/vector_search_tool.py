from typing import Any, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from configs.config import logger
from knowledge_graph.connection import Neo4jConnection


class VectorToolInput(BaseModel):
    query: str = Field(description="The query for searching in the Neo4j vector index")


class VectorSearchTool(BaseTool):
    """Tool for searching in vector index"""

    name: str = "VectorSearchTool"
    description: str = (
        "useful for when you need to answer questions based on articles content"
    )
    llm: Any
    embedding_model: Any
    args_schema: Type[BaseModel] = VectorToolInput
    return_direct: bool = True
    neo4j_connection: Neo4jConnection

    # static variables
    context_alias: str = "context"
    context_label: str = "CONTEXT"
    k: int = 10
    threshold: float = 0.6

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[str, dict]:

        embedded_query = self.embedding_model.embed_query(query)

        VECTOR_SEARCH_QUERY = f"""MATCH ({self.context_alias}:{self.context_label})
        WITH {self.context_alias},
            vector.similarity.cosine($query, {self.context_alias}.embedding) AS score
            ORDER BY score DESCENDING
            LIMIT $k WHERE score > $threshold
        RETURN {self.context_alias}.text_content as content, score as score""".strip()

        results = self.neo4j_connection.execute_query(
            query=VECTOR_SEARCH_QUERY,
            params={
                "query": embedded_query,
                "k": self.k,
                "threshold": self.threshold,
            },
        )

        # logger.info(f"Results: {results}")

        if self.return_direct:
            return results

        # answer with llm
        prompt_template = """You are an expert at analyzing medical literature. 
**Task:**
Your task is to answer a yes/no question based solely on the provided context from PubMed articles.

**Instructions:**
1.  **Read the Question:** Carefully examine the question provided within the `<question></question>` tags.
2.  **Analyze the Context:**  Thoroughly review the information within the `<context></context>` tags. This context is derived from PubMed articles.
3.  **Strictly Use the Context:** Your answer MUST be based exclusively on the information presented in the context. Do not use any external knowledge.
4.  **Infer if Possible:** If the answer is not explicitly stated but can be logically inferred from the context, answer accordingly.
5. **Answer Format:** Answer with exactly one word: "yes" or "no".
6.  **Insufficient or Irrelevant Context:** If the provided context is:
    *   Irrelevant to the question, or
    *   Insufficient to determine an answer,
    you MUST respond with "no".
7.  **Sufficient Context:** If the context contains information that directly answers the question or that the answer can be inferred from, you MUST respond with "yes".

**Output format:**
Respond only with the word "yes" or "no".

**Question:**
<question>
{question}
</question>

**Context:**
<context>
{context}
</context>""".strip()
        prompt = PromptTemplate.from_template(prompt_template)
        answer_chain = prompt | self.llm | StrOutputParser()
        answer = answer_chain.invoke({"question": query, "context": results})
        return answer

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[str, dict]:
        """Use the tool asynchronously."""
        return self._run(query, run_manager=run_manager.get_sync())
