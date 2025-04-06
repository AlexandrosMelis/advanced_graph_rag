import re
from typing import Any, Literal, Optional, Type, Union

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

from knowledge_graph.connection import Neo4jConnection


class VectorToolInput(BaseModel):
    query: str = Field(description="The query for searching in the Neo4j vector index")


class VectorSimilaritySearchTool(BaseTool):
    """Tool for performing similarity search in vector index to get the relevant chunks"""

    name: str = "VectorSimilaritySearchTool"
    description: str = (
        "useful for when you need to answer questions based on articles content"
    )
    llm: BaseLanguageModel
    embedding_model: Embeddings
    args_schema: Type[BaseModel] = VectorToolInput
    return_direct: bool = True
    neo4j_connection: Neo4jConnection
    str_parser: StrOutputParser = StrOutputParser()

    # static variables
    context_alias: str = "context"
    context_label: str = "CONTEXT"
    vector_index: str = "contextIndex"
    k: int = 10
    threshold: float = 0.70

    def _get_answer_template(self) -> PromptTemplate:
        LONG_ANSWER_PROMPT_TEMPLATE = """Role: You are a research expert in medical literature. 
<task>
Given the following context retrieved from PubMed article abstracts, answer the question.
Find the context placed in <context></context> tags and the question placed in <question></question> tags.  
</task>      

<instructions>
- Your answer should be based ONLY on the information presented in the context.
- Include the reasoning behind your answer and the conclusion.
- If there is no sufficient information in the context to answer the question, respond with "Cannot answer based on the provided information".
</instructions>

<output_format>
- Your output should be a consistent paragraph.
</output_format>

--Real data--
<context>
{context}
</context>

<question>
{question}
</question>""".strip()

        prompt_template = PromptTemplate.from_template(LONG_ANSWER_PROMPT_TEMPLATE)
        return prompt_template

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[str, dict]:

        embedded_query = self.embedding_model.embed_query(query)

        # VECTOR_SEARCH_QUERY = f"""MATCH (article:ARTICLE)-[:HAS_CONTEXT]->({self.context_alias}:{self.context_label})
        # WITH article, {self.context_alias},
        #     vector.similarity.cosine($query, {self.context_alias}.embedding) AS score
        #     ORDER BY score DESCENDING
        #     LIMIT $k WHERE score > $threshold
        # RETURN article.pmid as pmid, {self.context_alias}.text_content as content, score as score""".strip()

        VECTOR_QUERY = """CALL db.index.vector.queryNodes($vector_index, $k, $embedded_query)
YIELD node AS context, score""".strip()

        RETRIEVAL_QUERY = """MATCH (article:ARTICLE)-[:HAS_CONTEXT]->(context)
RETURN article.pmid as pmid, context.text_content as content, score as score""".strip()

        VECTOR_SEARCH_QUERY = f"""{VECTOR_QUERY}\n{RETRIEVAL_QUERY}""".strip()

        retrieved_contexts = self.neo4j_connection.execute_query(
            query=VECTOR_SEARCH_QUERY,
            params={
                "embedded_query": embedded_query,
                "k": self.k,
                "threshold": self.threshold,
                "vector_index": self.vector_index,
            },
        )

        if self.return_direct:
            return retrieved_contexts

        contexts_content = "- " + "\n- ".join(
            [chunk["content"] for chunk in retrieved_contexts]
        )

        prompt = self._get_answer_template()
        answer_chain = prompt | self.llm | self.str_parser

        answer = answer_chain.invoke({"question": query, "context": contexts_content})

        return {"answer": answer, "context": retrieved_contexts}

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[str, dict]:
        """Use the tool asynchronously."""
        return self._run(query, run_manager=run_manager.get_sync())

    def get_model_name(self) -> str:
        return self.llm.model.replace(":", "_").replace("-", "_").replace("/", "_")
