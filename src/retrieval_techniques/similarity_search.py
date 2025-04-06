from langchain_core.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import PromptTemplate

from knowledge_graph.connection import Neo4jConnection


class SimilaritySearchRetriever:
    def __init__(
        self,
        llm: BaseLanguageModel,
        embedding_model: Embeddings,
        neo4j_connection: Neo4jConnection,
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.neo4j_connection = neo4j_connection
        self.str_parser = StrOutputParser()

    def retrieve_chunks(self, **kwargs):
        func_mapper = {
            "relevant_contexts": self.get_relevant_contexts,
            "relevant_meshes": self.get_relevant_mesh_terms,
        }

        technique = kwargs.get("technique", None)
        if technique is None:
            raise ValueError("technique is not defined.")
        if technique not in func_mapper:
            raise ValueError(f"retrieval_technique {technique} is not defined.")
        if "k" not in kwargs:
            raise ValueError("k is not defined.")
        if "query" not in kwargs:
            raise ValueError("query is not defined.")
        return func_mapper[technique](query=kwargs["query"], k=kwargs["k"])

    def similarity_search(
        self, query: str, k: int, vector_index: str, retrieval_snippet: str
    ) -> list:
        """ """
        VECTOR_SNIPPET = """CALL db.index.vector.queryNodes($vector_index, $k, $embedded_query)
        YIELD node AS context, score""".strip()

        VECTOR_SEARCH_QUERY = f"{VECTOR_SNIPPET}\n{retrieval_snippet}".strip()

        # create the query embedding
        embedded_query = self.embedding_model.embed_query(query)

        chunks = self.neo4j_connection.execute_query(
            query=VECTOR_SEARCH_QUERY,
            params={
                "embedded_query": embedded_query,
                "k": k,
                "vector_index": vector_index,
            },
        )

        return chunks

    def get_relevant_contexts(self, query: str, k: int) -> list:
        """
        Get similar contexts from the knowledge graph using similarity search.
        """
        vector_index = "contextIndex"
        retrieval_snippet = """MATCH (article:ARTICLE)-[:HAS_CONTEXT]->(context)
        RETURN article.pmid as pmid, context.text_content as content, score as score""".strip()

        return self.similarity_search(
            query=query,
            k=k,
            vector_index=vector_index,
            retrieval_snippet=retrieval_snippet,
        )

    def get_relevant_mesh_terms(self, query: str, k: int) -> list:
        """
        Get similar mesh terms from the knowledge graph using similarity search.
        """
        vector_index = "meshIndex"
        retrieval_snippet = """MATCH (article:ARTICLE)-[:HAS_MESH]->(mesh)
        RETURN article.pmid as pmid, mesh.name as term, mesh.definition as definition, score as score""".strip()

        return self.similarity_search(
            query=query,
            k=k,
            vector_index=vector_index,
            retrieval_snippet=retrieval_snippet,
        )

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

    def get_answer_based_on_contexts(
        self, query: str, k: int, return_chunks: bool = False
    ) -> dict:
        """
        Get the answer based on the retrieved contexts using the LLM.
        """
        # get the relevant contexts
        relevant_contexts = self.get_relevant_contexts(query=query, k=k)
        contexts_str = "- " + "\n- ".join(
            [chunk["content"] for chunk in relevant_contexts]
        )

        prompt = self._get_answer_template()
        answer_chain = prompt | self.llm | self.str_parser

        answer = answer_chain.invoke({"question": query, "context": contexts_str})

        if return_chunks:
            return {"answer": answer, "context": relevant_contexts}
        else:
            return {"answer": answer}
