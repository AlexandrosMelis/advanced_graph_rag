from langchain_core.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import PromptTemplate

from knowledge_graph.connection import Neo4jConnection


class SimilaritySearchRetriever:

    # available indexes
    context_vector_index = "contextIndex"
    mesh_vector_index = "meshIndex"

    # cypher snippets
    SIMILARITY_SEARCH_CYPHER_SNIPPET = """CALL db.index.vector.queryNodes($vector_index, $k, $embedded_query) YIELD node AS context, score""".strip()
    CONTEXT_RETRIEVAL_CYPHER_SNIPPET = """MATCH (article:ARTICLE)-[:HAS_CONTEXT]->(context) RETURN elementId(context) as element_id, article.pmid as pmid, context.text_content as content, score as score""".strip()
    MESH_RETRIEVAL_CYPHER_SNIPPET = """MATCH (article:ARTICLE)-[:HAS_MESH]->(mesh) RETURN article.pmid as pmid, mesh.name as term, mesh.definition as definition, score as score""".strip()
    SIMILAR_CONTEXT_RETRIEVAL_CYPHER_SNIPPET = """MATCH (article:ARTICLE)-[:HAS_CONTEXT]->(context:CONTEXT) WHERE elementId(context) in $relevant_element_ids
        // For each context, find its top 5 most similar contexts
        WITH context, article.pmid AS original_pmid, context.text_content AS original_content
        MATCH (context)-[sim:IS_SIMILAR_TO]-(similar_context:CONTEXT)<-[:HAS_CONTEXT]-(similar_article:ARTICLE)
        // Group by original context and order similar contexts by similarity score
        WITH context, original_pmid, original_content, 
            elementId(context) AS original_context_element_id,
            similar_context.text_content AS similar_content, similar_article.pmid AS similar_pmid,
            sim.score AS similarity_score, elementId(similar_context) AS similar_element_id
        ORDER BY similarity_score DESC
        // Collect top n similar contexts for each original context
        WITH original_context_element_id, original_pmid,
            collect({
                element_id: similar_element_id,
                pmid: similar_pmid, 
                content: similar_content,
                score: similarity_score
            })[0..$n] AS top_similar_contexts
        // Return the results
        RETURN original_context_element_id AS element_id, original_pmid AS pmid, top_similar_contexts""".strip()

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

        # retrieval techniques mapper
        self.retrieval_techniques_mapper = {
            "relevant_contexts": self.get_relevant_contexts,
            "relevant_meshes": self.get_relevant_mesh_terms,
            "1_hop_similar_contexts": self.get_1_hop_similar_contexts,
        }

        # answer techniques mapper
        self.answer_techniques_mapper = {
            "relevant_contexts": self.get_answer_based_on_contexts,
        }

    def retrieve_chunks(self, retrieval_type: str, **kwargs):
        """
        Factory method to retrieve chunks based on the technique specified.
        Entry point for the chunk retrieval process.
        """
        if retrieval_type not in self.retrieval_techniques_mapper:
            raise ValueError(
                f"Retrieval technique '{retrieval_type}' is not supported. Valid options are: {list(self.retrieval_techniques_mapper.keys())}."
            )
        return self.retrieval_techniques_mapper[retrieval_type](**kwargs)

    def answer(self, **kwargs):
        """
        Factory method to get the answer based on the technique specified.
        Entry point for the answer retrieval process.
        """
        technique = kwargs.get("technique", None)
        if technique is None:
            raise ValueError("Technique is not defined. `technique` must be provided.")
        if technique not in self.answer_techniques_mapper:
            raise ValueError(
                f"Answer technique '{technique}' is not supported. Valid options are: {list(self.answer_techniques_mapper.keys())}."
            )
        if "k" not in kwargs:
            raise ValueError("k neighbors is not defined. `k` must be provided.")
        if "query" not in kwargs:
            raise ValueError("Query is not defined. `query` must be provided.")
        return self.answer_techniques_mapper[technique](
            query=kwargs["query"], k=kwargs["k"]
        )

    def similarity_search(
        self, query: str, k: int, vector_index: str, retrieval_snippet: str
    ) -> list:
        """
        Perform similarity search in the knowledge graph using the provided query and vector index.
        """
        VECTOR_SEARCH_QUERY = (
            f"{self.SIMILARITY_SEARCH_CYPHER_SNIPPET}\n{retrieval_snippet}".strip()
        )

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
        return self.similarity_search(
            query=query,
            k=k,
            vector_index=self.context_vector_index,
            retrieval_snippet=self.CONTEXT_RETRIEVAL_CYPHER_SNIPPET,
        )

    def get_1_hop_similar_contexts(
        self, query: str, k: int, n_similar_contexts: int
    ) -> list:
        """ """
        # retrieve top k relevant contexts
        relevant_contexts = self.get_relevant_contexts(query=query, k=k)
        # get the context element ids
        relevant_element_ids = [context["element_id"] for context in relevant_contexts]

        embedded_query = self.embedding_model.embed_query(query)

        # get top n similar neighbors for each context
        retrieved_similar_contexts = self.neo4j_connection.execute_query(
            query=self.SIMILAR_CONTEXT_RETRIEVAL_CYPHER_SNIPPET,
            params={
                "embedded_query": embedded_query,
                "n": n_similar_contexts,
                "relevant_element_ids": relevant_element_ids,
            },
        )

        # format results to appropriate format to concatenate two result lists
        formatted_similar_contexts = list(
            {
                similar_context["element_id"]: {
                    "element_id": similar_context["element_id"],
                    "pmid": similar_context["pmid"],
                    "content": similar_context["content"],
                    "score": similar_context["score"],
                }
                for retrieved_similar_context in retrieved_similar_contexts
                for similar_context in retrieved_similar_context["top_similar_contexts"]
            }.values()
        )

        # get distinct results based on `element_id`
        seen = set()
        results = [
            item
            for item in relevant_contexts + formatted_similar_contexts
            if item.get("element_id") not in seen
            and not seen.add(item.get("element_id"))
        ]

        return results

    def get_relevant_mesh_terms(self, query: str, k: int) -> list:
        """
        Get similar mesh terms from the knowledge graph using similarity search.
        """
        return self.similarity_search(
            query=query,
            k=k,
            vector_index=self.context_vector_index,
            retrieval_snippet=self.MESH_RETRIEVAL_CYPHER_SNIPPET,
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

    def get_answer_based_on_contexts(self, query: str, k: int) -> dict:
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
        return {"answer": answer, "context": relevant_contexts}
