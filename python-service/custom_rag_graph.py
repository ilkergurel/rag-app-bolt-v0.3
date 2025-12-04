from langchain_ollama import ChatOllama
from copy import deepcopy
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_vertexai import HarmCategory, HarmBlockThreshold, ChatVertexAI

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from typing import List, Optional, TypedDict, Annotated
from langchain_core.runnables import RunnableConfig


# New imports (v1)
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors.chain_filter import LLMChainFilter
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_classic.output_parsers.boolean import BooleanOutputParser
from langchain_classic.callbacks.base import AsyncCallbackHandler

# LangSmith Warning Filter
from langsmith.utils import LangSmithMissingAPIKeyWarning
import warnings
import logging
import asyncio, re
warnings.filterwarnings("ignore", category=LangSmithMissingAPIKeyWarning)

from custom_bm25_retriever import CustomBM25Retriever
from custom_semantic_retriever import CustomSemanticRetriever
from custom_retrieveandreorder import RetrieveAndReorder
from custom_chroma_db import vector_store


logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# --------------------- Main RAG Graph Implementation ------------------------------

class CitationReference(BaseModel):
    """Individual citation with document ID"""
    doc_id: str = Field(description="The document ID from source")

class StructuredResponse(BaseModel):
    """Structured response with citations embedded in text and extracted"""
    response_text: str = Field(
        description="The answer text, naturally written with paragraphs and Markdown format. STRICTLY DO NOT include citation tags, IDs, or brackets (e.g., [doc_1], or (doc_1)) inside this string."
    )
    citations: list[CitationReference] = Field(
        description="The list of citations. This is the ONLY place where document IDs should appear."
    )


def reduce_and_deduplicate(existing_list: List[Document], new_items: List[Document]) -> List[Document]:
    """
    This is a custom reducer function for LangGraph state.
    It combines two lists and removes all duplicates
    while preserving the order of appearance.
    
    This replaces the default 'operator.add' (which is just list + list).
    """
    seen = set()
    result = []
    for doc in existing_list + new_items:
        marker = doc.page_content  
        if marker not in seen:
            seen.add(marker)
            result.append(doc)
    return result

class CustomRagGraph:
    def __init__(self):
        #Limit LLM API calls concurrency to 10 with semaphore
        
        # Initialize LLM
        model_name_api = "gemini-2.5-flash" #"gemini-2.5-flash-lite" #"gemini-2.0-flash"
        project_name = "leafy-audio-472013-n3"
        location="us-central1"
        num_simul_llm_calls = 10

        self.llm_call_semaphore = asyncio.Semaphore(num_simul_llm_calls)

        self.vector_store = vector_store().get_vector_store()

        self.retrieve_node_semaphore = asyncio.Semaphore(num_simul_llm_calls)

        llm_local_api = 1

        if llm_local_api == 0:
            self.llm = ChatOllama(
                model="cogito:14b", 
                base_url="http://host.docker.internal:11434",
                temperature=0.0, 
                repeat_penalty=1.1, 
                disable_streaming=False)
        else:
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            self.llm = ChatVertexAI(
                project = project_name,
                location = location,
                model_name = model_name_api,
                temperature = 0.0,
                safety_settings = safety_settings,  # Pass the corrected dictionary
                request_parallelism = num_simul_llm_calls,
                wait_exponential_kwargs = {
                    "multiplier": 1.0,
                    "min": 4.0,
                    "max": 10,
                    "exp_base": 2.0
                }
            )       

        """
        Create the prompt manually instead of downloading as below:
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        """
        doc_template = """<source id="{id}">{page_content}</source>"""
        document_prompt =PromptTemplate(
                input_variables=["page_content", "id"], 
                template=doc_template
            )
    
        # 3. Define the System Message Template (updated for citation)

        # system_template = """You are a highly capable AI assistant. Your main task is to answer the user's QUESTION based SOLELY on the provided CONTEXT in the same language of the user's input and populate the required output structure correctly.
                
        #     If user's question is a follow-up question, use the provided conversation history to understand the content of the current question as conversation history is relevant to the user's question.
            
        #     Instructions:
        #     1. If the CONTEXT contain some text related to the user's question (not superficially), provide a clear, comprehensive answer ONLY using the information given in that CONTEXT.
        #         In that case, you MUST ensure that EACH sentence in the answer is directly traceable to the provided CONTEXT.
        #         So, at the END of each sentence, STRICTLY add string of IDs given in the CONTEXT with EXACTLY that format of "<some_doc_id> in order to show the traceable reference from CONTEXT".                
        #     2. If CONTEXT is not related to the user's question or only superficially related, STRICTLY, do not generate any answer. Instead, your entire response MUST be only one sentence and it MUST be translated to the language of the user's question: "There is no information in the provided context to answer the question.".
        #     In that case, do not put "<some_doc_id>".               
        #     3. If CONTEXT does not contain any text, STRICTLY, do not generate any answer. Instead, your entire response MUST be only one sentence and it MUST be translated to the language of the user's question: "There is no information in the provided context to answer the question.".
        #     In that case, do not put "<some_doc_id>". 
        #     4. ABSOLUTELY DO NOT invent, guess, or add information that is not explicitly present in the CONTEXT. Any ungrounded information will be treated as a severe error."           
            
        #     Example-1 Output Format:
        #     The psychological concept of slips of the tongue reveals suppressed desires<doc_123_chunk_123>. Freud found that<doc_456_chunk_234>. 

        #     Example-2 Output Format:
        #     There is no information about the question.          

        #     CONTEXT: {context}
        #     """
        
        system_template = """You are a highly capable AI assistant. Your task is to answer the user's QUESTION based SOLELY on the provided CONTEXT in the same language of the user's input and populate the required output structure correctly.

            If user's question is a follow-up question, use the provided conversation history to understand the context.

            Instructions for the Output Structure:

            1. **Answer Generation (Clean Text)**:
            - If the CONTEXT contains information to answer the question, formulate a clear, comprehensive answer in the **same language as the user's input**.
            - Populate the 'response_text' field with this text.
            - Write naturally with proper paragraphs (separate them with blank lines) in Markdown format
            - Use bullet points in new lines when listing multiple items
            - Use numbered lists in new lines for steps or ordered information
            - ABSOLUTELY DO NOT invent information.
            - **CRITICAL RULE**: The 'response_text' must be PURE TEXT. Do NOT put citation tags like <doc_1>, [1], or (Source:...) inside the sentence.         

            2. **Citations (Source Extraction)**:
            - You must identify the specific 'id' of every CONTEXT chunk used to generate the answer.
            - Extract these IDs and populate the 'citations' list with 'doc_id' field in the output structure.
            - Do NOT add the IDs inside the answer text string (unless the structure specifically asks for inline citations). Isolate them in the list.

            3. **Handling "No Information"**:
            - If the CONTEXT is not related or empty, STRICTLY do not generate an answer.
            - In the 'response_text' field, output exactly: "There is no information in the provided context to answer the question." (translated to the user's language).
            - Leave the 'citations' list with 'doc_id' field EMPTY.

            CONTEXT:
            {context}
            """

        system_message_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(input_variables=["context"], template=system_template)
        )

        # 4. Define the Placeholder for Chat History
        chat_history_placeholder = MessagesPlaceholder(variable_name="chat_history", optional=True)

        # 5. Define the Human Message Template
        human_template = "QUESTION: {input}"
        human_message_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(input_variables=["input"], template=human_template)
        )

        # 6. Combine them into a ChatPromptTemplate
        retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages(
            [
                system_message_prompt,
                chat_history_placeholder,
                human_message_prompt,
            ]
        )

        # --- This is the new, manual chain that replaces create_stuff_documents_chain ---

        # 8. Create a helper function to format (stuff) the documents
        def _format_docs_with_citations(docs: List[Document]) -> str:
            """Formats docs with id tags for citation."""
            formatted_docs = []
            for doc in docs:
                # Create the input dictionary for the prompt
                doc_inputs = {
                    "page_content": doc.page_content,
                    "id": doc.metadata["id"]  # This passes 'id' 
                }
                # Format the prompt and convert it to a string
                formatted_docs.append(document_prompt.format_prompt(**doc_inputs).to_string())
            
            # Join all formatted docs with newlines
            return "\n\n".join(formatted_docs)


        # 9. Build the final chain
        # This chain takes an input dictionary (e.g., {"context": [...], "input": "..."})
        # It then "assigns" a new "context" variable by running our formatting function
        # Finally, it pipes all variables to the prompt and the LLM.
        self.combine_docs_chain = (
            RunnablePassthrough.assign(
                context=lambda inputs: _format_docs_with_citations(inputs["context"])
            )
            | retrieval_qa_chat_prompt
            | self.llm.with_structured_output(StructuredResponse)
        )

        #-----------------------------------------------------------------------------------------------
        # Initialize Custom BM25 Retriever instead of default BM25Retriever
        #precomputed_folder = "/__Databases/kitaplar-bm25-db"             # !!!  for deployment in Docker
        precomputed_folder = "D:\\Bilgi\\__Databases\\kitaplar-bm25-db"   # !!!  for local work
        bm25_retriever = CustomBM25Retriever(                            
                            precomputed_folder=precomputed_folder,
                            chroma_collection=self.vector_store._collection, # Pass the Chroma collection object
                            k=10,  
                            phrase_boost=1.5,
                            fetch_k=15 # Number of candidates to check for phrase boosting 
                            )
        
        semantic_retriever = CustomSemanticRetriever(self.vector_store).get_semantic_retriever()

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever], 
            weights=[0.5, 0.5]
        )

        # Create the retrieval and reordering runnable for preventing lost in the middle problem
        retrieve_and_reorder = RetrieveAndReorder(retriever=ensemble_retriever)  

        logger.info("Ensemble retriever and RetrieveAndReorder ready...")  


        # Create the LLM-based filter

        filter_prompt = PromptTemplate(
            template="""
            You are a strict relevance assessor. Your task is to determine if the provided CONTEXT contains a precise and complete content to the main subject or content posed in the QUERY. Any ambiguity means the answer is NO.
            Focus ONLY on the relationship between the QUERY and the CONTEXT.
            
            Return YES ONLY IF the CONTEXT:
                Fully contains the specific QUERY content or completely addresses the core subject stated in the QUERY with specific, verifiable details. 
                The information must be self-contained within the context.
            Return NO if the CONTEXT:
                Provides only content of the QUERY superficially.
                Is merely related to the topic but doesn't contain the specific content.
                Contains specific keywords from the QUERY but lacks substantive information addressing the QUERY.
                Requires inference or information outside the CONTEXT to contain the QUERY.
                Is related in any way less than a direct, complete content.
            
            Output only the single word YES or NO.
            QUERY: {question}
            CONTEXT: {context} 

            Relevant (YES / NO):    
            """,
            input_variables=["question", "context"],
            output_parser=BooleanOutputParser(true_val="YES", false_val="NO")
        )

        # 1. Define the base runnable chain
        
        if filter_prompt.output_parser is not None:
            llm_chain_base = (
                filter_prompt 
                | self.llm 
                | StrOutputParser()  # Add StrOutputParser to convert AIMessage -> str
                | filter_prompt.output_parser
            )
        else:
            raise ValueError("filter_prompt must have an output_parser defined.")

        # 2. Define the config you want to permanently "bind" to this chain
        concurrency_config = RunnableConfig(max_concurrency=num_simul_llm_calls)

        # 3. Create a *new* runnable that has this config built-in
        configured_llm_chain = llm_chain_base.with_config(concurrency_config)

        # 4. Manually create the LLMChainFilter instance 
        llm_filter = LLMChainFilter(
            llm_chain=configured_llm_chain
        )

        # Create the contextual compression retriever
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=llm_filter,
            base_retriever=retrieve_and_reorder   #retrieve_and_reorder   #ensemble_retriever  #bm25_retriever #semantic_retriever
        )

        logger.info("All RAG components ready.")        


    """
    1. Define the Graph State
    """
    class GraphState(TypedDict):
        query: str                                                          # The original user query
        new_queries: List[str]                                              # Generated queries by processing with LLM from user query
        chat_history: List                                                  # Chat history, if any
        documents: Annotated[List[Document], reduce_and_deduplicate]        # The compressed, relevant documents
        answer: str                                                         # The final answer
        citations: list[CitationReference]                                  # The final citations

    class RetrieveState(TypedDict):
        new_sub_query: str                                                  # The original user query
        chat_history: Optional[List]

    """
    2. Define the Graph Nodes
    """

    async def enrich_query(self, state: GraphState):
        """
        Get the user query and recreate/enrich it with LLM.
        """

        # Get chat history
        MAX_HISTORY_TURNS = 8 # Keep the last 8 Human/AI pairs

        chat_history = state.get("chat_history") or []
        if len(chat_history) > (MAX_HISTORY_TURNS * 2):
            # Truncate history to keep only the last (2 * MAX_HISTORY_TURNS) messages
            chat_history = chat_history[-(MAX_HISTORY_TURNS * 2):]

        # for i, item in enumerate(chat_history):
        #     logger.info(f"chat_history[{i}]: {item}")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                "You are a content extracting expert for RAG application. In the same language, extract the content from the user's query according to the following instructions:"
                "Use the provided chat history to resolve ambiguity or references (e.g., 'it', 'he', 'that') in the user's query if there is any, also obtain and use the context if they are related with the user's query,"
                "If the user's query is composed of multiple queries or if the query is complex, decompose the user's query into a list of simpler, self-contained sub-contexts with maximum number of 8,"
                "If the user's query is simple, create a list of closely related, simple, self-contained subqueries with maximum number of 6,"
                "If the user's query contains completely unknown words, keep the query just as it is,"                
                "Extract the contexts incorporating synonyms to improve search recall,"
                "Extract the contexts as clearer, more, specific, less ambiguous, connected with,"
                "Extract exact and specific keywords and phrases from the query, but not add any answer to the query,"
                #"Generate a short, hypothetical answer (1-2 sentences) that a perfect document would contain to answer this query. Add that to the list of extracted contexts. This will be used for semantic search,"
                "Respond with ONLY a JSON list of strings with no explanatory text, markdown, or bullet points before or after."
                "Output Format:" # Add a clear format example
                "[" 
                "\"query 1 with ambiguity resolved\"," 
                "\"specific keyword or fact phrase\""
                "]"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "Query: {query}"),
            ]
        )     

        chain = prompt | self.llm | JsonOutputParser()    

        try:
            temp_new_queries = await chain.ainvoke({
                "query": state["query"],
                "chat_history": chat_history # Pass the history
            })
        except Exception as e:
            logger.error(f"LLM failed during query enrichment. Error: {e}")
            # Fallback to the original query if LLM fails
            temp_new_queries = [state["query"]]

        # Ensure the result is a non-None list, even if empty/invalid.
        if not isinstance(temp_new_queries, list):
            # If the output is None or not a list (e.g., a string, dict, or bool due to parsing failure)
            logger.warning(f"Query enrichment returned non-list data type: {type(temp_new_queries)}. Falling back to original query.")
            temp_new_queries = [state["query"]]            
    
        new_queries = list(dict.fromkeys(temp_new_queries))  # Deduplicate while preserving order
        # logger.info("--- Enrich query ---")    
        # logger.info(f"Enrich query: {state['query']}\n-->{new_queries}")

        # Create new queries with maximum number of 6
        new_queries = new_queries[:6] if len(new_queries) > 6 else new_queries

        logger.info(f" Transformed Queries:")
        for i, new_query in enumerate(new_queries):
            logger.info(f"  {i+1}. {new_query}")   
        logger.info(f"len(chat_history): {len(chat_history)}")

        return {"new_queries": new_queries}
  

    def map_queries(self, state: GraphState) -> List:
        """
        Map each generated query to a separate retrieve node.
        """

        logger.info(f"Document retrieval from DB started with {len(state['new_queries'])} processes in parallel...")
        return [
            Send(
                "retrieve",
                {
                    "new_sub_query": new_query,
                    "chat_history": state.get("chat_history", [])
                }
            )
            for new_query in state["new_queries"]  
        ]

    async def retrieve(self, state: RetrieveState):
        """
        Node to retrieve, reorder, and compress documents.
        """
        async with self.retrieve_node_semaphore:
            logger.info(f"Retrieval process started for a query...")
                        
            query = state["new_sub_query"]
            chat_history = state.get("chat_history", [])
            
            retrieved_docs = await self.compression_retriever.ainvoke(query)  # chat_history can be passed if needed

            logger.info(f"# of retrieved documents per query (after de-duplication): {len(retrieved_docs )}")
            logger.info(f"Corresponding query: {query}\n")

            return {"documents": retrieved_docs}


    async def generate(self, state: GraphState):
        """
        Streams text token-by-token and parses citations from the final string.
        """
        
        logger.info("Generating answer (text streaming mode)...")
        original_query = state["query"]
        query = " ".join(state["new_queries"])
        chat_history = state.get("chat_history", [])
        documents = state["documents"]

        logger.info(f"Reference documents: len(document): {len(documents)}")
        

        logger.info(f"Total # of retrieved documents (after de-duplication): {len(documents)}\n")
        # for i, document in enumerate(documents):
        #     logger.info(f"Document-{i+1}:\n{document}\n")

        answer_text = ""
        citations = []

        if len(state["documents"]) > 0:
            async for chunk in self.combine_docs_chain.astream({
                "input": query,
                "chat_history": chat_history,
                "context": documents
            }):
        # ✅ CORRECT: Check if chunk is the final StructuredResponse
                if isinstance(chunk, StructuredResponse):
                    # This is the final structured response
                    raw_text = chunk.response_text
                    
                    # 1. Define patterns to strip (Customize these based on your ID format)
                    # Examples: <source id="doc_1">...</source>, <doc_1>, [doc_1], (doc_1)
                    patterns = [
                        r'<source[^>]*>.*?</source>',  # Remove XML-like source tags
                        r'<[^>]+>',                    # Remove any other angle brackets
                        r'\[doc_[^\]]+\]',             # Remove [doc_...]
                        r'\(doc_[^\)]+\)'              # Remove (doc_...)
                    ]
                    
                    clean_text = raw_text
                    for pattern in patterns:
                        clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE | re.DOTALL)
                    
                    # Remove double spaces created by deletions
                    clean_text = re.sub(r'[ \t]+', ' ', clean_text).strip()

                    # 2. Assign the cleaned text back
                    answer_text = clean_text
                    citations = chunk.citations
                    logger.info(f"Final citations: {len(citations)}")
                else:
                    # These are token chunks during streaming
                    token = str(chunk)
                    if token:
                        answer_text += token
                        # Yield intermediate updates for real-time display
                        yield {"answer": answer_text}
                        await asyncio.sleep(0.01)  # Reduced sleep for smoother streaming

            # ✅ CORRECT: Yield final result with both text and structured citations
            yield {
                "answer": answer_text,
                "documents": state["documents"],
                "citations": citations
            }
        else:
            """
            There is no  information  in the database, because there are no retrieved documents. So, tell user there is no info.
            """
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system",
                    "You are an expert in extracting user's language. Answer the user **STRICTLY** with the translation of 'There is no information about the question' **in the exact same language of the user's question**. Do not add any extra text or explanation.",
                    ),
                    ("human", "Question: {input}"),
                ]
            )     

            chain = prompt | self.llm | StrOutputParser()  

            async for chunk in chain.astream({
                "input": original_query
            }):
                # Yield each token chunk as a partial update to the 'answer'
                token = str(chunk)
                answer_text += token

                # You can yield intermediate updates if you want
                yield {"answer": answer_text}
                await asyncio.sleep(0.2)

            yield {"answer": answer_text, "documents": state["documents"], "citations": citations}


    async def save_history(self, state: GraphState):
        """
        Updates the chat_history list with the current query and answer.
        """
        query = state["query"]
        answer = state["answer"]
        chat_history = state.get("chat_history", []) or [] # Ensure it's a list

        # 1. Clean the text using Regex
        # We assume 'answer' contains the full string with tags like <doc_123_chunk_456>
        # This regex removes the specific pattern <doc_{digits}_chunk_{digits}>
        answer_text = answer.strip() if answer else ""

        # 2. Add the user query and the clean LLM answer
        new_history = deepcopy(chat_history)
        new_history.append(HumanMessage(content=query))
        new_history.append(AIMessage(content=answer_text))

        return {"chat_history": new_history}

    
    def get_rag_graph(self):
        """
        3. Construct the Graph
        """

        main_workflow = StateGraph(self.GraphState)

        # Add the nodes
        main_workflow.add_node("enrich_query", self.enrich_query)
        main_workflow.add_node("retrieve", self.retrieve)
        main_workflow.add_node("generate", self.generate)
        main_workflow.add_node("save_history", self.save_history)

        # Define the edges
        main_workflow.set_entry_point("enrich_query")
        main_workflow.add_conditional_edges("enrich_query", self.map_queries, {"retrieve":"retrieve"})
        main_workflow.add_edge("retrieve", "generate")
        main_workflow.add_edge("generate", "save_history") 
        main_workflow.add_edge("save_history", END) 

        # Compile the graph
        memory = MemorySaver()
        main_app = main_workflow.compile(checkpointer=memory)

        logger.info("✅ Main RAG graph compiled. Ready to process.")    

        return main_app