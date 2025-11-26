from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.runnables.config import run_in_executor, RunnableConfig
from langchain_community.document_transformers import LongContextReorder

from typing import List

# Pydantic
from pydantic import Field

# Function to process retrieved documents
class RetrieveAndReorder(BaseRetriever):
    """A custom retriever that fetches documents and then reorders them."""
    retriever: BaseRetriever
    reordering: LongContextReorder = Field(default_factory=LongContextReorder)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        
        # The run_manager is guaranteed to be present, so we can
        # get its child and pass it in the config.
        child_run_manager = run_manager.get_child()
        config: RunnableConfig = {"callbacks": child_run_manager}

        # 1. Retrieve documents using the underlying retriever's .invoke()
        retrieved_docs = self.retriever.invoke(query, config=config)
        if not retrieved_docs:
            return []
        
        # 2. Reorder the documents
        reordered_docs_sequence = self.reordering.transform_documents(retrieved_docs)
        
        # 3. Return only the list of documents, as expected by the chain
        return list(reordered_docs_sequence)
    
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Async version:
        1. Awaits the async retrieval from the base retriever using its aget_relevant_documents method.
        2. Runs the synchronous reordering in an executor thread.
        """
        # Prepare the run_manager argument safely (call get_child if available)
        child_run_manager = run_manager.get_child()
        config: RunnableConfig = {"callbacks": child_run_manager}
        
        # 1. Retrieve documents asynchronously
        retrieved_docs = await self.retriever.ainvoke(query, config=config)
        if not retrieved_docs:
            return []


        # 2. Reorder the documents
        # The 'transform_documents' method is synchronous,
        # so we run it in an executor to avoid blocking the event loop.

        # !!! Print the list
        #logger.info(f"Total number of documents come to RetrieveAndReorder: {len(retrieved_docs)}")
        #logger.info("Total list of references:")
        #[logger.info(doc.metadata["source"] ) for doc in retrieved_docs]

        reordered_docs_sequence = await run_in_executor(
            None,  # Use default ThreadPoolExecutor
            self.reordering.transform_documents,
            retrieved_docs
        )

        # !!! Print the list
        #logger.info(f"Total number of documents after RetrieveAndReorder: {len(reordered_docs_sequence)}")
        #logger.info("Total list of references:")
        #[logger.info(doc.metadata["source"] ) for doc in reordered_docs_sequence]        
        
        # 3. Return the list
        return list(reordered_docs_sequence)