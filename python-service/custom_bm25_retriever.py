from rank_bm25 import BM25Okapi

# NLTK Imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Pydantic
from pydantic import Field, PrivateAttr

# LangChain Core
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.runnables.config import run_in_executor

from typing import List, Optional, Any
import logging

import msgspec
import re
import pickle
import os

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# --- NLTK Setup ---
nltk_data_path = os.getenv("NLTK_DATA")
if nltk_data_path:
    nltk.data.path.append(nltk_data_path)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stop_words_en = set(stopwords.words('english'))
stop_words_tr = set(stopwords.words('turkish'))

#--------------------------------------------------------------------------------------------------
#Custom BM25Retriever in order to utilize if all keywords are inside a text chunk, give higher score to that chunk in ranking with scores

class Data(msgspec.Struct):
    documents: list[str]
    metadata: list[dict]

class CustomBM25Retriever(BaseRetriever):
    """
    Loads a precomputed BM25 index and fetches documents from ChromaDB.
    """
    k: int = Field(default=50)             # Number of final documents to retrieve
    phrase_boost: float = Field(default=1.5)
    fetch_k: int = Field(default=50)       # How many candidates to fetch for re-ranking
    precomputed_folder: str
    chroma_collection: Any # Chroma collection object (passed during init)

    # Use PrivateAttr for internal state loaded from files
    _bm25_index: Optional[BM25Okapi] = PrivateAttr(default=None)
    _doc_ids: List[str] = PrivateAttr(default_factory=list)

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Load precomputed data during initialization
        self._load_precomputed_data()
        logger.info(f"CustomBM25Retriever initialized...")

    def _load_precomputed_data(self):
        """Loads the pickled BM25 index and document IDs."""
        logger.info(f"Loading precomputed BM25 data from {self.precomputed_folder}...")
        try:
            # Load the BM25 model object
            with open(os.path.join(self.precomputed_folder, 'bm25_model.pkl'), 'rb') as f:
                self._bm25_index = pickle.load(f)

            # Load the list of document IDs
            with open(os.path.join(self.precomputed_folder, 'doc_ids.pkl'), 'rb') as f:
                self._doc_ids = pickle.load(f)

            if not self._bm25_index or not self._doc_ids:
                raise FileNotFoundError("BM25 model or doc IDs file not found or empty.")

            logger.info(f"Loaded BM25 index for {len(self._doc_ids)} documents.")
        except Exception as e:
            logger.info(f"Error loading precomputed BM25 data: {e}")
            raise

    def _get_relevant_documents(self, query: str, *, run_manager:CallbackManagerForRetrieverRun) -> List[Document]:
        """ Retrieve and rank documents using BM25 with metadata and phrase boosting """
        
        if not self._bm25_index or not self._doc_ids:
            raise ValueError("BM25 index not loaded.")
        if not self.chroma_collection:
            raise ValueError("ChromaDB collection not provided.")        
        
        # 1. Tokenize the query (match precomputation tokenization)
        query = query.lower()
        query = re.sub(r'[^\w\s.,%@()*-+/!&_?#|]', '', query)  # Simplified cleaning, adjust if needed
        tokens = word_tokenize(query)
        filtered_query_tokens = [word for word in tokens if word not in stop_words_en]        
        
        # 2. Get initial scores from the loaded BM25 index
        # This calculates scores for ALL documents but is relatively fast
        all_scores = self._bm25_index.get_scores(filtered_query_tokens)    
        
        # 3. Get top N candidate indices based on raw scores
        # We fetch more candidates than needed (fetch_k) for potential re-ranking
        num_candidates = min(self.fetch_k, len(self._doc_ids))
        candidate_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)[:num_candidates]

        # 4. Map indices to actual Document IDs
        candidate_ids = [self._doc_ids[i] for i in candidate_indices]
        candidate_initial_scores = {self._doc_ids[i]: all_scores[i] for i in candidate_indices}

        if not candidate_ids:
            return []

        # 5. Fetch candidate documents' content from ChromaDB
        try:
            # Fetch only documents and metadata needed for phrase boost and final return
            fetched_results = self.chroma_collection.get(
                ids=candidate_ids,
                include=["metadatas", "documents"] # Fetch content for phrase boost
            )
            # Create Document objects from fetched data
            candidate_docs_map = {
                fetched_results['ids'][i]: Document(
                    page_content=fetched_results['documents'][i],
                    metadata=fetched_results['metadatas'][i]
                ) for i in range(len(fetched_results['ids']))
            }
        except Exception as e:
             logger.info(f"Error fetching documents from ChromaDB: {e}")
             # Fallback: return top k based on initial scores without boosting?
             # Or return empty list / raise error
             top_k_ids = candidate_ids[:self.k]
             # If you MUST return something, fetch just the top K without boosting:
             try:
                 top_k_results = self.chroma_collection.get(ids=top_k_ids, include=["metadatas", "documents"])
                 return [Document(page_content=top_k_results['documents'][i], metadata=top_k_results['metadatas'][i]) for i in range(len(top_k_results['ids']))]
             except:
                 return [] # Final fallback

        # 6. Apply Phrase Boosting and Re-rank the fetched candidates
        boosted_scores = []
        valid_docs_for_ranking = []
        query_lower = query.lower() # Pre-lower query for boosting check

        for doc_id in candidate_ids: # Iterate in original candidate order
            doc = candidate_docs_map.get(doc_id)
            if doc:
                initial_score = candidate_initial_scores.get(doc_id, 0)
                # Reconstruct text for phrase check (or use fetched content directly)
                metadata_str = " ".join(f"{key}: {value}" for key, value in doc.metadata.items())
                full_text = f"{doc.page_content.lower()} {metadata_str}" # Match tokenization logic

                phrase_bonus = self.phrase_boost if query_lower in full_text else 1.0
                boosted_scores.append(initial_score * phrase_bonus)
                valid_docs_for_ranking.append(doc) # Keep track of the Document object

        # 7. Sort the valid candidates by their final boosted score
        ranked_candidates = sorted(zip(valid_docs_for_ranking, boosted_scores), key=lambda x: x[1], reverse=True)

        # !!! Print the list
        # logger.info(f"Total number of documents come from BM25: {len(ranked_candidates)}, selected top-k: {self.k}")
        # logger.info("BM25 top-k list of references:")
        # [logger.info(doc.metadata["source"], " - ", f"BM25-score: {score}" ) for doc, score in (ranked_candidates[:self.k])]

        # 8. Return the top K documents
        return [doc for doc, score in ranked_candidates[:self.k]]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Async version of retrieval: you can run blocking code in a thread
        if necessary, or convert parts to truly async (e.g., async DB calls)
        """
        docs =  await run_in_executor(
            None,
            self._get_relevant_documents,
            query,
            run_manager=run_manager.get_sync(),
        )
        return docs
  

