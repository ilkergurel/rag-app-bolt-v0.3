class CustomSemanticRetriever():
    def __init__(self, vector_store):
        # lambda=1.0 more relevant, lambda=0.0 more diverse
        # search_as = { "k": 10, "lambda_mult": 0.8,  "score_threshold": 0.1, "fetch_k": 20}   
        # self.semantic_retriever = vector_store.as_retriever(search_type="mmr",search_kwargs=search_as)

        search_as = { "k": 10, "score_threshold": 0.2}   
        self.semantic_retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs=search_as)

        print("Semantic retriever from ChromaDB ready...")        



    def get_semantic_retriever(self):
        """ Function to return the semantic retriever """
        return self.semantic_retriever