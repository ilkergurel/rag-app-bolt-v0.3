from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class vector_store():
    def __init__(self):
        #embeddings = OllamaEmbeddings(model="bge-m3", base_url="http://host.docker.internal:11434")  # !!! for deployment in Docker
        embeddings = OllamaEmbeddings(model="bge-m3", base_url="http://localhost:11434")   # !!! For local work

        self.vector_store = Chroma(
            collection_name = "my-doc-assistant-db",
            embedding_function=embeddings,
            persist_directory="D:\\Bilgi\\__Databases\\kitaplar-vectordb-embedding-bgem3"  # !!! For local work
            #persist_directory="/__Databases/kitaplar-vectordb-embedding-bgem3"            # !!! for deployment in Docker
            )
        
    def get_vector_store(self):
        """ Function to initialize and return the Chroma Vector Store """
        return self.vector_store