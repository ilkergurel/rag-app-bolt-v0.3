"""
Documentation sources:

https://api.python.langchain.com/en/v0.1/langchain_api_reference.html

https://python.langchain.com/docs/introduction/

https://langchain-ai.github.io/langgraph/

A New Data Source:

https://pymupdf.readthedocs.io/en/latest/

In order to download the web contents with all links, use the following command on cmd line for each source:
wget -r -A.html -P langgraph-docs https://langchain-ai.github.io/langgraph/

"""

"""
Reading and splitting documents
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from collections import defaultdict
import chromadb
from chromadb.config import Settings
from chromadb import HttpClient
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
from rank_bm25 import BM25Okapi

import os
from markdown import markdown
from bs4 import BeautifulSoup, Comment
import fitz  # pip install pymupdf
import bisect
import ebooklib
from ebooklib import epub
import warnings 
import logging
import logging.handlers
import time
import requests
import json
import msgspec

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


warnings.filterwarnings('ignore') 

from multiprocessing import Process, Value, Lock,  Semaphore, Queue, JoinableQueue
from threading import Thread
from dotenv import load_dotenv

load_dotenv()

stop_words_en = set(stopwords.words('english'))


# ---------------------------------------------------------------------------------
def collect_files(directory, extensions):
    """
    First get the file names and labels from file, then traverse the given directory and collect all file paths and put labels.
    """

    # Traverse the directory and collect file paths    
    file_counter=0  #it is required because some files may have the same name
    file_dict = defaultdict(list)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):                       
                file_path = os.path.join(root, file)                                 #Obtain filepath and remove path length control with \\?\\
                file_name, file_ext = os.path.splitext(file)                         #Obtain filename without extension                                   
                parent_path = os.path.basename(os.path.dirname(file_path))           #Obtain parent directory
                file_size = os.path.getsize(file_path)                               #Obtain file size
                file_dict[file_path].append([file_name, parent_path, file_size, file_ext])     
                
                file_counter += 1                
    
    logger.info(f"Number of files under all folders (files may have same name): {file_counter}")

    return file_dict, file_counter

# ---------------------------------------------------------------------------------
class Data(msgspec.Struct):
    documents: list[str]
    metadata: list[dict]

def tokenize_for_bm25(doc_content: str, metadata: dict):
    """ Tokenizes content and metadata. Matches retriever's logic. """
    metadata_str = " ".join(f"{key}: {value}" for key, value in metadata.items())
    full_text = f"{doc_content.lower()} {metadata_str}" # Lowercase consistency

    # Basic split
    return full_text.split()
    
def merge_json_files(input_folder, output_folder, file_name):
    """Merge multiple JSON files into a single JSON file."""

    logger.info(f"Merging json files for each document file into single file started...")
    all_documents = []
    all_metadata = []
    file_counter=0  #it is required because some files may have the same name

    for root, dirs, files in os.walk(input_folder):
        for file in files:     
            file_path = os.path.join(root, file)                                 
      
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)               
                    # Extend the lists with data from each file
                    all_documents.extend(data.get('documents', []))
                    all_metadata.extend(data.get('metadata', []))

            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

            file_counter += 1

    # Combine the lists into a single dictionary
    merged_data = {
        'documents': all_documents,
        'metadata': all_metadata
    }

    # Save the merged json file
    with open(output_folder + file_name, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    logger.info(f"Merging json files for each document file into single file finished...")

    logger.info("Loading raw data from JSON...")
    # Read the merged json file
    with open(os.path.join(output_folder, 'bm25_data_docs_and_metadata.json'), 'rb') as file:
        data = msgspec.json.decode(file.read(), type=Data)
    logger.info(f"Loaded {len(data.documents)} documents.")        

    logger.info("Preparing documents and IDs...")
    doc_ids = []
    documents_for_tokenization = []
    for i, (text, meta) in enumerate(zip(merged_data["documents"], merged_data["metadata"])):
        # --- IMPORTANT: Ensure this ID matches ChromaDB's ID for the same document ---
        doc_id = meta.get("id") # Or however you get/generate IDs
        doc_ids.append(doc_id)
        documents_for_tokenization.append({"content": text, "metadata": meta})


    # --- Tokenize Documents ---
    logger.info("Tokenizing all documents...")
    tokenized_corpus = [tokenize_for_bm25(doc["content"], doc["metadata"]) for doc in documents_for_tokenization]
    print("Tokenization complete.")

    logger.info("Tokenization complete.")


    # --- Build BM25 Index ---
    logger.info("Building BM25Okapi index...")
    # Use the same parameters you intend to use in the retriever
    bm25_index = BM25Okapi(tokenized_corpus, k1=1.0, b=0.5)
    logger.info("BM25 index built.")

    # --- Save Processed Data ---
    logger.info(f"Saving precomputed data to {output_folder}...")

    # Save the BM25 model object
    with open(os.path.join(output_folder, 'bm25_model.pkl'), 'wb') as f:
        pickle.dump(bm25_index, f)

    # Save the list of document IDs (in the exact same order as tokenized_corpus)
    with open(os.path.join(output_folder, 'doc_ids.pkl'), 'wb') as f:
        pickle.dump(doc_ids, f)

    logger.info("Precomputation finished.")

    logger.info("Merging finished...")
# ---------------------------------------------------------------------------------
def html_to_markdown(html_content, task_id, logger):
    """Convert HTML content to markdown"""

    # Parse HTML with BeautifulSoup (optional cleanup can be done here)
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
    except Exception as e:
        logger.exception(f"Failed to parse HTML content: {e}")
        return None, None
        
    # Remove excluded tags (form, header, footer, nav)
    for tag in soup(['form', 'header', 'footer', 'nav', 'iframe', 'script', 'a', 'img']):
        tag.decompose()    

    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    # Get plain text version
    text = soup.get_text('\n', strip=True)

    # #Initialize HTML2Text with your settings
    # h = html2text.HTML2Text()
    # h.ignore_links = True
    # h.ignore_images = True
    # h.skip_internal_links = True    

    # # Get cleaned HTML
    # cleaned_html = str(soup)
    
    # # Escape HTML if needed (for remaining tags you want to show as text)
    # cleaned_html = html.escape(cleaned_html)
    
    # # Convert to markdown
    # markdown = h.handle(cleaned_html)    

    # # Clean up excessive newlines
    # markdown = '\n'.join(line for line in markdown.split('\n') if line.strip())
    markdown = "0"

    return markdown, text  

# ---------------------------------------------------------------------------------
class RetryableOllamaEmbeddings:
    """Create Ollama embedding function which has retry capability in case of connection problem. """

    def __init__(self, logger, model: str = "bge-m3", base_url: str = "http://localhost:11434", retries: int = 5, delay: int = 1):
        self.embedding = OllamaEmbeddings(model=model, base_url=base_url)
        self.retries = retries
        self.delay = delay
        self.logger = logger

    def embed_documents(self, texts, task_id, logger):
        for attempt in range(1, self.retries + 1):
            try:
                return self.embedding.embed_documents(texts)
            except Exception as e:
                self.logger.exception(f"Attempt {attempt} failed: {e}")
                time.sleep(self.delay)
        logger.info("All retry attempts for embed_documents failed.")

    def embed_query(self, text):
        for attempt in range(1, self.retries + 1):
            try:
                return self.embedding.embed_query(text)
            except requests.exceptions.RequestException as e:
                logger.info(f"Attempt {attempt} failed: {e}")
                time.sleep(self.delay)
        raise Exception("All retry attempts for embed_query failed.")
    
# ---------------------------------------------------------------------------------

def get_doc_from_html(file_path, logger, file_no, file_path_print, task_id):
    """Read html documents, obtain text content and put them in document with metadata"""

    #Initialize output variables    
    error_flag = False         
    html_content = None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except FileNotFoundError as e:
        logger.error(f"File not found - file_counter:{file_no} -- {file_path_print}")

    text = None
    markdown = None                  
    try:    
        markdown, text = html_to_markdown(html_content, task_id, logger)
    except Exception as e:
        # Capture the exception and traceback
        error_flag = True  
        logger.exception(f"Error in task {task_id}: {e} - file_counter: {file_no} -- {file_path_print}")        
 
    full_text = text
    page_starts = []

    return full_text, page_starts, error_flag

# ---------------------------------------------------------------------------------
def get_doc_from_pdf_epub(file_path, logger, file_no, file_path_print, task_id):
    """Read pdf and epub documents, obtain text content and put them in document with metadata"""

    #Initialize output variables
    error_flag = False  
    full_text = ""
    page_starts = []

    if file_path.lower().endswith(".pdf"):
        try:
            doc = fitz.open(file_path)
            parts = []
            page_starts = []
            current = 0

            for page in doc:
                page_starts.append(current)           # this page starts here
                text = page.get_text("text")        # type: ignore[attr-defined]
                parts.append(text)
                current += len(text)

            full_text = "".join(parts)

        except Exception as e:
            # Capture the exception and traceback
            error_flag = True      
            logger.exception(f"Error in task {task_id}: {e} - file_counter: {file_no} -- {file_path_print}")
            
    elif file_path.lower().endswith(".epub"):
        try:
            book = epub.read_epub(file_path, options={'ignore_ncx': True})

            parts: list[str] = []
            page_starts: list[int] = []
            current = 0

            # spine = real reading order
            spine_ids = [item_id for (item_id, _) in book.spine]

            DOC_TYPE = getattr(ebooklib, "ITEM_DOCUMENT", 9)

            for item_id in spine_ids:
                item = book.get_item_with_id(item_id)
                if item is None:
                    continue
                if item.get_type() != DOC_TYPE:
                    continue

                html = item.get_content().decode("utf-8")
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator="\n", strip=True)

                # record start of this section
                page_starts.append(current)
                parts.append(text)

                # for strict parity with your PDF code, we join with "" (no extra \n)
                current += len(text)

            full_text = "".join(parts)
      
        except Exception as e:
            # Capture the exception and traceback
            error_flag = True      
            logger.exception(f"Error in task {task_id}: {e} - file_counter: {file_no} -- {file_path_print}") 
    
    return full_text, page_starts, error_flag

# ---------------------------------------------------------------------------------
def repair_start_indexes(full_text: str, chunks):
    cursor = 0
    for ch in chunks:
        si = ch.metadata.get("start_index", -1)
        if si is not None and si >= 0:
            cursor = si + len(ch.page_content)
            continue

        # try to find this chunk starting from the last cursor
        prefix = ch.page_content[:120]  # enough to be unique
        found = full_text.find(prefix, cursor)
        if found == -1:
            # fallback: search from beginning
            found = full_text.find(prefix)
        if found == -1:
            found = 0   # last resort

        ch.metadata["start_index"] = found
        cursor = found + len(ch.page_content)
# ---------------------------------------------------------------------------------
def create_batches(data, batch_size):
    """Yield successive batch_size-sized chunks from data."""

    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# ---------------------------------------------------------------------------------
def process_file(input_file, extensions, semaphore, collection, embedding_function, file_counter, lock, number_of_files, logger, task_id):
    """
    Read files, split them, embed them and put the embedded chunks to database
    """

    #This process deals with a single file. So, file counter increments once.
    with lock:
        file_counter.value += 1  
        file_no = file_counter.value   

    #persist_directory = "D:\\my_chroma_db_with_langchain"
    #client = chromadb.PersistentClient(path=persist_directory)
    #collection = client.get_or_create_collection(name="my-rag-db")


    for file_path, info  in input_file.items():
        #Obtain file path + name and file extension
        file_path_print = file_path[4:]  #From file path "\\\\?\\" part is removed
        file_extension = info[0][3]

        #Inintialize inputs
        error_flag = False  
        text = "" 
        page_starts = []

        if file_extension == ".html" or file_extension == ".htm":
            text, page_starts, error_flag = get_doc_from_html(file_path, logger, file_no, file_path_print, task_id)
        elif file_extension == ".pdf" or file_extension == ".epub":
            text, page_starts, error_flag = get_doc_from_pdf_epub(file_path, logger, file_no, file_path_print, task_id)

        #If no error and non-empty input text, continue processing of the current file    
        if not error_flag and (text is not None) and (len(text) > 0): 

            #Start text splitting into chunks
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=512,                   # ~512 tokens, not characters
                chunk_overlap=128,                # overlap also in tokens
                add_start_index=True
            )

            # Chunks will be obtained with start_indexes
            chunks = text_splitter.split_documents([Document(page_content=text, metadata={"source": file_path_print})])
            # Text splitter has bug inside, it cannot get index as -1 (error in getting). So, let's fix them.
            repair_start_indexes(text, chunks)
            # Add page_number and id info to the metadata of the chunks 
            for i, ch in enumerate(chunks):
                si = ch.metadata["start_index"]
                page_idx = bisect.bisect_right(page_starts, si) - 1
                ch.metadata["page"] = page_idx 
                ch.metadata["id"] =  f"doc_{file_no}_chunk_{i}"

            logger.info(f"Processed file no: {str(file_no)} / {str(number_of_files)} - len(chunks): {len(chunks)} - Adding file: {file_path_print}")


            #Create bm25 retriever and save it for one time
            data_to_save = {
                'documents': [chunk.page_content.encode('utf-8', errors='replace').decode('utf-8') for chunk in chunks],
                'metadata': [
                    {
                        "source": chunk.metadata["source"], 
                        "id": chunk.metadata["id"],
                        "page": chunk.metadata["page"],
                     
                    } for i, chunk in enumerate(chunks)
                ]
            }
            
            #For bm25 retriever, save the chunks obtained from one file/document for one time
            try:
                #folder = "D:\\Langchain-Langgraph-Doc-WebSites\\__Databases\\langchain-docs-bm25-db-temp\\"
                #folder = "D:\\PyMuPDF-Doc-WebSites\\__Databases\\pymupdf-docs-bm25-db-temp\\"
                folder = "D:\\Bilgi\\__Databases\\kitaplar-bm25-db-temp\\"

                with open(folder +f"bm25_data_docs_and_metadata_temp_{file_no}.json", 'w', encoding='utf-8') as file:   
                    json.dump(data_to_save, file, ensure_ascii=False, indent=4)   
            except Exception as e:
                logger.exception(f"Error in task {task_id} in json.dump: {e} - file_counter: {file_no} -- {file_path_print}")   


            # From this part on, embedded vector data is created and saved to database
            # Generate unique IDs for each chunk
            
            batch_size = 128  # Adjust based on your system's capacity
                            
            #Add chunks data in batches for speed consideration. Operation is same as one chunk at a time in chromadb.      
            for batch in create_batches(chunks, batch_size):
                # Precompute embeddings for the current batch
                batch_documents = [chunk.page_content.encode('utf-8', errors='replace').decode('utf-8') for chunk in batch]
                batch_metadata = [chunk.metadata for chunk in batch]
                batch_metadata_id = [chunk.metadata["id"] for chunk in batch]

                batch_embeddings = []
                try:
                    with semaphore:
                        batch_embeddings = embedding_function.embed_documents(batch_documents, task_id, logger)                            

                except Exception as e:
                    logger.exception(f"Embedding error - {task_id}: {e} - file_counter: {file_no} -- {file_path_print}")        


                try:
                    #time.sleep(random.uniform(0.1, 0.3))
                    if collection is not None:
                        collection.upsert(
                            documents=batch_documents,
                            metadatas=batch_metadata,
                            ids=batch_metadata_id,
                            embeddings=batch_embeddings,
                        )
                    else:
                        logger.error(f"Collection1 is not initialized - {task_id}: file_counter: {file_no} -- {file_path_print}")

                except Exception as e:
                    logger.exception(f"ChromaDB1 upsert error - {task_id}: {e} - file_counter: {file_no} -- {file_path_print}")


        elif text is None or len(text) == 0:
            logger.info(f"html file has no text. So, there is no embedded vector - {task_id}: file_counter: {file_no} -- {file_path_print}")

# ---------------------------------------------------------------------------------
def worker(file_queue, extensions, semaphore, file_counter, task_id, log_queue, lock, number_of_files):
    """
    Each worker is run under a different process in multiprocessing case
    Worker function to process files from the queue.
    Logger created, embedding function created, chromadb client created
    Chromadb server should be started with following command on powershell:
        chroma run --host localhost --port 8000 --path D:/Bilgi/__Databases/kitaplar-vectordb-embedding-bgem3
    """

    #Create logger
    logger = logging.getLogger(f'Worker-{task_id}')
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(handler)

    #For multiprocessing, create the embedding function having retry mechanism for Ollama embeddings

    embedding_function = RetryableOllamaEmbeddings(logger, model="bge-m3", retries=5, delay=1)


    # Initialize Chroma client with persistence
    settings = Settings(
        chroma_server_host="localhost",
        chroma_server_http_port=8000,
    )
    client_chromadb = chromadb.HttpClient(settings=settings)

    # Create two collections in Chroma DB for long embedding with 3072 vector size and short embedding with 768 vector size
    collection = None
    try:
        collection = client_chromadb.get_or_create_collection(name="my-doc-assistant-db")         
    except Exception as e:
        logger.exception(f"ChromaDB error - {task_id}: {e}")


    while True:
        input_file = None

        try:
            input_file = file_queue.get()
            if input_file is None:  # Sentinel value to signal the end
                break

            process_file(
                input_file,
                extensions, 
                semaphore,
                collection,
                embedding_function, 
                file_counter, 
                lock, 
                number_of_files,
                logger,
                task_id)
            
        except Exception as e:
            # Log the error from within the worker if process_file fails
            logger.error(f"Worker-{task_id} encountered error processing {input_file}: {e}", exc_info=True)
        finally:
            # This ensures task_done() is called whether process_file
            # succeeded, failed, or the input_file was None (sentinel)
            if input_file is not None or file_queue.empty(): # Check avoids calling task_done() twice for None if queue is already empty
                 file_queue.task_done()

# ---------------------------------------------------------------------------------
def process_worker(num_threads, file_queue, extensions, semaphore, file_counter, task_id, log_queue, lock, number_of_files):
    """
    Function run by each process to start threads.
    Not utilized now !!!
    """

    #Create logger
    logger = logging.getLogger(f'Worker-{task_id}')
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(handler)

    threads = []
    for _ in range(num_threads):
        t = Thread(target=worker, args=(file_queue, extensions, semaphore, file_counter, task_id, log_queue, lock, number_of_files))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()

# ---------------------------------------------------------------------------------
def configure_listener():
    """
    Create logger. Configures handlers for the listener.
    """
    # File handler
    file_handler = logging.FileHandler('rag_db_creation_multiprocessing.log')
    file_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    file_handler.setFormatter(file_formatter)

    ##Console handler
    # console_handler = logging.StreamHandler()
    # console_formatter = logging.Formatter('%(name)s %(levelname)s: %(message)s')
    # console_handler.setFormatter(console_formatter)
    #return [file_handler, console_handler]
    return [file_handler]

# ---------------------------------------------------------------------------------
def main():
    """
    Main function to set up multiprocessing and multithreading.

    """
    logger.info("------------------- Database creation stared...")

    #files_folder = '\\\\?\\D:\\Langchain-Langgraph-Doc-WebSites\\langchain-langgraph-docs'
    #files_folder = '\\\\?\\D:\\PyMuPDF-Doc-WebSites\\pymupdf-docs'
    files_folder = '\\\\?\\D:\\Bilgi\\Kitaplar'

    num_processes = 16 #int(os.cpu_count() / 2)  # Number of processes to match CPU cores
    
    max_concurrent_calls = 4  # Considering Ollama simultaneous call with retries
    semaphore = Semaphore(max_concurrent_calls)
    
    extensions=("html", "htm", "pdf", "epub")  # Specify the file extensions to look for
    logger.info("Data reading from folders started...")
    input_files, number_of_files = collect_files(files_folder, extensions)
    logger.info("Data reading from folders finished...")

    logger.info("Vectorizing process starts...")

    logger.info(f"Number of processes: {num_processes}")

    # Create a queue to hold the file paths
    file_queue = JoinableQueue()

    #Create a separate queue for logging
    log_queue = JoinableQueue()

    # Configure handlers for the listener
    handlers = configure_listener()
    # Set up the QueueListener with the handlers
    listener = logging.handlers.QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()    


    # Populate the queue with file paths
    for file_path, info in input_files.items():
        file_queue.put({file_path : info})

    for _ in range(num_processes):
        file_queue.put(None)        
        
    # Create a shared counter and a lock
    counter = Value('i', 0)
    lock = Lock()
    
    # Create and start processes
    processes = []
    for task_no in range(num_processes):
        p = Process(target=worker, args=(
            file_queue, extensions, semaphore, counter, task_no, log_queue, lock, number_of_files
            ))
        p.start()                             
        processes.append(p)
    
    while processes:
        for task_no, p in enumerate(processes):
            p.join(timeout=0.1)  # Non-blocking join
            if not p.is_alive():
                if p.exitcode == 0:
                    logger.info(f"Process {p.pid} completed successfully with exit code {p.exitcode}")
                    processes.pop(task_no)
                else:
                    logger.info(f"Process {p.pid} crashed with exit code {p.exitcode}. Restarting...")                            
                    new_p = Process(target=worker, args=(
                        file_queue, extensions, semaphore, counter, task_no, log_queue, lock, number_of_files
                        ))
                    new_p.start()                       
                    processes[task_no] = new_p
        time.sleep(1)        

    try:
    # Wait for all files to be processed
        file_queue.join()
        logger.info("file_queue is empty. ")


        #input_folder = "D:\\Langchain-Langgraph-Doc-WebSites\\__Databases\\langchain-docs-bm25-db-temp\\"
        #output_folder = "D:\\Langchain-Langgraph-Doc-WebSites\\__Databases\\langchain-docs-bm25-db\\"
        #input_folder = "D:\\PyMuPDF-Doc-WebSites\\__Databases\\pymupdf-docs-bm25-db-temp\\"
        #output_folder = "D:\\PyMuPDF-Doc-WebSites\\__Databases\\pymupdf-docs-bm25-db\\"
        input_folder = "D:\\Bilgi\\__Databases\\kitaplar-bm25-db-temp\\"
        output_folder = "D:\\Bilgi\\__Databases\\kitaplar-bm25-db\\"    

        file_name = "bm25_data_docs_and_metadata.json"
        try:
            merge_json_files(input_folder, output_folder, file_name)
            logger.info("All json temp files merged...")
        except Exception as e:
            logger.exception(f"merge_json_files() function error.")        

    except Exception as e:
        logger.exception("An error occurred in the main processing block.")
    
    finally:
        # Stop the listener
        listener.stop()    
        logger.info("listener stopped...")  

        logger.info("All processes finished...")  


if __name__ == "__main__":
    main()

