from fastapi import FastAPI
from langchain_core.runnables import RunnableConfig
from fastapi.responses import StreamingResponse
from langchain_core.documents import Document
from pydantic import BaseModel
from typing import cast, Dict
import json
import os, re
import logging
import asyncio
from copy import deepcopy

from custom_rag_graph import CustomRagGraph

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global dictionary to store running tasks, mapping chat_id to its asyncio.Task
running_tasks: Dict[str, asyncio.Task] = {}

# 1. Create an instance of your RAG class
rag_instance = CustomRagGraph()
# 2. Build/Compile the graph and store it
rag_graph = rag_instance.get_rag_graph()


class QueryRequest(BaseModel):
    chat_id: str
    query: str

class AbortRequest(BaseModel):
    chat_id: str


async def generate_rag_response(chat_id: str, query: str, queue: asyncio.Queue):
    """
    The core RAG logic.
    Instead of 'yielding' tokens, it 'puts' them into an asyncio.Queue.
    """

    inputs = {
        "query": query,
        "new_queries": [],
        "documents": []
    }

    state = cast(CustomRagGraph.GraphState, inputs)
    retrieved_docs = []

    # Buffer specifically for tag parsing
    accumulated_answer = ""
    accumulated_citation_ids = set()
    nodes_executed = set()
    previous_answer_length = 0  # Track what we've already sent

    try:
        config: RunnableConfig = RunnableConfig(
            configurable={"thread_id": chat_id},
            recursion_limit=100
        )        
        async for step in rag_graph.astream(state, stream_mode=["messages", "updates"], config=config):
            mode, payload = step

            # Debug: log node updates
            if mode == "updates" and isinstance(payload, dict):
                logger.info(f"Update from nodes: {list(payload.keys())}")

            # Handle State Updates (for sources and progress tracking)
            if mode == "updates":
                if isinstance(payload, dict):
                    # Check which node is executing and emit progress
                    # Each node appears once in payload when it completes

                    # PROGRESS: enrich_query node → "analyzing"
                    if "enrich_query" in payload and "enrich_query" not in nodes_executed:
                        nodes_executed.add("enrich_query")
                        progress_event = json.dumps({"type": "progress", "stage": "analyzing"}) + "\n"
                        await queue.put(progress_event)
                        logger.info("Stage: analyzing")

                    # PROGRESS: retrieve node → "retrieving"
                    if "retrieve" in payload and "retrieve" not in nodes_executed:
                        nodes_executed.add("retrieve")
                        progress_event = json.dumps({"type": "progress", "stage": "retrieving"}) + "\n"
                        await queue.put(progress_event)
                        logger.info("Stage: retrieving")

                    # PROGRESS: generate node → "generating"
                    if "generate" in payload and "generate" not in nodes_executed:
                        nodes_executed.add("generate")
                        progress_event = json.dumps({"type": "progress", "stage": "generating"}) + "\n"
                        await queue.put(progress_event)
                        logger.info("Stage: generating")

                        node_output = payload["generate"]
                        if isinstance(node_output, dict):
                            if "documents" in node_output:
                                retrieved_docs = node_output.get("documents", [])
                                retrieved_citations = node_output.get("citations", [])
                                answer = node_output.get("answer", [])

                                if retrieved_citations:
                                    for citation in retrieved_citations:
                                        citation_id = citation.doc_id
                                        if citation_id:
                                            accumulated_citation_ids.add(citation_id)

                                if answer:
                                    # Get the new text that was added
                                    new_text = answer[previous_answer_length:] if len(answer) > previous_answer_length else ""

                                    if new_text:
                                        # Split by words while preserving newlines and structure
                                        # Use regex to split on spaces but keep newlines, tabs, etc.
                                        # Split into tokens: words and whitespace chunks
                                        tokens = re.findall(r'\S+|\s+', new_text)

                                        # Debug: Check if newlines are in the tokens
                                        has_newlines = any('\n' in t for t in tokens)
                                        if has_newlines:
                                            logger.info(f"Sending {len(tokens)} tokens (includes newlines)")

                                        for token in tokens:
                                            # Send each token (word or whitespace block)
                                            chunk_event = json.dumps({"type": "chunk", "content": token}) + "\n"
                                            await queue.put(chunk_event)

                                            # Only add delay for actual words, not for whitespace
                                            if token.strip():  # If it's a word (not just whitespace)
                                                await asyncio.sleep(0.03)  # Small delay for smooth word-by-word effect

                                        previous_answer_length = len(answer)

            # Handle Final Message Content Streaming
            # elif mode == "messages":
            #     message_chunk, metadata_dict = payload
            #     if isinstance(metadata_dict, dict) and metadata_dict.get('langgraph_node') == 'generate':
            #         if not isinstance(message_chunk, str) and hasattr(message_chunk, 'content'):
            #             chunk = message_chunk.content
            #         else:
            #             chunk = message_chunk

            #         if chunk:
            #             accumulated_answer += chunk

            #             await queue.put(accumulated_answer)
                        

        logger.info(f"Stream generation finished normally for query: {query}")

        # 6. Build final list of cited document objects
        documents = retrieved_docs
        cited_documents = []
    
        for doc in documents:
            # Check if this doc's ID was found in the stream
            if doc.metadata.get("id") in accumulated_citation_ids:
                cited_doc = deepcopy(doc)
                
                # Your modification logic
                cited_doc.page_content = cited_doc.metadata["id"] + " -- " + cited_doc.page_content
                cited_doc.metadata["id"] += "_"
                cited_doc.metadata["source"] += "_"
                cited_documents.append(cited_doc)

        for doc in retrieved_docs:
            doc.page_content = doc.metadata["id"] + " --- " + doc.page_content
        
        # 7. Yield the final documents list
        retrieved_docs += [Document(page_content="", metadata={"start_index": 0, "page":0, "source":"--","id":"--"})] + cited_documents    

        # After the stream is complete, output the sources
        references_data = []
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                page_info = str(int(doc.metadata.get("page", 0)) + 1)
                
                references_data.append({
                    "id": f"ref_{i+1}",
                    "source": doc.metadata.get("source", "Unknown Source"),
                    "page": page_info, 
                    "content": doc.page_content
                })

        # Serialize with JSON and send as a chunk event
        if references_data:
            json_payload = json.dumps(references_data)
            references_chunk = f"\n\n__JSON_START__{json_payload}"
            chunk_event = json.dumps({"type": "chunk", "content": references_chunk}) + "\n"
            await queue.put(chunk_event)
        else:
            references_chunk = "\n\n__JSON_START__[]"
            chunk_event = json.dumps({"type": "chunk", "content": references_chunk}) + "\n"
            await queue.put(chunk_event)

        # Send done event
        done_event = json.dumps({"type": "done"}) + "\n"
        await queue.put(done_event)

    except asyncio.CancelledError:
        logger.warning(f"LangGraph generation cancelled for chat_id: {chat_id}")
        raise 
    except Exception as e:
        logger.error(f"Unexpected error during LangGraph streaming for chat_id: {chat_id}: {e}")
        error_event = json.dumps({"type": "error", "message": str(e)}) + "\n"
        await queue.put(error_event)
    finally:
        await queue.put(None)

async def run_generation_task(chat_id: str, query: str, queue: asyncio.Queue):
    """
    Task wrapper for run_generation_task.
    This manages the lifecycle and cleanup of the task.
    """
    try:
        await generate_rag_response(chat_id, query, queue)
    except asyncio.CancelledError:
        logger.warning(f"Task wrapper caught cancellation for chat_id: {chat_id}")
    except Exception as e:
        logger.error(f"Task wrapper caught exception for chat_id: {chat_id}: {e}")
        # You could put a specific error message in the queue if needed
        # await queue.put(f"__ERROR__{e}")
    finally:
        # Task is done (completed, cancelled, or failed), remove it
        running_tasks.pop(chat_id, None)
        logger.info(f"Cleaned up task from running_tasks for chat_id: {chat_id}")

async def stream_from_queue(queue: asyncio.Queue, generation_task: asyncio.Task):
    """
    Generator that yields items from a queue until 'None' is received.
    It also links client disconnection to task cancellation.
    """
    try:
        while True:
            item = await queue.get()
            # logger.info(f"\n\nchunk get from queue: {item}")
            if item is None:
                # End of stream signal
                break
            yield item
    except asyncio.CancelledError:
        # This block executes if the client (Node.js) disconnects
        logger.warning(f"Stream reader cancelled (client disconnected). Cancelling generation task.")
        if not generation_task.done():
            generation_task.cancel()
    finally:
        # Ensure the task is awaited to allow cleanup, even if reader is cancelled
        if not generation_task.done():
            try:
                await generation_task
            except asyncio.CancelledError:
                pass # We just cancelled it, this is expected


@app.post("/query")
async def query_rag(payload: QueryRequest):
    """
    Handles the query request, manages tasks, and returns the stream.
    """
    chat_id = payload.chat_id
    query = payload.query

    # If a task for this chat_id is already running, cancel it
    if chat_id in running_tasks:
        old_task = running_tasks[chat_id]
        if not old_task.done():
            logger.warning(f"Cancelling previous task for chat_id: {chat_id}")
            old_task.cancel()
            try:
                await old_task # Wait for it to clean up
            except asyncio.CancelledError:
                pass # Expected

    # Create a new queue for this request
    queue = asyncio.Queue()
    
    # Create the task for the generator
    generation_task = asyncio.create_task(
        run_generation_task(chat_id, query, queue)
    )
    
    # Store the handle to the task
    running_tasks[chat_id] = generation_task
    
    # Return the streaming response which reads from the queue
    return StreamingResponse(
        stream_from_queue(queue, generation_task), 
        media_type="text/plain"
    )

@app.post("/abort")
async def abort_task(body: AbortRequest):
    """
    Receives an abort request from Node.js and cancels the running task.
    """
    
    chat_id = body.chat_id
    task = running_tasks.get(chat_id)
    
    if not task or task.done():
        logger.warning(f"No active task found to abort for chat_id: {chat_id}")
        return {"message": "No active task to abort."}

    logger.info(f"Received abort request for chat_id: {chat_id}. Cancelling task.")
    task.cancel()
    
    try:
        await task # Wait for the task to finish cancelling and clean up
    except asyncio.CancelledError:
        logger.info(f"Task for chat_id {chat_id} successfully cancelled via abort endpoint.")
    
    return {"message": "aborted"}


@app.get("/health")
async def health():
    return {"status": "ok", "message": "Python RAG service is running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)