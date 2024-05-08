import logging
import os
import sys
import time
from shutil import rmtree

import requests
from dotenv import load_dotenv

# LlamaIndex
from llama_index.core import Settings

# Handler
from app.rag.handler import setup_langfuse

# LLMS
from app.rag.llm import setup_llm

# Prompt
from app.rag.prompt import update_prompt

# TODO Retrievers
# https://docs.llamaindex.ai/en/latest/examples/retrievers/recursive_retriever_nodes.html
from app.rag.retrieval import initialize_colbert_reranker

# Vectore Store
from app.rag.vector_store import initialize_vector_store

# Load environment variables
start_time = time.time()
load_dotenv()
local_mode = os.getenv("LOCAL_MODE", "false").lower() == "true"
top_k = int(os.getenv("TOP_K", "4"))
check = os.getenv("CHECK", "false").lower() == "true"

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

if local_mode:
    print("Running in local mode")
else:
    if check:
        connectivity_test_url = "https://httpbin.org/get"
        try:
            with requests.get(connectivity_test_url, timeout=5) as response:
                response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
                print("Running in cloud mode")
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            print(f"Cloud mode connectivity failed ({e}), switching to local mode")
            local_mode = True
    else:
        print("Running in cloud mode without connectivity check")
# Setup LLM
used_llm = setup_llm(local_mode)
logging.info(f"Using LLM: {Settings.llm.__class__.__name__}")
# Setup Langfuse as handler
if os.getenv("TEST", "false").lower() != "true":
    langfuse_handler = setup_langfuse(local_mode)
    logging.info("Using LangFuse handler")
else:
    langfuse_handler = None

# Setup vector store
document_files, vector_database, vector_index, used_embedding_model = (
    initialize_vector_store(local_mode)
)

colbert_reranker, retrieval_strategy = initialize_colbert_reranker(top_k)
logging.info(f"Creating query engine with similarity top k set to '{top_k}'.")
query_engine = vector_index.as_query_engine(
    similarity_top_k=top_k, node_postprocessors=[colbert_reranker]
)

update_prompt(query_engine)
# Log all variables in metadata
metadata = {
    "used_llm": used_llm,
    "llm": Settings.llm.__class__.__name__,
    "used_embedding_model": used_embedding_model,
    "embedding_model": Settings.embed_model.__class__.__name__,
    "top_k": top_k or None,
    "docs": document_files,
    "retrieval": retrieval_strategy,
    "vector_database": vector_database,
}

for key, value in metadata.items():
    logging.info(f"Metadata - {key}: {value}")

initialization_time = time.time() - start_time
logging.info(f"RAG app initialized in {initialization_time:.2f} seconds")


def rag(query, user_id="test_user", session_id="test_session"):
    if langfuse_handler is not None:
        langfuse_handler.set_trace_params(
            name="kakawa-qa",
            metadata=metadata,
            user_id=user_id,
            session_id=session_id,
            release="alpha",
            version="0.0.1",
            tags=["production"] if os.getenv("PRODUCTION") else ["development"],
        )
    response_vector = query_engine.query(query)
    logging.info(f"Query response: {response_vector}")
    return response_vector
