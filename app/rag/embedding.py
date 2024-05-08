import logging
import os
import time

from llama_index.core import Settings
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


def setup_embedding(local_mode):
    start_time = time.time()
    cache_dir = os.getenv("CACHE_DIR", "/tmp/rag_cache")
    embedding_cache = cache_dir + "/embeddings"
    logging.info("Setting up embedding model.")
    local_embedding = os.getenv("LOCAL_EMBEDDING", "false").lower() == "true"
    cohere_api_key = os.getenv("COHERE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    used_embedding_model = None
    default_together_embedding_model = "togethercomputer/m2-bert-80M-8k-retrieval"
    default_openai_embedding_model = "text-embedding-3-small"
    default_cohere_embedding_model = "embed-english-v3.0"
    default_baai_embedding_model = "BAAI/bge-small-en-v1.5"

    logging.info(f"local_mode: {local_mode}, local_embedding: {local_embedding}")
    if local_mode or local_embedding:
        used_embedding_model = default_baai_embedding_model
        logging.info("Setting up FastEmbedEmbedding.")
        try:
            Settings.embed_model = FastEmbedEmbedding(
                model_name=used_embedding_model, cache_dir=embedding_cache
            )
            logging.info(f"Using local embedding model: {used_embedding_model}")
        except Exception as e:
            logging.error(f"Failed to set up FastEmbedEmbedding: {e}")
    elif openai_api_key:
        used_embedding_model = os.getenv(
            "EMBEDDING_MODEL", default_openai_embedding_model
        )
        Settings.embed_model = OpenAIEmbedding(
            api_key=openai_api_key, model_name=used_embedding_model
        )
        logging.info(f"Using OpenAI embedding model: {used_embedding_model}")
    elif cohere_api_key:
        used_embedding_model = os.getenv(
            "EMBEDDING_MODEL", default_cohere_embedding_model
        )
        Settings.embed_model = CohereEmbedding(
            cohere_api_key=cohere_api_key,
            model_name=used_embedding_model,
            input_type="search_document",
        )
        logging.info(f"Using Cohere embedding model: {used_embedding_model}")
    else:
        logging.error("No API key found for OpenAI or Cohere.")
        raise ValueError("No API key found for OpenAI or Cohere.")

    initialization_time = time.time() - start_time
    logging.info(
        f"Embedding model setup completed in {initialization_time:.2f} seconds"
    )
    return used_embedding_model
