import json
import logging
import os

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.eval.gen_docs import documents_directory

# Embedding
from app.rag.embedding import setup_embedding


def create_vector_store(used_embedding_model, vector_store):
    # Setup embedding
    logging.info(f"Embedding model selected: {Settings.embed_model}")

    # Create storage context with the vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    logging.info("Created new storage context with QdrantVectorStore.")

    # Load documents
    documents = SimpleDirectoryReader(documents_directory).load_data()
    logging.info(f"Loaded {len(documents)} documents from {documents_directory}.")

    # Create or load VectorStoreIndex from documents
    logging.info("Attempting to create or load VectorStoreIndex from documents.")
    vector_index = None
    try:
        vector_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )
        logging.info("Created and persisted new VectorStoreIndex.")
    except Exception as e:
        logging.error(f"Failed to create VectorStoreIndex: {e}")
        raise Exception("Failed to create or load VectorStoreIndex.") from e
    return vector_index


def initialize_vector_store(local_mode):
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    vector_database = "Qdrant"
    cache_dir = os.getenv("CACHE_DIR", "/tmp/rag_cache")
    vector_store_path = cache_dir + "/qdrant_data"
    document_files = [
        f
        for f in os.listdir(documents_directory)
        if os.path.isfile(os.path.join(documents_directory, f))
        and not f.startswith(".")
    ]
    logging.info(f"Loading documents from {documents_directory} directory.")
    collection_name = "docs"
    used_embedding_model = setup_embedding(local_mode)
    vector_index = None
    logging.info(f"Initializing Qdrant client with data path: {vector_store_path}")
    client = (
        QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        if qdrant_api_key and qdrant_url and not local_mode
        else QdrantClient(path=vector_store_path)
    )
    logging.info("Creating QdrantVectorStore for the 'docs' collection.")
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    with open(os.path.join(vector_store_path, "meta.json"), "r") as meta_file:
        meta_data = json.load(meta_file)
    if not meta_data.get("collections"):
        logging.info(
            f"No collections found in {vector_store_path}/meta.json, skipping VectorStoreIndex loading."
        )
        vector_index = create_vector_store(local_mode, vector_store)
    else:
        try:
            logging.info("Attempting to load VectorStoreIndex.")

            vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        except Exception as e:
            logging.error(f"Failed to load VectorStoreIndex: {e}")
            vector_index = create_vector_store(local_mode, vector_store)
            raise e

    # Ensure VectorStoreIndex creation or loading was successful
    if vector_index is None:
        raise Exception("VectorStoreIndex creation or loading unsuccessful.")

    return document_files, vector_database, vector_index, used_embedding_model
