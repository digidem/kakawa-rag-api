import logging
import os
from shutil import rmtree

from llama_index.core import (
    ServiceContext,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


def initialize_vector_store(local_mode):
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    vector_database = "Qdrant"
    vector_store_path = "./qdrant_data"
    documents_directory = "./docs"
    document_files = [
        f
        for f in os.listdir(documents_directory)
        if os.path.isfile(os.path.join(documents_directory, f))
        and not f.startswith(".")
    ]
    logging.info("Loading documents from './docs/' directory.")
    documents = SimpleDirectoryReader(documents_directory).load_data()

    logging.info("Initializing Qdrant client with data path './qdrant_data'.")
    client = (
        QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        if qdrant_api_key and qdrant_url and not local_mode
        else QdrantClient(path=vector_store_path)
    )
    logging.info("Creating QdrantVectorStore for the 'docs' collection.")
    vector_store = QdrantVectorStore(client=client, collection_name="docs")

    logging.info("Setting up storage context with the vector store.")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    logging.info("Setting up service context with LLM and embedding model.")
    service_context = ServiceContext.from_defaults(
        llm=Settings.llm, embed_model=Settings.embed_model
    )
    splitter = SentenceSplitter(chunk_size=256)
    index = None
    logging.info("Creating VectorStoreIndex from documents.")
    try:
        index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context,
            storage_context=storage_context,
            transformations=[splitter],
        )
    except Exception as e:
        logging.error(f"Failed to create VectorStoreIndex on first attempt: {e}")
        try:
            rmtree(vector_store_path)
            logging.info(f"Deleted vector store directory: {vector_store_path}")
            index = VectorStoreIndex.from_documents(
                documents,
                service_context=service_context,
                storage_context=storage_context,
                transformations=[splitter],
            )
        except Exception as e:
            logging.error(f"Failed to create VectorStoreIndex on second attempt: {e}")
            raise Exception(
                "Unable to create VectorStoreIndex after multiple attempts."
            )

    if index is None:
        raise Exception("VectorStoreIndex could not be created.")

    return document_files, vector_database, index
