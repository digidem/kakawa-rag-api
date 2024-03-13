import logging
import os
import sys

from dotenv import load_dotenv

# langfuse
from langfuse.llama_index import LlamaIndexCallbackHandler

# LlamaIndex
# from llama_index.settings import Settings
from llama_index.core import (
    PromptTemplate,
    ServiceContext,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.callbacks import CallbackManager

# from llama_index.core.ingestion import IngestionCache, IngestionPipeline
# from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding

# Embedding
from llama_index.embeddings.openai import OpenAIEmbedding

# LLMS
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.vector_stores.qdrant import QdrantVectorStore

# qdrant
from qdrant_client import QdrantClient

# Load environment variables
langfuse_default_host = (
    "http://localhost:3000"
    if os.getenv("OFFLINE", "false").lower() == "true"
    else "https://cloud.langfuse.com"
)
load_dotenv()
os.getenv("LANGFUSE_PUBLIC_KEY")
os.getenv("LANGFUSE_SECRET_KEY")
os.getenv("LANGFUSE_HOST", langfuse_default_host)
openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
ollama_model = os.getenv("OLLAMA_MODEL", "gemma:2b")
openai_api_key = os.getenv("OPENAI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
offline_mode = os.getenv("OFFLINE", "false").lower() == "true"
local_embedding = os.getenv("LOCAL_EMBEDDING", "false").lower() == "true"

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Setup llamaindex
used_llm = openai_model if openai_api_key and not offline_mode else ollama_model
if used_llm == openai_model:
    Settings.llm = OpenAI(temperature=0.1, model=openai_model, api_key=openai_api_key)
else:
    Settings.llm = Ollama(model=ollama_model, request_timeout=120.0)
# Setup Langfuse as handler
langfuse_handler = LlamaIndexCallbackHandler()
Settings.callback_manager = CallbackManager([langfuse_handler])

# Setup embedding
used_embedding_model = None
default_openai_embedding_model = "text-embedding-3-small"
default_cohere_embedding_model = "embed-english-v3.0"
default_baai_embedding_model = "BAAI/bge-small-en-v1.5"
if offline_mode or local_embedding:
    used_embedding_model = default_baai_embedding_model
    Settings.embed_model = FastEmbedEmbedding()
elif openai_api_key:
    used_embedding_model = os.getenv("EMBEDDING_MODEL", default_openai_embedding_model)
    Settings.embed_model = OpenAIEmbedding(
        api_key=openai_api_key, model_name=used_embedding_model
    )
elif cohere_api_key:
    used_embedding_model = os.getenv("EMBEDDING_MODEL", default_cohere_embedding_model)
    Settings.embed_model = CohereEmbedding(
        cohere_api_key=cohere_api_key,
        model_name=used_embedding_model,
        input_type="search_document",
    )
    raise ValueError("No API key found for OpenAI or Cohere.")
# Log embedding and LLM details
logging.info(f"Using embedding model: {Settings.embed_model.__class__.__name__}")
logging.info(f"Using LLM: {Settings.llm.__class__.__name__}")

vector_store_path = "./qdrant_data"
# Vector store
# Read the contents of the ./docs directory and create an array with file names
documents_directory = "./docs"
document_files = [
    f
    for f in os.listdir(documents_directory)
    if os.path.isfile(os.path.join(documents_directory, f)) and not f.startswith(".")
]
logging.info("Loading documents from './docs/' directory.")
documents = SimpleDirectoryReader(documents_directory).load_data()

logging.info("Initializing Qdrant client with data path './qdrant_data'.")
client = (
    QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    if qdrant_api_key and qdrant_url and not offline_mode
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
logging.info("Creating VectorStoreIndex from documents.")
index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context,
    storage_context=storage_context,
)

# Querying
top_k = 10
logging.info("Loading ColbertRerank.")
colbert_reranker = ColbertRerank(
    top_n=top_k,
    model="colbert-ir/colbertv2.0",
    tokenizer="colbert-ir/colbertv2.0",
    keep_retrieval_score=True,
)

logging.info(f"Creating query engine with similarity top k set to '{top_k}'.")
query_engine = index.as_query_engine(
    similarity_top_k=top_k, node_postprocessors=[colbert_reranker]
)
qa_prompt_tmpl_str = (
    "You are an expert Q&A system that is trusted around the world.\n"
    "Always answer the query using the provided context information, and not prior knowledge.\n"
    "Some rules to follow:\n"
    "1. Never directly reference the given context in your answer.\n"
    "2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.\n"
    "3. If the context is not relevant to the query, respond you're unaware'.\n"
    "user: Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Query: {query_str}\n"
    "Answer: \n"
    "assistant: "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

# Test
# Log all variables in metadata
metadata = {
    "used_llm": used_llm,
    "llm": Settings.llm.__class__.__name__,
    "used_embedding_model": used_embedding_model,
    "embedding_model": Settings.embed_model.__class__.__name__,
    "top_k": top_k or None,
    "docs": document_files,
}

for key, value in metadata.items():
    logging.info(f"Metadata - {key}: {value}")


def rag(query, user_id="test_user", session_id="test_session"):
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
