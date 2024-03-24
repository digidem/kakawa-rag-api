import os

from llama_index.core import Settings
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


def setup_embedding(local_mode):
    local_embedding = os.getenv("LOCAL_EMBEDDING", "false").lower() == "true"
    cohere_api_key = os.getenv("COHERE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    used_embedding_model = None
    default_together_embedding_model = "togethercomputer/m2-bert-80M-8k-retrieval"
    default_openai_embedding_model = "text-embedding-3-small"
    default_cohere_embedding_model = "embed-english-v3.0"
    default_baai_embedding_model = "BAAI/bge-small-en-v1.5"
    if local_mode or local_embedding:
        used_embedding_model = default_baai_embedding_model
        Settings.embed_model = FastEmbedEmbedding()
    elif openai_api_key:
        used_embedding_model = os.getenv(
            "EMBEDDING_MODEL", default_openai_embedding_model
        )
        Settings.embed_model = OpenAIEmbedding(
            api_key=openai_api_key, model_name=used_embedding_model
        )
    elif cohere_api_key:
        used_embedding_model = os.getenv(
            "EMBEDDING_MODEL", default_cohere_embedding_model
        )
        Settings.embed_model = CohereEmbedding(
            cohere_api_key=cohere_api_key,
            model_name=used_embedding_model,
            input_type="search_document",
        )

    else:
        raise ValueError("No API key found for OpenAI or Cohere.")
    return used_embedding_model
