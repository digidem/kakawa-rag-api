# Kakawa API

The API app for Kakawa, a product-support bot that efficiently responds to questions given any documentation. It can run offline on regular hardware, and can also be integrated with existing platforms such as WhatsApp.

The Kakawa API utilizes [LlamaIndex](https://llamaindex.ai) to perform efficient augmented retrieval generation (RAG) queries, which are state-of-the-art techniques for using AI to respond to questions from any pre-loaded documents.

Evaluation-driven-development is guaranteed by the [RAGAS](https://ragas.io) automatic evaluation framework, alongside [LangFuse](https://langfuse.com) dashboard for monitoring and manual evaluation.

## Features

### Offline first

The API is designed to work with both local and cloud-based models for generating embeddings and reponses, ensuring adaptability for different environments. Locally, it supports [Ollama](https://ollama.ai/) for calling Large Language Models (LLMs) and [fastembed](https://qdrant.github.io/fastembed/) for local embeddings. For cloud services, it integrates with OpenAI which include advanced models like GPT-3-Turbo and GPT-4, to provide fast and robust language understanding, and OpenAI's and Cohere embedding models.

RAGAS evaluations can run both online and offline, although it's highly recommended to run online with advance models such as GPT-4 for better results.

The monitoring dashboard, LangFuse, can be self-hosted and runs well offline. A free to use cloud hosted version is also [available](https://cloud.langfuse.com).

### Core Components

- **LlamaIndex**: Sets up with the OpenAI or Ollama model and integrates with LangFuse for logging and RAGAS for evaluation.
- **Embedding Model**: Selects and configures the embedding model.
- **Vector Store**: Initializes the documents and indexes them using [Qdrant](qdrant.github.io).
- **Indexing and Querying**: Builds an index from documents and creates a query engine for document retrieval.
- **Evaluation and Tracing**: Outlines the evaluation setup with RAGAS and includes provisions for LangFuse tracing.
- **API Endpoints**: Uses FastAPI for generating an API and documentation.

Dependencies for this script are managed by Poetry and are specified in the `pyproject.toml` file. Sensitive keys and configurations are stored in an `.env` file, which is not included in version control as indicated by the `.gitignore` file.

## Usage

To run, follow these steps:

1. Run `cp .env.example .env` and ensure all the required environment variables are set.
2. Install all dependencies using Poetry by running `poetry install`.
3. Activate the virtual environment created by Poetry with `poetry shell`.
4. Run the app with `uvicorn app.main:app`, add the `--reload` flag for development.

Alternatively, if you are using Docker:

1. Build the Docker image with `docker build -t kakakwa-api .`
2. Run the Docker container with `docker run --env-file .env -v ./docs:/usr/src/app/docs --network host kakakwa-api`

For Docker Compose:

1. Ensure all the required environment variables are set in the `.env` file.
2. Start the service with `docker-compose up`, or just desired services such as `docker-compose up langfuse-server langfuse-db`.

## Roadmap

- [x] Basic `/rag` endpoint with FastAPI
- [X] Working RAG querying
- [x] Integration with LangFuse
- [x] Offline-first RAG
- [ ] Caching of vector-database and embeddings
- [ ] RAGAS test data generation
- [ ] RAGAS evaluation
- [ ] Tuning of RAG: tecniques, embeddings, top_k, chunk sizes...