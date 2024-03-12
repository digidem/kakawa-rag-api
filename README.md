# Kakawa RAG API

The document retrieval API for Kakawa, a product-support bot that efficiently responds to questions given any documentation. It can run online with cutting-edge AI models, or offline on regular hardware. It can also be integrated with existing platforms such as WhatsApp.

This service leverages [LlamaIndex](https://llamaindex.ai) to perform efficient Augmented Retrieval Generation (RAG) queries, which are state-of-the-art techniques for using AI to respond to questions from pre-loaded documents.

Evaluation-driven-development is guaranteed by the [RAGAS](https://ragas.io) automatic evaluation framework, alongside [LangFuse](https://langfuse.com) dashboard for monitoring and manual evaluation.

## Features

### Offline first

The API is designed to work with both local and cloud-based models for generating embeddings and reponses, ensuring adaptability for different environments. Locally, it supports [Ollama](https://ollama.ai/) for calling Large Language Models (LLMs) and [fastembed](https://qdrant.github.io/fastembed/) for local embeddings. For cloud services, it integrates with OpenAI which include advanced models like GPT-3-Turbo and GPT-4, to provide fast and robust language understanding, and OpenAI's and Cohere embedding models.

RAGAS evaluations can run both online and offline, although it's highly recommended to run online with advance models such as GPT-4 for better results.

The monitoring dashboard, LangFuse, can be self-hosted and runs well offline. A free to use cloud hosted version is also [available](https://cloud.langfuse.com).

### Core Components

- **LlamaIndex**: Sets up with the OpenAI or Ollama model and integrates with LangFuse for logging and RAGAS for evaluation.
- **Embedding Model**: Selects and configures the embedding model.
- **Vector Store**: Initializes the documents and indexes them using [Qdrant](RAGatouille).
- **Indexing and Querying**: Builds an index from documents and creates a query engine for document retrieval.
- **Evaluation and Tracing**: Outlines the evaluation setup with RAGAS and includes provisions for LangFuse tracing.
- **API Endpoints**: Uses FastAPI for generating an API and documentation.

Dependencies for this script are managed by Poetry and are specified in the `pyproject.toml` file. Sensitive keys and configurations are stored in an `.env` file, which is not included in version control as indicated by the `.gitignore` file.

## Usage

### Running locally

In order to run the full stack locally you'll need to have QDrant, Ollama and LangFuse installed and running. You can run them all using Docker and docker-compose, skip to the section for details.

### Running on the cloud

To use the QDrant, OpenAI and LangFuse cloud platforms, follow these steps:

#### OpenAI Cloud
- Sign up for an OpenAI account at https://openai.com
- After signing up, navigate to the API section to obtain your API key
- Update the `.env` file with your OpenAI API key by setting the `OPENAI_API_KEY` variable

#### QDrant Cloud
- Sign up for a QDrant cloud account at https://cloud.qdrant.io
- Obtain your API key and QDrant URL
- Update the `.env` file with your QDrant API key and URL

#### LangFuse Cloud
- Sign up for a LangFuse cloud account at https://cloud.langfuse.com
- Obtain your public and secret keys
- Update the `.env` file with your LangFuse public and secret keys, and set the `LANGFUSE_HOST` to the cloud URL

### Using Python and Poetry

Before starting, make sure you've set up the stack locally or on the cloud, and added the keys and URLs to the `.env` file.

Make sure to have Python and [Poetry](https://python-poetry.org) installed. To run the API follow these steps:

1. Run `cp .env.example .env` and ensure all the required environment variables are set.
2. Install all dependencies using Poetry by running `poetry install`.
3. Activate the virtual environment created by Poetry with `poetry shell`.
4. Run the app with `uvicorn app.main:app`, add the `--reload` flag for development.

This will start a server on port `8000`, you can test the API with curl:
```
curl -X 'GET' \
  'http://127.0.0.1:8000/rag?query=How%20to%20use%20Mapeo%20tracks%20feature%3F&user_id=test_user&session_id=test_session' \
  -H 'accept: application/json'
```
FastAPI automatically provides [swagger documentation](http://localhost:8000/docs) and [interactive documentation](http://localhost:8000/redoc).

### Using Docker

Alternatively, if you are using Docker:

1. Build the Docker image with `docker build -t kakakwa-rag-api .`
2. Run the Docker container with `docker run --env-file .env -v ./docs:/usr/src/app/docs --network host kakakwa-rag-api`

To run the full stack (QDrant, Ollama and LangFuse) use [docker-compose]():
1. Ensure all the required environment variables are set in the `.env` file.
2. Start the service with `docker-compose up`, or just desired services such as `docker-compose up langfuse-server langfuse-db`.

## Roadmap

- [x] Basic `/rag` endpoint with FastAPI
- [X] Working RAG querying
- [x] Integration with LangFuse
- [x] Offline-first RAG
- [ ] Integrate a cloud-hosted vector-store
- [ ] Caching of vector-database and embeddings
- [ ] RAGAS test data generation
- [ ] RAGAS evaluation
- [ ] Tuning of RAG: tecniques, embeddings, top_k, chunk sizes...

## Further exploration

- [RAGatouille](https://github.com/bclavie/RAGatouille)
- [Genstruct-7B](https://huggingface.co/NousResearch/Genstruct-7B)