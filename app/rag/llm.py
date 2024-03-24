import json
import logging
import os

import requests
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.together import TogetherLLM


def setup_llm(local_mode):
    openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    ollama_model = os.getenv("OLLAMA_MODEL", "phi")
    togetherai_api_key = os.getenv("TOGETHERAI_API_KEY")
    togetherai_model = os.getenv("TOGETHERAI_MODEL", "microsoft/phi-2")
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_timeout = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))

    # Setup LLM
    used_llm = (
        ollama_model
        if local_mode
        else (
            groq_model
            if groq_api_key
            else (
                togetherai_model
                if togetherai_api_key
                else (openai_model if openai_api_key else ollama_model)
            )
        )
    )
    if used_llm == openai_model:
        Settings.llm = OpenAI(
            temperature=0.1,
            model=openai_model,
            api_key=openai_api_key,
            openai_base_url=openai_base_url,
        )
    elif used_llm == togetherai_model:
        Settings.llm = TogetherLLM(model=togetherai_model, api_key=togetherai_api_key)
    elif used_llm == groq_model:
        Settings.llm = Groq(model=groq_model, api_key=groq_api_key)
    else:
        # Check if the OLLAMA_MODEL is available
        try:
            logging.info(f"Checking if {ollama_model} is available...")
            response = requests.post(
                f"{ollama_base_url}/api/show", data=json.dumps({"name": ollama_model})
            )
            if not response.ok:
                print(response)
                print(f"Error checking {ollama_model}: {response.text}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Request to check {ollama_model} failed: {e}")
        Settings.llm = Ollama(
            model=ollama_model, request_timeout=ollama_timeout, base_url=ollama_base_url
        )
    return Settings.llm
