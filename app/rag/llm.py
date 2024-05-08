import json
import logging
import os

import requests
from llama_index.core import Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.together import TogetherLLM


def setup_llm(local_mode, eval=False):
    openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    ollama_model = os.getenv("OLLAMA_MODEL", "phi")
    togetherai_api_key = os.getenv("TOGETHERAI_API_KEY")
    togetherai_model = os.getenv("TOGETHERAI_MODEL", "microsoft/phi-2")
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_timeout = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
    eval_model = os.getenv("EVAL_MODEL", "openhermes2.5-mistral")

    if eval and not local_mode:
        if eval_model.startswith("claude"):
            tokenizer = Anthropic().tokenizer
            Settings.tokenizer = tokenizer
            Settings.llm = Anthropic(
                temperature=0, model=eval_model, api_key=anthropic_api_key
            )
        elif eval_model.startswith("gpt"):
            Settings.llm = OpenAI(
                temperature=0,
                model=eval_model,
                api_key=openai_api_key,
                openai_base_url=openai_base_url,
            )
        else:
            Settings.llm = Ollama(
                model=eval_model,
                request_timeout=ollama_timeout,
                base_url=ollama_base_url,
            )
    else:
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
            Settings.llm = TogetherLLM(
                model=togetherai_model, api_key=togetherai_api_key
            )
        elif used_llm == groq_model:
            Settings.llm = Groq(model=groq_model, api_key=groq_api_key)
        elif used_llm == ollama_model:
            # Check if the OLLAMA_MODEL is available
            try:
                logging.info(f"Checking if {ollama_model} is available...")
                response = requests.post(
                    f"{ollama_base_url}/api/show",
                    data=json.dumps({"name": ollama_model}),
                )
                if not response.ok:
                    print(response)
                    print(f"Error checking {ollama_model}: {response.text}")
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Request to check {ollama_model} failed: {e}")
            Settings.llm = Ollama(
                model=ollama_model,
                request_timeout=ollama_timeout,
                base_url=ollama_base_url,
            )
        else:
            raise ValueError(f"No LLM configured for model: {used_llm}")
    return Settings.llm
