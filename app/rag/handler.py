import logging
import os

import requests
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager


def setup_langfuse(local_mode):
    langfuse_default_host = (
        "http://localhost:3000" if local_mode else "https://cloud.langfuse.com"
    )
    os.getenv("LANGFUSE_PUBLIC_KEY")
    os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_handler = None
    langfuse_host = os.getenv("LANGFUSE_HOST", langfuse_default_host)
    check = os.getenv("CHECK", "false").lower() == "true"
    if check:
        logging.info(f"Pinging {langfuse_host}/api/public/health for health check.")
        try:
            response = requests.get(f"{langfuse_host}/api/public/health", timeout=5)
            if response.ok:
                logging.info("Langfuse is running")
                langfuse_handler = LlamaIndexCallbackHandler()
                Settings.callback_manager = CallbackManager([langfuse_handler])
            else:
                print("Langfuse isn't running")
        except requests.exceptions.RequestException as e:
            print("Langfuse isn't running, error:", e)
    return langfuse_handler
