from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langfuse import Langfuse
from pydantic import BaseModel

from app.internal.rag import rag

# Init fastapi
langfuse = Langfuse()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Operation on startup
    yield  # wait until shutdown
    # Flush all events to be sent to Langfuse on shutdown and terminate all Threads gracefully. This operation is blocking.
    langfuse.flush()


app = FastAPI(lifespan=lifespan)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# FastAPI
@app.get("/rag")
async def prompt_rag(
    query: str,
    user_id: Optional[str] = "test_user",
    session_id: Optional[str] = "test_session",
):
    response = rag(query, user_id=user_id, session_id=session_id)
    return {"message": response}
