from contextlib import asynccontextmanager
from typing import Optional

from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline
from langfuse import Langfuse
from pydantic import BaseModel

from app.internal.rag import langfuse_handler, rag

# Init fastapi
langfuse = None
if langfuse_handler:
    langfuse = Langfuse()
else:
    print("Starting without Langfuse")


@asynccontextmanager
async def lifespan(app: FastAPIOffline):
    # Operation on startup
    yield  # wait until shutdown
    # Flush all events to be sent to Langfuse on shutdown and terminate all Threads gracefully. This operation is blocking.
    if langfuse:
        langfuse.flush()


app = FastAPIOffline(lifespan=lifespan)

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
