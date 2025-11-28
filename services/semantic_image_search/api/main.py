import time
from typing import Optional

import uvicorn
from elasticsearch.dsl.query import Query as ESQuery
from fastapi import Depends, FastAPI, HTTPException, Request

from api.database import APIDatabase
from api.embedding import EmbeddingService
from api.search import SearchService
from common.database.api_validators import parse_message_filter
from common.database.semantic_search_result import SemanticSearchResult
from common.settings import settings
from shared.model_init import init_image_embedding_model

app = FastAPI(title="Image Search API", version="0.1")

# global instances
database: APIDatabase
embedding_service: EmbeddingService
search_service: SearchService


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Welcome"}


@app.on_event("startup")
async def initialize_model_and_database() -> None:
    global database, embedding_service, search_service
    try:
        print("Loading CLIP model...")
        model, processor = init_image_embedding_model()
        embedding_service = EmbeddingService(model, processor)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise
    try:
        database = APIDatabase(connect=True)
        search_service = SearchService(database)
        print("Database connection initialized.")
    except Exception as e:
        print(f"Failed to initialize database connection: {e}")
        raise


@app.get(
    "/search",
    response_description="Return semantic image search results",
    response_model=list[SemanticSearchResult],
)
async def search_images(
    query: str,
    top_k: int = 500,
    filter: Optional[ESQuery] = Depends(parse_message_filter),
):
    """
    Perform text-to-image search.

    Args:
        query: The text query to search for similar images.
        top_k: The number of top results to return (default: 500, max: 10000).

    Returns:
        list[SearchResultEntry]: A list of SearchResultEntry, ordered by score.
    """
    # Validate parameters
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1")
    if top_k > 10000:
        raise HTTPException(status_code=400, detail="top_k cannot exceed 10000")
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    try:
        query_embedding = embedding_service.embed_text(query)

        if not query_embedding:
            raise HTTPException(
                status_code=500, detail="Failed to generate embedding for query"
            )

        # Perform text-to-image search using the Clip Model
        results = await search_service.search_similar_images(
            text_embedding=query_embedding, top_k=top_k, filter=filter
        )
        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.on_event("shutdown")
async def shutdown_db_client() -> None:
    await database.close()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


if __name__ == "__main__":
    uvicorn.run(
        "main:app", host="0.0.0.0", reload=True, port=settings.image_fast_api_port
    )
