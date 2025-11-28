import asyncio
from typing import Optional

import numpy as np
import torch
from api.database import Database
from api.model_init import init_embedding_model


def generate_embedding(emb_model, query: str) -> Optional[list[float]]:
    """
    Generate an embedding for a given query.

    Args:
        query (str): The query for which to generate an embedding.

    Returns:
        Optional[List[float]]: The embedding for the query as a list of floats,
        or None if the embedding cannot be generated.

    """
    try:
        embedding = emb_model.encode(query)

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()

        if isinstance(embedding, np.ndarray):
            return embedding.tolist()

    except Exception as e:
        print(f"Failed to generate embedding for query: {e}")

    return None


async def main():
    db = Database(connect=True)

    emb_model = init_embedding_model()

    # query for testing
    search_query = "Der Mond sieht so schön aus"
    print("search query:", search_query)
    query_vector = generate_embedding(emb_model, search_query)

    if query_vector is None:
        print("Failed to generate embedding for query.")
        return

    try:
        await db.find_relevant_messages(query_vector)
    finally:
        await db.close()


asyncio.run(main())
