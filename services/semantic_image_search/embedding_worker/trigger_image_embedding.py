import argparse

from main import app
from worker.database import Database
from worker.tasks import init_image_embedding

from common.database.index_alias import IndexAlias

if __name__ == "__main__":
    db = Database()

    parser = argparse.ArgumentParser(
        description="Initialize image embedding task and optionally delete the Celery queue or vector indices before."
    )
    parser.add_argument(
        "--clear-queue",
        action="store_true",
        help="Clear the image_embedding Celery queue before running.",
    )
    parser.add_argument(
        "--clear-indices",
        action="store_true",
        help="Clear all vectorized image indices in Elasticsearch before running.",
    )
    args = parser.parse_args()

    if args.clear_queue:
        try:
            with app.connection() as conn:
                with conn.channel() as channel:
                    channel.queue_delete("image_embeddings")
                    print("Cleared the image_embedding queue.")
        except Exception as e:
            print(f"Failed to clear image_embedding queue: {e}")

    if args.clear_indices:
        try:
            vectorized_image_indices = db.get_all_indices_by_alias(
                IndexAlias.IMAGE_VECTOR_INDEX_ALIAS
            )
            for index in vectorized_image_indices:
                print("Deletion of index", index)
                db.client.indices.delete(index=index)
            print("Vectorized images indices cleared.")
        except Exception as e:
            print(f"Failed to clear vectorized images indices: {e}")

    print("Triggering embedding task..")
    init_image_embedding.delay()
