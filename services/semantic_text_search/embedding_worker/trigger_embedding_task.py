import argparse

from common.database.index_alias import IndexAlias
from main import app
from worker.database import Database
from worker.tasks import init_embeddings

if __name__ == "__main__":
    db = Database()

    parser = argparse.ArgumentParser(
        description="Initialize semantic embedding task and optionally delete the Celery queue or vector indices before."
    )
    parser.add_argument(
        "--clear-queue",
        action="store_true",
        help="Clear the text embeddings Celery queue before running.",
    )
    parser.add_argument(
        "--clear-indices",
        action="store_true",
        help="Clear all vectorized message indices in Elasticsearch before running.",
    )
    args = parser.parse_args()

    if args.clear_queue:
        try:
            with app.connection() as conn:  # TODO: create common method
                with conn.channel() as channel:
                    channel.queue_delete("text-embeddings")
            print("Text embeddings Celery queue cleared.")
        except Exception as e:
            print(f"Failed to clear text embeddings Celery queue: {e}")

    if args.clear_indices:
        try:
            vectorized_message_indices = db.get_all_indices_by_alias(
                IndexAlias.TEXT_VECTOR_INDEX_ALIAS
            )
            for index in vectorized_message_indices:
                print("Deletion of index", index)
                db.client.indices.delete(index=index)
            print("Vectorized text messages indices cleared.")
        except Exception as e:
            print(f"Failed to clear vectorized text messages indices: {e}")

    print("Triggering embedding task..")
    init_embeddings.delay()
