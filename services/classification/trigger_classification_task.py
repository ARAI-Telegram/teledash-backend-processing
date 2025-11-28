import argparse

from worker.classification.tasks import init_classification
from worker.database import Database

if __name__ == "__main__":
    database = Database()
    parser = argparse.ArgumentParser(
        description="Initialize classification and optionally remove classification results from all docs."
    )
    parser.add_argument(
        "--clear",
        action="store_true",
    )  # when running the script with flag --clear, all classification related fields are being removed first
    args = parser.parse_args()

    if args.clear:
        database.remove_classification_results()
        print("All classification fields removed from all docs.")

    print("Triggering classification task..")
    init_classification.delay()
