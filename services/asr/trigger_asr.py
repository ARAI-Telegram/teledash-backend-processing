import argparse

from worker.database import Database
from worker.tasks import init_asr

if __name__ == "__main__":
    database = Database()
    parser = argparse.ArgumentParser(description="Initialize asr.")

    parser.add_argument(
        "--clear",
        action="store_true",
    )
    args = parser.parse_args()

    if args.clear:
        database.remove_all_transcription_fields()
        print("All transcription fields removed from all docs.")

    print("Triggering init of asr task manually")

    init_asr.delay()
