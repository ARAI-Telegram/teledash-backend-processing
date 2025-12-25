#!/usr/bin/env python3
"""
Trigger script for topic modeling task.

Usage:
    python trigger_topic_modeling.py                 # Run topic modeling
    python trigger_topic_modeling.py --update        # Update existing assignments
    python trigger_topic_modeling.py --clear         # Clear existing topics
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from celery import Celery
from common.settings import settings

# Create Celery app
app = Celery("teledash-processing-topic-modeling")
app.config_from_object("config")


def trigger_topic_modeling():
    """Trigger topic modeling task."""
    print("Triggering topic modeling task...")

    result = app.send_task(
        "topic_modeling.init_topic_modeling",
        queue="topic_modeling"
    )

    print(f"Task sent with ID: {result.id}")
    print("Waiting for result...")

    try:
        output = result.get(timeout=3600)  # 1 hour timeout
        print(f"Topic modeling complete: {output}")
    except Exception as e:
        print(f"Error: {e}")


def trigger_update():
    """Trigger topic assignment update."""
    print("Triggering topic assignment update...")

    result = app.send_task(
        "topic_modeling.update_topic_assignments",
        kwargs={"chat_id": None},
        queue="topic_modeling"
    )

    print(f"Task sent with ID: {result.id}")
    print("Waiting for result...")

    try:
        output = result.get(timeout=1800)  # 30 min timeout
        print(f"Update complete: {output}")
    except Exception as e:
        print(f"Error: {e}")


def clear_topics():
    """Clear all topics from database."""
    print("Clearing topics from database...")

    from worker.database import Database

    try:
        db = Database()

        # Delete topics index
        if db.client.indices.exists(index="topics"):
            db.client.indices.delete(index="topics")
            print("Topics index deleted")

        # TODO: Remove topic assignments from messages
        # This would require updating all message documents

        print("Topics cleared successfully")

    except Exception as e:
        print(f"Error clearing topics: {e}")


def main():
    parser = argparse.ArgumentParser(description="Topic Modeling Trigger Script")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing topic assignments"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all topics from database"
    )

    args = parser.parse_args()

    if args.clear:
        confirm = input("Are you sure you want to clear all topics? (yes/no): ")
        if confirm.lower() == "yes":
            clear_topics()
        else:
            print("Cancelled")
    elif args.update:
        trigger_update()
    else:
        trigger_topic_modeling()


if __name__ == "__main__":
    main()
