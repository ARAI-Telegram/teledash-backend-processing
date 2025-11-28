from worker.database import Database

if __name__ == "__main__":
    db = Database()
    try:
        db.clear_pending_states()
    except Exception as e:
        print(f"An error occurred while removing pending states: {e}")
