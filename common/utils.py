from datetime import datetime, timezone

from .database.index_alias import IndexAlias


def get_chat_id_from_index_name(index_name: str) -> str:
    """
    Extracts the chat ID from the index name.

    Args:
        index_name (str): The name of the index.

    Returns:
        str: The extracted chat ID.
    """
    return index_name.split("_")[
        -1
    ]  # Assuming the chat_id is the last part of the index name


def build_index_name(index_alias: IndexAlias, chat_id: str) -> str:
    """Build an index name from alias, which corresponds to prefix, and chat_id."""
    return f"{index_alias.value}_{chat_id}"


def naive_utcnow() -> datetime:
    """Returns the current UTC time as a naive datetime (without timezone info)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
