from datetime import datetime
from typing import List, Optional

from elasticsearch.dsl import Q
from elasticsearch.dsl.query import Query as ESQuery
from fastapi import Query


def parse_message_filter(
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    chat_ids: Optional[List[int]] = Query(None),
) -> Optional[ESQuery]:
    """
    A dependency function for FastAPI path operations that parses query parameters into an
    Elasticsearch query.
    """
    filters = []

    if date_from or date_to:
        date_range_query = {}
        if date_from:
            date_range_query["gte"] = date_from
        if date_to:
            date_range_query["lt"] = date_to
        filters.append(Q("range", **{"date": date_range_query}))

    if chat_ids:
        filters.append(Q("terms", **{"chat_id": chat_ids}))

    return Q("bool", filter=filters) if filters else None
