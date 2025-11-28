from common.database.database import DatabaseAsync
from common.database.index_alias import IndexAlias


class APIDatabase(DatabaseAsync):
    """
    Database class for asynchronous operations in the API context.
    """

    def __init__(self, connect: bool = True) -> None:
        super().__init__(connect=connect)
        self.service_index_alias = IndexAlias.IMAGE_VECTOR_INDEX_ALIAS
