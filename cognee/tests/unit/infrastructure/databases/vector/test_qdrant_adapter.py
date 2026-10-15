"""QDrantAdapter: one shared client, dataset-scoped points, node-set filtering."""

from uuid import uuid4

import pytest

pytest.importorskip("qdrant_client")

from cognee.infrastructure.databases.vector.qdrant.QDrantAdapter import (
    IndexSchema,
    QDrantAdapter,
)


class _Embedder:
    """Deterministic 4-d vectors keyed on the first character."""

    def get_vector_size(self):
        return 4

    async def embed_text(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] if t.startswith("a") else [0.0, 1.0, 0.0, 0.0] for t in texts]


@pytest.mark.asyncio
async def test_shared_client_scoped_search_and_close():
    adapter = QDrantAdapter(
        url=None, api_key=None, embedding_engine=_Embedder(), database_name="ds1"
    )
    other = QDrantAdapter(url=None, api_key=None, embedding_engine=_Embedder(), database_name="ds2")
    # Both datasets must hit the same in-memory store to prove scoping.
    other._client = adapter.get_qdrant_client()

    client = adapter.get_qdrant_client()
    assert adapter.get_qdrant_client() is client  # reused, not rebuilt per call

    await adapter.create_collection("Entity_name")
    a_id, b_id = uuid4(), uuid4()
    await adapter.create_data_points(
        "Entity_name",
        [
            IndexSchema(id=a_id, text="apple", belongs_to_set=["project"]),
            IndexSchema(id=b_id, text="banana", belongs_to_set=["user"]),
        ],
    )
    await other.create_data_points("Entity_name", [IndexSchema(id=uuid4(), text="apricot")])

    results = await adapter.search("Entity_name", query_text="avocado", limit=None)
    assert {r.id for r in results} == {a_id, b_id}  # ds2's point is not visible
    assert results[0].id == a_id  # original graph IDs come back, not scoped storage IDs

    filtered = await adapter.search("Entity_name", query_text="x", limit=None, node_name=["user"])
    assert [r.id for r in filtered] == [b_id]

    assert adapter.get_qdrant_client() is client  # still open after many calls
    await adapter.close()
    assert adapter._client is None
