"""Ladybug's native iter_bounded_neighborhood stops at max_nodes instead of reading
the whole neighbourhood (graph views seed from hubs, where two hops reach everything)."""

import json
from unittest.mock import patch

import pytest

from cognee.tests.utils.assert_bounded_neighborhood_contract import (
    assert_bounded_neighborhood_contract,
)

pytest.importorskip("ladybug")


async def _hub_graph(tmp_path):
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    adapter = LadybugAdapter(db_path=str(tmp_path / "hubs.lbug"))
    nodes, edges = {"hub_a": "Hub", "hub_b": "Hub"}, []
    for hub in ("hub_a", "hub_b"):
        for i in range(30):
            spoke = f"{hub}_s{i}"
            nodes[spoke] = "Entity"
            edges.append((hub, spoke))
            for j in range(5):
                leaf = f"{spoke}_l{j}"
                nodes[leaf] = "Entity"
                edges.append((spoke, leaf))
    for node_id, node_type in nodes.items():
        await adapter.query(
            "CREATE (:Node {id: $id, name: $id, type: $type, properties: $p})",
            {
                "id": node_id,
                "type": node_type,
                "p": json.dumps({"description": f"about {node_id}"}),
            },
        )
    for source, target in edges:
        await adapter.query(
            "MATCH (a:Node {id: $s}), (b:Node {id: $t}) "
            "CREATE (a)-[:EDGE {relationship_name: 'rel', properties: '{}'}]->(b)",
            {"s": source, "t": target},
        )
    return adapter, [(s, t, "rel") for s, t in edges]


@pytest.mark.asyncio
async def test_ladybug_bounded_neighborhood_stops_at_max_nodes(tmp_path):
    adapter, graph_edges = await _hub_graph(tmp_path)
    seeds = ["hub_a", "missing", "hub_b"]

    with patch.object(type(adapter), "get_neighborhood", side_effect=AssertionError("full read")):
        chunks = [c async for c in adapter.iter_bounded_neighborhood(seeds, 2, 40, chunk_size=15)]

    members, _ = assert_bounded_neighborhood_contract(
        chunks, max_nodes=40, chunk_size=15, seed_ids=["hub_a", "hub_b"], graph_edges=graph_edges
    )
    assert len(members) == 40 and members[:2] == ["hub_a", "hub_b"]
    # Round robin: both hubs' spokes are in, not 38 from one hub.
    assert sum(m.startswith("hub_a_") for m in members) == sum(
        m.startswith("hub_b_") for m in members
    )

    projected = [
        c async for c in adapter.iter_bounded_neighborhood(["hub_a"], 1, 5, property_keys=[])
    ]
    assert all(set(props) <= {"name", "type"} for _, props in projected[0][0])
