"""Skill lookups read Skill nodes by type or id, never the whole graph."""

import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.engine.models import Skill
from cognee.modules.tools import resolve_skills as rs

DATASET = uuid4()


def _node(name, scope=(str(DATASET),), node_type="Skill"):
    return {"id": str(uuid4()), "type": node_type, "name": name, "dataset_scope": list(scope)}


@pytest.mark.asyncio
async def test_resolve_and_find_use_typed_and_id_lookups_only():
    alpha, beta, foreign = _node("alpha"), _node("beta"), _node("gamma", scope=())
    engine = AsyncMock()
    engine.get_nodes_by_type = AsyncMock(return_value=[alpha, beta, foreign])
    engine.get_node = AsyncMock(
        side_effect=lambda i: {"n": _node("x", node_type="NodeSet")}.get(i, beta)
    )

    with patch.object(rs, "_get_graph_engine", AsyncMock(return_value=engine)):
        skills = await rs.resolve_skills(["alpha", "beta", "gamma", "missing"], dataset_id=DATASET)
        assert [s.name for s in skills] == ["alpha", "beta"]
        assert engine.get_nodes_by_type.await_count == 1  # one query for every name

        assert (await rs.find_skill_by_id(beta["id"], dataset_id=DATASET)).name == "beta"
        assert await rs.find_skill_by_id("n", dataset_id=DATASET) is None  # not a Skill node

    engine.get_graph_data.assert_not_called()
    engine.get_nodeset_subgraph.assert_not_called()


@pytest.mark.asyncio
async def test_ladybug_get_nodes_by_type(tmp_path):
    pytest.importorskip("ladybug")
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    adapter = LadybugAdapter(db_path=str(tmp_path / "skills.lbug"))
    for node_id, name, node_type in [("s1", "alpha", "Skill"), ("e1", "alpha", "Entity")]:
        await adapter.query(
            "CREATE (:Node {id: $id, name: $name, type: $type, properties: $props})",
            {
                "id": node_id,
                "name": name,
                "type": node_type,
                "props": json.dumps({"is_active": True}),
            },
        )
    nodes = await adapter.get_nodes_by_type(Skill)
    assert [(n["id"], n["name"], n["is_active"]) for n in nodes] == [("s1", "alpha", True)]
