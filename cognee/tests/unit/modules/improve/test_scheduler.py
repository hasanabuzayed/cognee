"""IMPROVE_SCHEDULED: idle sessions bridge server-side; heavy stages honour their cooldown."""

import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.improve import scheduler
from cognee.modules.improve.config import ImproveConfig
from cognee.modules.session_lifecycle.models import SessionRecord

USER, DATASET = uuid4(), uuid4()


class _Cache:
    def __init__(self):
        self.values = {}

    async def get_value(self, key):
        return self.values.get(key)

    async def set_value(self, key, value, ttl=None):
        self.values[key] = value


@pytest_asyncio.fixture
async def env():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SessionRecord.__table__.create)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def get_async_session():
        async with sessions() as session:
            yield session

    async def touch(session_id, minutes_ago):
        async with sessions() as session:
            row = await session.get(SessionRecord, (session_id, USER))
            when = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
            if row is None:
                session.add(
                    SessionRecord(
                        session_id=session_id,
                        user_id=USER,
                        dataset_id=DATASET,
                        last_activity_at=when,
                    )
                )
            else:
                row.last_activity_at = when
            await session.commit()

    cache = _Cache()
    improve = AsyncMock(return_value=SimpleNamespace(status="completed", lock_held=False))
    config = ImproveConfig(scheduled=True, idle_minutes=30, heavy_cooldown_hours=6)
    with (
        patch(
            "cognee.infrastructure.databases.relational.get_relational_engine",
            return_value=SimpleNamespace(get_async_session=get_async_session),
        ),
        patch(
            "cognee.infrastructure.databases.cache.get_cache_engine.get_cache_engine",
            return_value=cache,
        ),
        patch("cognee.modules.users.methods.get_user", AsyncMock(return_value="user")),
        patch("cognee.api.v1.improve.improve", improve),
        patch.object(scheduler, "get_improve_config", return_value=config),
    ):
        yield SimpleNamespace(touch=touch, cache=cache, improve=improve)
    await engine.dispose()


def _calls(improve):
    calls = [
        (c.kwargs["session_ids"], "deferred_stages" in c.kwargs) for c in improve.call_args_list
    ]
    improve.reset_mock()
    return calls


@pytest.mark.asyncio
async def test_idle_sessions_bridge_once_and_heavy_stages_wait_for_cooldown(env):
    await env.touch("idle", minutes_ago=120)
    await env.touch("active", minutes_ago=1)

    await scheduler.run_due_improves()
    assert _calls(env.improve) == [(["idle"], False)]  # first run is full; active waits

    await scheduler.run_due_improves()
    assert _calls(env.improve) == []  # nothing new since the bridge

    await env.touch("idle", minutes_ago=60)  # new activity, now idle again
    await scheduler.run_due_improves()
    assert _calls(env.improve) == [(["idle"], True)]  # inside the cooldown: heavy deferred

    state = json.loads(env.cache.values[scheduler.STATE_KEY])
    assert state["heavy_pending"] == {str(DATASET): {str(USER): ["idle"]}}
    state["heavy_at"][str(DATASET)] = (datetime.now(timezone.utc) - timedelta(hours=7)).isoformat()
    env.cache.values[scheduler.STATE_KEY] = json.dumps(state)

    await scheduler.run_due_improves()
    assert _calls(env.improve) == [(["idle"], False)]  # cooldown over: deferred work runs
    state = json.loads(env.cache.values[scheduler.STATE_KEY])
    assert state["heavy_pending"] == {}


@pytest.mark.asyncio
async def test_lock_held_retries_next_pass_but_failures_back_off(env):
    await env.touch("idle", minutes_ago=120)
    env.improve.return_value = SimpleNamespace(status="skipped", lock_held=True)
    await scheduler.run_due_improves()
    await scheduler.run_due_improves()
    assert len(_calls(env.improve)) == 2  # lock held: plain retry, no backoff

    env.improve.return_value = SimpleNamespace(status="errored", lock_held=False)
    await scheduler.run_due_improves()
    await scheduler.run_due_improves()
    assert len(_calls(env.improve)) == 1  # errored: backed off, the next pass skips it
    retry = json.loads(env.cache.values[scheduler.STATE_KEY])["retry"]
    assert next(iter(retry.values()))["failures"] == 1

    env.improve.side_effect = RuntimeError("rate limited")
    state = json.loads(env.cache.values[scheduler.STATE_KEY])
    for entry in state["retry"].values():
        entry["after"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    env.cache.values[scheduler.STATE_KEY] = json.dumps(state)
    await scheduler.run_due_improves()
    retry = json.loads(env.cache.values[scheduler.STATE_KEY])["retry"]
    entry = next(iter(retry.values()))
    wait = datetime.fromisoformat(entry["after"]) - datetime.now(timezone.utc)
    assert entry["failures"] == 2 and timedelta(seconds=590) < wait <= timedelta(seconds=600)

    env.improve.side_effect = None
    env.improve.return_value = SimpleNamespace(status="completed", lock_held=False)
    state = json.loads(env.cache.values[scheduler.STATE_KEY])
    for entry in state["retry"].values():
        entry["after"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    env.cache.values[scheduler.STATE_KEY] = json.dumps(state)
    _calls(env.improve)
    await scheduler.run_due_improves()
    assert _calls(env.improve) == [(["idle"], False)]
    assert json.loads(env.cache.values[scheduler.STATE_KEY])["retry"] == {}  # success clears it


def test_router_treats_only_plain_session_bridges_as_hints():
    from cognee.api.v1.improve.routers.get_improve_router import (
        ImprovePayloadDTO,
        _is_schedulable_session_bridge,
    )

    on = ImproveConfig(scheduled=True)
    with patch("cognee.modules.improve.get_improve_config", return_value=on):
        assert _is_schedulable_session_bridge(
            ImprovePayloadDTO(dataset_name="d", session_ids=["s"])
        )
        assert not _is_schedulable_session_bridge(ImprovePayloadDTO(dataset_name="d"))
        assert not _is_schedulable_session_bridge(
            ImprovePayloadDTO(dataset_name="d", session_ids=["s"], node_name=["x"])
        )
    with patch("cognee.modules.improve.get_improve_config", return_value=ImproveConfig()):
        assert not _is_schedulable_session_bridge(
            ImprovePayloadDTO(dataset_name="d", session_ids=["s"])
        )
