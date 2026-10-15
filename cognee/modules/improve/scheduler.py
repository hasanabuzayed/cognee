"""Server-owned session bridging (``IMPROVE_SCHEDULED=true``).

Session writes are instant cache appends; turning them into graph content is
the server's call, not the client's (the deriver / dream split Honcho uses).
Every session write already bumps ``session_records.last_activity_at``, so the
queue is that table: a session is due once it has been idle for
``IMPROVE_IDLE_MINUTES`` and has activity newer than its last bridge. Due
sessions are bridged one (user, dataset) group at a time, which keeps one
improve in flight and memory bounded; the improve lock and the stages'
watermarks make a repeat run cheap.

``IMPROVE_HEAVY_COOLDOWN_HOURS`` > 0 runs the LLM-drafting stages at most once
per dataset per window. In between they are deferred, and the sessions that
skipped them are kept so the next heavy run covers them; their watermarks pick
up everything that accumulated.

State (bridge markers, heavy timestamps, deferred sessions) is one small JSON
document in the cache store, so it survives restarts. Only this loop writes it.
Markers older than the session TTL are dropped, as the sessions they track are.
"""

import asyncio
import json
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import select

from cognee.shared.logging_utils import get_logger

from .config import get_improve_config

logger = get_logger("improve.scheduler")

POLL_SECONDS = 60
STATE_KEY = "improve_scheduler:v1"
HEAVY_STAGES = ("extract_agent_context", "distill_sessions")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _session_ttl_seconds() -> int:
    from cognee.infrastructure.databases.cache.config import get_cache_config

    return get_cache_config().session_ttl_seconds or 604800


async def _load_state(cache) -> dict:
    raw = await cache.get_value(STATE_KEY)
    state = json.loads(raw) if raw else {}
    state.setdefault("bridged", {})
    state.setdefault("heavy_at", {})
    state.setdefault("heavy_pending", {})
    return state


async def _save_state(cache, state: dict) -> None:
    oldest = (_now() - timedelta(seconds=_session_ttl_seconds())).isoformat()
    state["bridged"] = {k: v for k, v in state["bridged"].items() if v >= oldest}
    await cache.set_value(STATE_KEY, json.dumps(state))


async def _due_sessions(
    state: dict, idle_minutes: float
) -> tuple[dict[tuple[str, str], list[str]], dict[str, str]]:
    """Idle sessions with activity newer than their last bridge, grouped by (user, dataset),
    plus the activity timestamp each one is being bridged up to."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.session_lifecycle.models import SessionRecord

    now = _now()
    query = select(
        SessionRecord.user_id,
        SessionRecord.dataset_id,
        SessionRecord.session_id,
        SessionRecord.last_activity_at,
    ).where(
        SessionRecord.dataset_id.isnot(None),
        SessionRecord.last_activity_at <= now - timedelta(minutes=idle_minutes),
        SessionRecord.last_activity_at >= now - timedelta(seconds=_session_ttl_seconds()),
    )
    async with get_relational_engine().get_async_session() as session:
        rows = (await session.execute(query)).all()

    due: dict[tuple[str, str], list[str]] = {}
    activity: dict[str, str] = {}
    for user_id, dataset_id, session_id, last_activity_at in rows:
        key = f"{user_id}:{session_id}"
        activity[key] = _aware(last_activity_at).isoformat()
        if state["bridged"].get(key, "") < activity[key]:
            due.setdefault((str(user_id), str(dataset_id)), []).append(session_id)
    return due, activity


async def run_due_improves() -> None:
    """One scheduler pass: bridge every due session group, oldest state first."""
    from cognee.api.v1.improve import improve
    from cognee.infrastructure.databases.cache.get_cache_engine import get_cache_engine
    from cognee.modules.users.methods import get_user

    config = get_improve_config()
    cache = get_cache_engine()
    if cache is None:
        logger.warning("improve scheduler: no cache engine, sessions cannot be bridged")
        return

    state = await _load_state(cache)
    due, activity = await _due_sessions(state, config.idle_minutes)
    # Datasets whose deferred heavy work came off cooldown with no new sessions.
    for dataset_id, users in state["heavy_pending"].items():
        for user_id in users:
            due.setdefault((user_id, dataset_id), [])

    cooldown = timedelta(hours=config.heavy_cooldown_hours)
    for (user_id, dataset_id), session_ids in due.items():
        pending = state["heavy_pending"].get(dataset_id, {}).get(user_id, [])
        heavy_at = state["heavy_at"].get(dataset_id)
        heavy_due = (
            not cooldown
            or heavy_at is None
            or _now() - _aware(datetime.fromisoformat(heavy_at)) >= cooldown
        )
        run_ids = list(dict.fromkeys([*session_ids, *(pending if heavy_due else [])]))
        if not run_ids:
            continue

        try:
            user = await get_user(UUID(user_id))
            started_at = _now()
            result = await improve(
                dataset=UUID(dataset_id),
                session_ids=run_ids,
                user=user,
                **({} if heavy_due else {"deferred_stages": list(HEAVY_STAGES)}),
            )
        except Exception:
            logger.exception("improve scheduler: bridge failed for dataset %s", dataset_id)
            continue
        if getattr(result, "lock_held", False) or result.status == "errored":
            continue  # markers stay put, so the next pass retries

        for session_id in session_ids:
            key = f"{user_id}:{session_id}"
            state["bridged"][key] = activity[key]
        users = state["heavy_pending"].setdefault(dataset_id, {})
        if heavy_due:
            if cooldown:
                state["heavy_at"][dataset_id] = started_at.isoformat()
            users.pop(user_id, None)
        else:
            users[user_id] = list(dict.fromkeys([*pending, *session_ids]))
        if not users:
            state["heavy_pending"].pop(dataset_id, None)
        await _save_state(cache, state)
        logger.info(
            "improve scheduler: bridged %d session(s) into %s (%s)",
            len(run_ids),
            dataset_id,
            "full" if heavy_due else "heavy stages deferred",
        )


async def _loop() -> None:
    while True:
        try:
            await run_due_improves()
        except Exception:
            logger.exception("improve scheduler pass failed; retrying next poll")
        await asyncio.sleep(POLL_SECONDS)


@asynccontextmanager
async def improve_scheduler_service():
    """Bridge idle sessions in the background until API shutdown (no-op unless enabled)."""
    if not get_improve_config().scheduled:
        yield
        return
    task = asyncio.create_task(_loop())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
