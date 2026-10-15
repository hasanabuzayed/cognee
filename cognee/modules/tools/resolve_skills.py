"""Dataset-scoped Skill lookup helpers for the agentic retriever."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from cognee.modules.engine.models import Skill
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognee.tools.resolve_skills")


def _skill_in_dataset_scope(skill: Skill, dataset_id: UUID) -> bool:
    return skill.is_active and str(dataset_id) in (skill.dataset_scope or [])


async def resolve_skills(
    skills: Sequence[str | Skill] | None = None,
    *,
    dataset_id: UUID | None = None,
) -> list[Skill]:
    """Resolve explicit skills inside one dataset.

    v1 intentionally rejects unscoped skill lookup and multi-dataset lookup.
    """
    if dataset_id is None:
        raise ValueError("Skill lookup requires one explicit dataset.")

    resolved: list[Skill] = []
    seen_ids = set()
    by_name: dict[str, Skill] | None = None

    for item in skills or []:
        skill = item if isinstance(item, Skill) else None
        if isinstance(item, str):
            if by_name is None:  # one typed lookup serves every requested name
                by_name = _skills_by_name(await _load_skill_nodes(), dataset_id)
            skill = by_name.get(item)
        elif skill is None:
            logger.warning("Skill entries must be Skill or str; got %s", type(item).__name__)

        if skill is None:
            logger.warning("Requested skill not found in dataset %s; skipping", dataset_id)
            continue
        if not _skill_in_dataset_scope(skill, dataset_id):
            logger.warning("Skill %r is outside dataset scope; skipping", skill.name)
            continue
        if skill.id not in seen_ids:
            resolved.append(skill)
            seen_ids.add(skill.id)

    return resolved


async def find_skill_by_id(skill_id: str, *, dataset_id: UUID) -> Skill | None:
    graph_engine = await _get_graph_engine()
    if graph_engine is None:
        return None
    try:
        raw = await graph_engine.get_node(str(skill_id))
    except Exception as exc:
        logger.warning("Skill lookup by id failed: %s", exc, exc_info=True)
        return None
    if not raw or raw.get("type") != "Skill":
        return None
    skill = _coerce_skill(raw)
    if skill is not None and _skill_in_dataset_scope(skill, dataset_id):
        return skill
    return None


async def find_skill_by_name(name: str, *, dataset_id: UUID) -> Skill | None:
    return _skills_by_name(await _load_skill_nodes(), dataset_id).get(name)


def _skills_by_name(raw_nodes, dataset_id: UUID) -> dict[str, Skill]:
    """In-scope skills keyed by name; the first match wins, as with a linear search."""
    by_name: dict[str, Skill] = {}
    for raw in raw_nodes:
        skill = _coerce_skill(raw)
        if skill is not None and _skill_in_dataset_scope(skill, dataset_id):
            by_name.setdefault(skill.name, skill)
    return by_name


async def _get_graph_engine():
    try:
        from cognee.infrastructure.databases.graph import get_graph_engine

        return await get_graph_engine()
    except Exception:
        logger.debug("Graph engine unavailable for skill lookup", exc_info=True)
        return None


async def _load_skill_nodes():
    """All Skill nodes via the typed lookup (native on Ladybug, a scan elsewhere)."""
    graph_engine = await _get_graph_engine()
    if graph_engine is None:
        return []
    try:
        return await graph_engine.get_nodes_by_type(node_type=Skill)
    except Exception as exc:
        logger.warning("Skill lookup by type failed: %s", exc, exc_info=True)
        return []


def _coerce_skill(raw) -> Skill | None:
    if isinstance(raw, Skill):
        return raw
    node_id = None
    if isinstance(raw, (list, tuple)) and len(raw) > 1:
        node_id = raw[0]
        raw = raw[1]
    data = raw.model_dump() if hasattr(raw, "model_dump") else raw
    if not isinstance(data, dict):
        return None
    data = {k: v for k, v in data.items() if k != "metadata"}
    if node_id is not None and "id" not in data:
        data["id"] = node_id
    try:
        return Skill.model_validate(data)
    except Exception:
        logger.debug("Falling back to None after error in _coerce_skill", exc_info=True)
        return None
