"""Opt-in Redis cache for model calls, through litellm's own cache.

Every LLM and embedding call goes through ``litellm.acompletion`` /
``litellm.aembedding``, so one ``litellm.cache`` covers them all: a cognify
or improve that re-sends a prompt it already paid for (a retried pipeline, a
re-processed document) gets the stored response instead.

Environment::

    LLM_RESPONSE_CACHE=redis                     # off when unset
    LLM_RESPONSE_CACHE_TTL_SECONDS=86400         # every key expires
    LLM_RESPONSE_CACHE_CALL_TYPES=acompletion    # csv; add aembedding to cache vectors
    LLM_RESPONSE_CACHE_NAMESPACE=cognee:llm

The Redis server is the cache layer's (``CACHE_HOST`` / ``CACHE_PORT`` /
``CACHE_PASSWORD``). Embeddings are opt-in: one 3072-d vector is tens of KB,
which adds up fast on a Redis shared with other services under ``noeviction``.
Completions are small text, and the TTL bounds both.
"""

import os

from cognee.shared.logging_utils import get_logger

logger = get_logger("llm.response_cache")


def enable_llm_response_cache() -> bool:
    """Install the Redis cache on litellm when ``LLM_RESPONSE_CACHE=redis``."""
    backend = os.getenv("LLM_RESPONSE_CACHE", "").strip().lower()
    if not backend:
        return False
    if backend != "redis":
        raise ValueError(f"LLM_RESPONSE_CACHE supports only 'redis', got {backend!r}")

    import litellm
    from litellm.caching.caching import Cache

    from cognee.infrastructure.databases.cache.config import get_cache_config

    ttl = int(os.getenv("LLM_RESPONSE_CACHE_TTL_SECONDS", "86400"))
    if ttl <= 0:
        raise ValueError("LLM_RESPONSE_CACHE_TTL_SECONDS must be > 0: keys must expire")
    call_types = [
        part.strip()
        for part in os.getenv("LLM_RESPONSE_CACHE_CALL_TYPES", "acompletion").split(",")
        if part.strip()
    ]
    namespace = os.getenv("LLM_RESPONSE_CACHE_NAMESPACE", "cognee:llm")
    config = get_cache_config()

    litellm.cache = Cache(
        type="redis",
        host=config.cache_host,
        port=str(config.cache_port),
        password=config.cache_password,
        namespace=namespace,
        ttl=ttl,
        supported_call_types=call_types,
    )
    logger.info(
        "LLM response cache on: redis %s:%s, namespace %s, ttl %ss, calls %s",
        config.cache_host,
        config.cache_port,
        namespace,
        ttl,
        ",".join(call_types),
    )
    return True
