"""LLM_RESPONSE_CACHE: litellm's Redis cache, namespaced, every key with a TTL."""

import litellm
import pytest

from cognee.infrastructure.llm.response_cache import enable_llm_response_cache


@pytest.fixture(autouse=True)
def _reset_litellm_cache():
    yield
    litellm.cache = None


def test_off_unless_requested(monkeypatch):
    monkeypatch.delenv("LLM_RESPONSE_CACHE", raising=False)
    assert enable_llm_response_cache() is False
    assert litellm.cache is None


def test_redis_cache_is_namespaced_expiring_and_completion_only_by_default(monkeypatch):
    monkeypatch.setenv("LLM_RESPONSE_CACHE", "redis")
    monkeypatch.setenv("LLM_RESPONSE_CACHE_TTL_SECONDS", "3600")
    assert enable_llm_response_cache() is True

    cache = litellm.cache
    assert cache.ttl == 3600
    assert cache.supported_call_types == ["acompletion"]  # vectors are opt-in
    assert cache.cache.check_and_fix_namespace("abc") == "cognee:llm:abc"


@pytest.mark.parametrize(
    "env", [{"LLM_RESPONSE_CACHE": "memcached"}, {"LLM_RESPONSE_CACHE_TTL_SECONDS": "0"}]
)
def test_rejects_unknown_backend_and_keys_without_expiry(monkeypatch, env):
    monkeypatch.setenv("LLM_RESPONSE_CACHE", "redis")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        enable_llm_response_cache()
