"""GLINER_REMOTE_URL: extraction goes to a GLiNER2 service; nothing local is needed."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cognee.modules.cognify.config import (
    GLINER_DEMO_EXTRACTOR,
    CognifyConfig,
    ensure_extractor_runtime,
)
from cognee.tasks.graph.gliner_demo import extractor as ex
from cognee.tasks.graph.gliner_demo.schema import GlinerSchema

SCHEMA = GlinerSchema({"person": "a person"}, {"works_at": "employment"})


@pytest.fixture
def remote(monkeypatch):
    config = CognifyConfig(gliner_remote_url="http://gliner2:8080/", gliner_remote_timeout=9)
    monkeypatch.setattr(ex, "get_cognify_config", lambda: config)
    monkeypatch.setattr(ex, "_extractors", {})
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        body = (
            {"results": [{"entities": {"person": ["Ada"]}}]} if "batch" in url else {"result": {}}
        )
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

    with patch("httpx.post", fake_post):
        yield calls


def test_remote_extractor_sends_cognees_schema_and_options(remote):
    ex.require_gliner2()  # no local gliner2 needed
    handle = ex.load_extractor("fastino/gliner2.5-base-v1")
    assert isinstance(handle, ex.RemoteExtractor) and ex.load_extractor() is handle

    assert ex.extract_batch(handle, ["Ada works at IBM"], SCHEMA, threshold=0.4) == [
        {"entities": {"person": ["Ada"]}}
    ]
    assert ex.extract_once(handle, "Ada", SCHEMA) == {}
    (batch_url, batch, timeout), (once_url, once, _) = remote
    assert batch_url == "http://gliner2:8080/extract_batch" and timeout == 9
    assert batch["model"] == "fastino/gliner2.5-base-v1"
    assert batch["entity_types"] == {"person": "a person"}
    assert batch["relation_types"] == {"works_at": "employment"}
    assert (batch["threshold"], batch["window_words"], batch["window_overlap_words"]) == (
        0.4,
        384,
        64,
    )
    assert once_url.endswith("/extract_once") and once["text"] == "Ada"


def test_remote_mode_never_installs_the_runtime():
    config = CognifyConfig(gliner_remote_url="http://gliner2:8080", gliner_auto_install=False)
    with patch(
        "cognee.tasks.graph.gliner_demo.install.gliner_runtime_installed", return_value=False
    ):
        asyncio.run(ensure_extractor_runtime(GLINER_DEMO_EXTRACTOR, config))  # no raise, no install
