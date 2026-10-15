"""API mode is a thin HTTP bridge: importing the server must not load the cognee SDK."""

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
PROBE = (
    "import sys; sys.path.insert(0, sys.argv[1]); import server; "
    "assert server.API_MODE; "
    "loaded = [m for m in sys.modules if m == 'cognee' or m.startswith('cognee.')]; "
    "assert not loaded, loaded[:5]; assert 'litellm' not in sys.modules"
)


def test_api_mode_server_imports_no_cognee_modules():
    env = {**os.environ, "API_URL": "http://cognee:8000"}
    subprocess.run([sys.executable, "-c", PROBE, str(SRC)], env=env, check=True)
