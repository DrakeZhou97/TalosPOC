from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """
    Ensure the project root is on sys.path so `import src...` works in all pytest import modes.

    This file lives at `src/tests/conftest.py`, so the project root is 2 levels up.
    """
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
