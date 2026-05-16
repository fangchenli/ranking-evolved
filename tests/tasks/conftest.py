"""Keep repo-root task imports stable when pytest targets tests/tasks directly."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root_s = str(REPO_ROOT)
if repo_root_s not in sys.path:
    sys.path.insert(0, repo_root_s)
