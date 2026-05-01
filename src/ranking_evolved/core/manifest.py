"""Per-run manifest: the small JSON that identifies a run and binds it to git/uv."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Manifest:
    run_id: str
    task: str
    config_path: str
    git_sha: str | None
    git_dirty: bool
    uv_lock_sha256: str | None
    host: str
    platform: str
    python_version: str
    started_at: str
    ended_at: str | None = None
    exit_status: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


def write_manifest(run_dir: Path, manifest: Manifest) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "manifest.json"
    out.write_text(json.dumps(asdict(manifest), indent=2))
    return out


def update_manifest(run_dir: Path, **fields: str | None) -> None:
    path = run_dir / "manifest.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    data.update({k: v for k, v in fields.items() if v is not None})
    path.write_text(json.dumps(data, indent=2))


def build_manifest(
    *,
    run_id: str,
    task: str,
    config_path: str | Path,
    repo_root: Path | None = None,
) -> Manifest:
    return Manifest(
        run_id=run_id,
        task=task,
        config_path=str(config_path),
        git_sha=_git_sha(repo_root),
        git_dirty=_git_dirty(repo_root),
        uv_lock_sha256=_uv_lock_sha(repo_root),
        host=socket.gethostname(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        started_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
    )


def _git_sha(repo_root: Path | None) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root or Path.cwd()),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _git_dirty(repo_root: Path | None) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root or Path.cwd()),
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return False


def _uv_lock_sha(repo_root: Path | None) -> str | None:
    root = repo_root or Path.cwd()
    lock = root / "uv.lock"
    if not lock.exists():
        return None
    h = hashlib.sha256()
    h.update(lock.read_bytes())
    return h.hexdigest()


def make_run_id(task: str) -> str:
    """`<YYYYMMDD_HHMMSS>_<short-hash>` — short, sortable, unique enough."""
    now = _dt.datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    seed = f"{stamp}-{task}-{os.getpid()}".encode()
    short = hashlib.sha256(seed).hexdigest()[:6]
    return f"{stamp}_{short}"
