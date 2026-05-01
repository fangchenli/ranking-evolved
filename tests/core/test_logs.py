"""Tests for core.logs: rotation respects size cap; gzip flag compresses rotated files."""
from __future__ import annotations

from pathlib import Path

from ranking_evolved.core.logs import make_run_logger


def test_logger_rotates_at_size_cap(tmp_path: Path, record_io):
    log_path = tmp_path / "run.log"

    def run() -> dict:
        logger = make_run_logger(
            "test.run", log_path,
            level="INFO", max_mb=1, backups=2, gzip_rotated=False,
        )
        # 1 MB cap; write more than that
        big = "x" * 4096
        for _ in range(400):  # ~1.6 MB of payload
            logger.info(big)
        for h in logger.handlers:
            h.flush()
            h.close()
        files = sorted(p.name for p in tmp_path.iterdir())
        return {"files": files}

    out = record_io(
        module="src/ranking_evolved/core/logs.py",
        function="make_run_logger",
        input={"max_mb": 1, "backups": 2, "writes": 400, "payload_bytes": 4096},
        run=run,
    )
    # Expect at least one rotation: run.log + run.log.1
    assert "run.log" in out["files"]
    assert any(name.startswith("run.log.") for name in out["files"])


def test_logger_gzips_rotated(tmp_path: Path, record_io):
    log_path = tmp_path / "proposer.log"

    def run() -> dict:
        logger = make_run_logger(
            "test.proposer", log_path,
            level="INFO", max_mb=1, backups=2, gzip_rotated=True,
        )
        big = "y" * 4096
        for _ in range(400):
            logger.info(big)
        for h in logger.handlers:
            h.flush()
            h.close()
        files = sorted(p.name for p in tmp_path.iterdir())
        return {"files": files, "has_gz": any(f.endswith(".gz") for f in files)}

    out = record_io(
        module="src/ranking_evolved/core/logs.py",
        function="GzipRotatingFileHandler.doRollover",
        input={"max_mb": 1, "backups": 2, "gzip_rotated": True},
        run=run,
    )
    assert out["has_gz"] is True
