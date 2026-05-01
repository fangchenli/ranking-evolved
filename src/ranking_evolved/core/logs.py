"""Per-run rotated loggers.

Three streams: `run.log` (controller INFO+), `proposer.log` (LLM transcripts,
gzipped on rotation), `evaluator.log` (per-iteration eval stdout/stderr).
Sizes are config-bounded; nothing here can silently fill a disk.
"""
from __future__ import annotations

import gzip
import logging
import shutil
from logging.handlers import RotatingFileHandler
from pathlib import Path


class GzipRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler that gzips the rotated file (`.log.1` -> `.log.1.gz`)."""

    def doRollover(self) -> None:  # type: ignore[override]
        super().doRollover()
        for i in range(1, self.backupCount + 1):
            src = Path(f"{self.baseFilename}.{i}")
            if src.exists() and not src.name.endswith(".gz"):
                dst = src.with_name(src.name + ".gz")
                with src.open("rb") as f_in, gzip.open(dst, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                src.unlink()


def make_run_logger(
    name: str,
    log_path: Path,
    *,
    level: str = "INFO",
    max_mb: int = 10,
    backups: int = 5,
    gzip_rotated: bool = False,
) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = [h for h in logger.handlers if not isinstance(h, RotatingFileHandler)]
    handler_cls = GzipRotatingFileHandler if gzip_rotated else RotatingFileHandler
    handler = handler_cls(
        log_path, maxBytes=max_mb * 1024 * 1024, backupCount=backups, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
