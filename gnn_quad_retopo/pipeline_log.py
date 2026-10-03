"""Mirror stdout/stderr to runs/<stem>/pipeline.log."""
from __future__ import annotations

import sys
from pathlib import Path


class PipelineLogTee:
    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.log_path, "a", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def write(self, data: str) -> int:
        if not data:
            return 0
        enc = getattr(self._stdout, "encoding", None) or "utf-8"
        try:
            self._stdout.write(data)
        except UnicodeEncodeError:
            self._stdout.write(
                data.encode(enc, errors="replace").decode(enc, errors="replace")
            )
        else:
            # cp932 console may not raise but still fail on some code points
            try:
                data.encode(enc)
            except UnicodeEncodeError:
                self._stdout.write(
                    data.encode(enc, errors="replace").decode(enc, errors="replace")
                )
        self._file.write(data)
        return len(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def fileno(self) -> int:
        return self._stdout.fileno()

    def isatty(self) -> bool:
        return self._stdout.isatty()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


_active_tee: PipelineLogTee | None = None


def install_pipeline_log(out_dir: Path | str) -> Path:
    global _active_tee
    from .run_layout import pipeline_log_path

    path = pipeline_log_path(out_dir)
    _active_tee = PipelineLogTee(path)
    sys.stdout = _active_tee  # type: ignore[assignment]
    return path
