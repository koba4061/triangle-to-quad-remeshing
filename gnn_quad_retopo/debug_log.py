"""Debug logging with detailed progress (%, ETA, stage timing)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _fmt_sec(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{int(sec // 60)}m{int(sec % 60):02d}s"
    return f"{int(sec // 3600)}h{int((sec % 3600) // 60):02d}m"


@dataclass
class DebugLog:
    stage: str
    out_dir: Path
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter)
    _step_t0: float = field(default_factory=time.perf_counter)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.debug_dir = self.out_dir / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def _elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def _prefix(self) -> str:
        return f"[{self.stage} +{_fmt_sec(self._elapsed())}]"

    def info(self, msg: str) -> None:
        print(f"{self._prefix()} {msg}", flush=True)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"{self._prefix()} WARN {msg}", flush=True)

    def step(self, msg: str) -> None:
        """主要サブステップ開始。"""
        self._step_t0 = time.perf_counter()
        print(f"{self._prefix()} >>> {msg}", flush=True)

    def progress(
        self,
        current: int,
        total: int,
        label: str = "",
        *,
        every: int | None = None,
    ) -> None:
        """進捗バー風ログ (current/total, %, ETA)。"""
        if total <= 0:
            return
        if every is None:
            every = max(1, total // 50)
        if current % every != 0 and current != total:
            return
        pct = 100.0 * current / total
        dt = time.perf_counter() - self._step_t0
        rate = current / max(dt, 1e-9)
        eta = (total - current) / max(rate, 1e-9)
        tail = f"{current:,}/{total:,} ({pct:.1f}%)"
        if label:
            tail = f"{label} {tail}"
        if current > 0 and current < total:
            tail += f" | step {_fmt_sec(dt)} | ETA {_fmt_sec(eta)}"
        elif current >= total:
            tail += f" | done in {_fmt_sec(dt)}"
        print(f"{self._prefix()} ... {tail}", flush=True)

    def set(self, k: str, v: Any, *, quiet: bool = False) -> None:
        self.stats[k] = v
        if not quiet:
            self.info(f"{k}={v}")

    def done(self, msg: str = "finished") -> None:
        self.info(f"{msg} (total {_fmt_sec(self._elapsed())})")

    def save(self) -> Path:
        self.stats["stage"] = self.stage
        self.stats["elapsed_sec"] = round(self._elapsed(), 3)
        self.stats["warnings"] = self.warnings
        self.stats["finished_at"] = datetime.now(timezone.utc).isoformat()
        p = self.debug_dir / "debug_stats.json"
        p.write_text(json.dumps(self.stats, indent=2, ensure_ascii=False), encoding="utf-8")
        return p


class PipelineTracker:
    """パイプライン全体の段階表示。"""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self._t0 = time.perf_counter()
        self._idx = 0
        self._total = 0

    def configure(self, stages: list[str]) -> None:
        self._stages = stages
        self._total = len(stages)
        self._idx = 0
        print("\n" + "=" * 64, flush=True)
        print(f"[PIPELINE] output={self.out_dir}", flush=True)
        print(f"[PIPELINE] stages: {' -> '.join(stages)}", flush=True)
        print("=" * 64 + "\n", flush=True)

    def enter(self, name: str) -> None:
        self._idx += 1
        elapsed = _fmt_sec(time.perf_counter() - self._t0)
        print(
            f"\n{'=' * 64}\n"
            f"[PIPELINE] Stage {self._idx}/{self._total}: {name}  (pipeline +{elapsed})\n"
            f"{'=' * 64}\n",
            flush=True,
        )

    def finish(self) -> None:
        print(
            f"\n[PIPELINE] ALL DONE in {_fmt_sec(time.perf_counter() - self._t0)} "
            f"| output={self.out_dir}\n",
            flush=True,
        )

    def fail(self, err: str) -> None:
        print(
            f"\n[PIPELINE] FAILED after {_fmt_sec(time.perf_counter() - self._t0)} "
            f"| stage {self._idx}/{self._total} | {err}\n",
            flush=True,
        )
