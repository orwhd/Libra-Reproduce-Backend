#!/usr/bin/env python3
"""One-command headless reproduction entrypoint for Wu, Wang & Evans (Nature 2019)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libra_reproduce.cli import DEFAULT_NATURE_PROMPT, _run_headless  # noqa: E402
import asyncio  # noqa: E402


if __name__ == "__main__":
    out_dir = ROOT / "outputs" / "latest_run"
    mode = "simulation" if "--simulation" in sys.argv else "live"
    code = asyncio.run(
        _run_headless(
            prompt=DEFAULT_NATURE_PROMPT,
            paper_id="paper_A",
            mode=mode,
            output_dir=out_dir,
        )
    )
    raise SystemExit(code)
