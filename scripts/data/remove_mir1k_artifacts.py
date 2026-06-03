#!/usr/bin/env python3
"""Remove local MIR-1K dataset/evaluation artifacts from this workspace.

This only removes known MIR-1K paths created by the report-only dataset
workflow. It refuses to delete arbitrary paths.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
KNOWN_MIR1K_PATHS = [
    REPO_ROOT / "data/external/mir1k",
    REPO_ROOT / "data/manifests/human_singing/mir1k.jsonl",
    REPO_ROOT / "data/manifests/human_singing/mir1k.summary.json",
    REPO_ROOT / "reports/human_singing_eval/mir1k",
]


def remove_path(path: Path, dry_run: bool) -> str:
    if not path.exists():
        return f"missing: {path}"
    if dry_run:
        return f"would remove: {path}"
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return f"removed: {path}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="Actually remove files. Without this, performs a dry run.")
    args = parser.parse_args()
    for path in KNOWN_MIR1K_PATHS:
        print(remove_path(path, dry_run=not args.yes))
    if not args.yes:
        print("\nDry run only. Re-run with --yes to remove these MIR-1K artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
